"""
Hail Hero Celery Application - Background Task Processing System

This module provides comprehensive background task processing for the Hail Hero platform,
including NOAA data ingestion, lead enrichment, CRM synchronization, messaging,
and scheduled maintenance tasks.

Features:
- Async processing of hail event data
- Background lead enrichment and scoring
- GoHighLevel CRM synchronization
- Twilio messaging queue management
- Periodic data processing and cleanup
- Comprehensive error handling and retry logic
- Redis-backed queue management
- Task monitoring and metrics
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import traceback

from celery import Celery, Task
from celery.signals import (
    task_prerun, task_postrun, task_failure, task_success,
    worker_ready, worker_init, celeryd_after_setup
)
from celery.exceptions import Retry, MaxRetriesExceededError
from celery.schedules import crontab

# Import Hail Hero modules
from .config import get_config, Config
from .noaa_integration.noaa_storm_events_integration import (
    NOAAStormEventsIntegration, create_noaa_integration, EventSeverity
)
from .integrations.gohighlevel.client import (
    GoHighLevelClient, Contact, ContactStatus, PipelineStage
)
from .property_enrichment.property_enrichment_engine import PropertyEnrichmentEngine
from .compliance.models import ConsentManager, DataProvenance

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize Celery app
celery_app = Celery(
    'hailhero',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    include=['src.celery_app']
)

# Celery configuration
celery_app.conf.update(
    # Task configuration
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    result_expires=3600,  # 1 hour
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Rate limiting
    task_annotations={
        'tasks.process_noaa_data': {'rate_limit': '10/m'},
        'tasks.send_twilio_message': {'rate_limit': '30/m'},
        'tasks.sync_to_gohighlevel': {'rate_limit': '20/m'},
    },
    
    # Retry configuration
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Broker connection
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=5,
    
    # Result backend
    result_backend_transport_options={
        'retry_policy': {
            'timeout': 5.0
        }
    },
    
    # Timezones
    timezone='UTC',
    enable_utc=True,
    
    # Security
    security_key=os.getenv('CELERY_SECURITY_KEY', 'default-secret-key'),
    
    # Beat scheduler
    beat_scheduler='celery.beat.PersistentScheduler',
    beat_schedule_filename='celerybeat-schedule',
    
    # Task routes
    task_routes={
        'tasks.process_noaa_data': {'queue': 'noaa'},
        'tasks.enrich_lead_data': {'queue': 'enrichment'},
        'tasks.send_twilio_message': {'queue': 'messaging'},
        'tasks.sync_to_gohighlevel': {'queue': 'crm'},
        'tasks.cleanup_old_data': {'queue': 'maintenance'},
    }
)

# Task state tracking
class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"

@dataclass
class TaskResult:
    """Standard task result format."""
    task_id: str
    task_name: str
    state: TaskState
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessingMetrics:
    """Processing metrics for tasks."""
    tasks_processed: int = 0
    tasks_successful: int = 0
    tasks_failed: int = 0
    tasks_retried: int = 0
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.tasks_processed == 0:
            return 0.0
        return self.tasks_successful / self.tasks_processed

# Base task class with common functionality
class HailHeroTask(Task):
    """Base task class for Hail Hero with enhanced error handling and metrics."""
    
    def __init__(self):
        self.metrics = ProcessingMetrics()
        self.start_time = None
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        """Execute the task with timing and error handling."""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting task {self.name} with args: {args}, kwargs: {kwargs}")
        
        try:
            result = super().__call__(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(f"Task {self.name} retrying: {exc}")
        self.metrics.tasks_retried += 1
        super().on_retry(exc, task_id, args, kwargs, einfo)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {self.name} failed: {exc}")
        self.metrics.tasks_failed += 1
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {self.name} completed successfully")
        self.metrics.tasks_successful += 1
        super().on_success(retval, task_id, args, kwargs)
    
    def _record_success(self):
        """Record successful task completion."""
        if self.start_time:
            processing_time = (datetime.utcnow() - self.start_time).total_seconds()
            self.metrics.total_processing_time += processing_time
            self.metrics.tasks_processed += 1
            self.metrics.average_processing_time = (
                self.metrics.total_processing_time / self.metrics.tasks_processed
            )
    
    def _record_failure(self, exc):
        """Record task failure."""
        self.metrics.tasks_failed += 1
        self.metrics.tasks_processed += 1

# Signal handlers for monitoring
@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **kwds):
    """Handle task pre-run."""
    logger.info(f"Task {task.name} starting execution")

@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **kwds):
    """Handle task post-run."""
    logger.info(f"Task {task.name} completed with state: {state}")

@task_failure.connect
def task_failure_handler(task_id, exc, einfo, **kwds):
    """Handle task failure."""
    logger.error(f"Task {task_id} failed: {exc}")

@task_success.connect
def task_success_handler(sender, **kwds):
    """Handle task success."""
    logger.info(f"Task {sender.name} completed successfully")

@worker_ready.connect
def worker_ready_handler(**kwds):
    """Handle worker ready."""
    logger.info("Celery worker is ready")

@worker_init.connect
def worker_init_handler(**kwds):
    """Handle worker initialization."""
    logger.info("Celery worker initializing")

# Periodic task schedule
celery_app.conf.beat_schedule = {
    'process-noaa-data-daily': {
        'task': 'tasks.process_noaa_data_daily',
        'schedule': crontab(hour=2, minute=0),  # 2:00 AM daily
        'options': {'queue': 'noaa'},
    },
    'enrich-leads-hourly': {
        'task': 'tasks.enrich_pending_leads',
        'schedule': crontab(minute=0),  # Every hour
        'options': {'queue': 'enrichment'},
    },
    'sync-crm-every-30-minutes': {
        'task': 'tasks.sync_crm_data',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
        'options': {'queue': 'crm'},
    },
    'cleanup-old-data-daily': {
        'task': 'tasks.cleanup_old_data',
        'schedule': crontab(hour=3, minute=0),  # 3:00 AM daily
        'options': {'queue': 'maintenance'},
    },
    'health-check-every-5-minutes': {
        'task': 'tasks.health_check',
        'schedule': crontab(minute='*/5'),  # Every 5 minutes
        'options': {'queue': 'maintenance'},
    },
}

# Task implementations
@celery_app.task(
    base=HailHeroTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def process_noaa_data(self, start_date: str, end_date: str, 
                     target_states: Optional[List[str]] = None,
                     min_hail_size: float = 0.5) -> TaskResult:
    """
    Process NOAA storm events data for hail events.
    
    Args:
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        target_states: List of states to filter by
        min_hail_size: Minimum hail size in inches
        
    Returns:
        TaskResult with processing statistics
    """
    try:
        logger.info(f"Processing NOAA data from {start_date} to {end_date}")
        
        # Initialize NOAA integration
        integration = asyncio.run(create_noaa_integration())
        
        # Fetch hail events
        hail_events = asyncio.run(integration.get_hail_events_only(
            start_date=start_date,
            end_date=end_date,
            min_hail_size=min_hail_size,
            target_states=target_states
        ))
        
        logger.info(f"Found {len(hail_events)} hail events")
        
        # Process events and create leads
        processed_leads = []
        for event in hail_events:
            try:
                # Enrich event data and create lead
                lead_data = {
                    'event_id': event.event_id,
                    'geometry': asdict(event.geometry),
                    'severity': event.geometry.severity.value,
                    'risk_score': event.geometry.risk_score,
                    'processed_at': event.processed_at.isoformat(),
                    'data_quality_score': event.data_quality_score,
                    'source': 'noaa_api'
                }
                
                # Queue lead enrichment
                enrich_lead_data.delay(lead_data)
                processed_leads.append(lead_data)
                
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")
                continue
        
        # Close integration
        asyncio.run(integration.close())
        
        result_data = {
            'events_processed': len(hail_events),
            'leads_generated': len(processed_leads),
            'processing_time': (datetime.utcnow() - self.start_time).total_seconds(),
            'date_range': {'start': start_date, 'end': end_date},
            'filter_criteria': {
                'target_states': target_states,
                'min_hail_size': min_hail_size
            }
        }
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.SUCCESS,
            result=result_data,
            started_at=self.start_time,
            completed_at=datetime.utcnow(),
            metadata=result_data
        )
        
    except Exception as e:
        logger.error(f"Error in NOAA data processing: {e}")
        
        # Retry on transient errors
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < self.max_retries:
            raise self.retry(countdown=60 ** (self.request.retries + 1))
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.FAILURE,
            error=str(e),
            started_at=self.start_time,
            completed_at=datetime.utcnow()
        )

@celery_app.task(
    base=HailHeroTask,
    bind=True,
    max_retries=3,
    default_retry_delay=30
)
def enrich_lead_data(self, lead_data: Dict[str, Any]) -> TaskResult:
    """
    Enrich lead data with property information and scoring.
    
    Args:
        lead_data: Basic lead information from NOAA processing
        
    Returns:
        TaskResult with enriched lead data
    """
    try:
        logger.info(f"Enriching lead data for event {lead_data.get('event_id')}")
        
        # Initialize property enrichment engine
        enricher = PropertyEnrichmentEngine()
        
        # Extract location information
        geometry = lead_data.get('geometry', {})
        lat = geometry.get('begin_lat')
        lon = geometry.get('begin_lon')
        
        if not lat or not lon:
            raise ValueError("Missing latitude/longitude in lead data")
        
        # Enrich property data
        enriched_data = asyncio.run(enricher.enrich_property(
            latitude=lat,
            longitude=lon,
            event_data=lead_data
        ))
        
        # Calculate lead score
        lead_score = calculate_lead_score(enriched_data)
        enriched_data['lead_score'] = lead_score
        
        # Queue CRM synchronization
        sync_to_gohighlevel.delay(enriched_data)
        
        result_data = {
            'lead_id': enriched_data.get('lead_id'),
            'lead_score': lead_score,
            'property_value': enriched_data.get('property_value'),
            'owner_info': enriched_data.get('owner_info', {}),
            'enrichment_fields': list(enriched_data.keys()),
            'processing_time': (datetime.utcnow() - self.start_time).total_seconds()
        }
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.SUCCESS,
            result=result_data,
            started_at=self.start_time,
            completed_at=datetime.utcnow(),
            metadata=result_data
        )
        
    except Exception as e:
        logger.error(f"Error enriching lead data: {e}")
        
        # Retry on transient errors
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < self.max_retries:
            raise self.retry(countdown=30 ** (self.request.retries + 1))
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.FAILURE,
            error=str(e),
            started_at=self.start_time,
            completed_at=datetime.utcnow()
        )

@celery_app.task(
    base=HailHeroTask,
    bind=True,
    max_retries=3,
    default_retry_delay=30
)
def send_twilio_message(self, to_phone: str, message: str, 
                       message_type: str = 'sms') -> TaskResult:
    """
    Send Twilio message (SMS/MMS).
    
    Args:
        to_phone: Recipient phone number
        message: Message content
        message_type: Type of message ('sms' or 'mms')
        
    Returns:
        TaskResult with message sending status
    """
    try:
        logger.info(f"Sending {message_type} message to {to_phone}")
        
        # Import Twilio client (lazy import to avoid circular dependencies)
        from .integrations.twilio.client import TwilioClient
        
        # Initialize Twilio client
        client = TwilioClient()
        
        # Send message
        message_result = asyncio.run(client.send_message(
            to_phone=to_phone,
            message=message,
            message_type=message_type
        ))
        
        result_data = {
            'message_sid': message_result.get('sid'),
            'status': message_result.get('status'),
            'to_phone': to_phone,
            'message_type': message_type,
            'sent_at': datetime.utcnow().isoformat(),
            'processing_time': (datetime.utcnow() - self.start_time).total_seconds()
        }
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.SUCCESS,
            result=result_data,
            started_at=self.start_time,
            completed_at=datetime.utcnow(),
            metadata=result_data
        )
        
    except Exception as e:
        logger.error(f"Error sending Twilio message: {e}")
        
        # Retry on transient errors
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < self.max_retries:
            raise self.retry(countdown=30 ** (self.request.retries + 1))
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.FAILURE,
            error=str(e),
            started_at=self.start_time,
            completed_at=datetime.utcnow()
        )

@celery_app.task(
    base=HailHeroTask,
    bind=True,
    max_retries=3,
    default_retry_delay=30
)
def sync_to_gohighlevel(self, lead_data: Dict[str, Any]) -> TaskResult:
    """
    Synchronize lead data to GoHighLevel CRM.
    
    Args:
        lead_data: Enriched lead data
        
    Returns:
        TaskResult with sync status
    """
    try:
        logger.info(f"Syncing lead to GoHighLevel: {lead_data.get('lead_id')}")
        
        # Initialize GoHighLevel client
        client = GoHighLevelClient()
        
        # Convert lead data to GoHighLevel contact
        contact = convert_lead_to_contact(lead_data)
        
        # Check if contact exists
        existing_contacts = client.search_contacts(
            query=contact.phone or contact.email,
            limit=1
        )
        
        if existing_contacts:
            # Update existing contact
            updated_contact = client.update_contact(
                contact_id=existing_contacts[0].id,
                contact=contact
            )
            action = 'updated'
        else:
            # Create new contact
            created_contact = client.create_contact(contact)
            action = 'created'
        
        result_data = {
            'contact_id': updated_contact.id if action == 'updated' else created_contact.id,
            'action': action,
            'lead_id': lead_data.get('lead_id'),
            'synced_at': datetime.utcnow().isoformat(),
            'processing_time': (datetime.utcnow() - self.start_time).total_seconds()
        }
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.SUCCESS,
            result=result_data,
            started_at=self.start_time,
            completed_at=datetime.utcnow(),
            metadata=result_data
        )
        
    except Exception as e:
        logger.error(f"Error syncing to GoHighLevel: {e}")
        
        # Retry on transient errors
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < self.max_retries:
            raise self.retry(countdown=30 ** (self.request.retries + 1))
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.FAILURE,
            error=str(e),
            started_at=self.start_time,
            completed_at=datetime.utcnow()
        )

@celery_app.task(
    base=HailHeroTask,
    bind=True,
    max_retries=2,
    default_retry_delay=300
)
def cleanup_old_data(self, days_to_keep: int = 90) -> TaskResult:
    """
    Clean up old data and perform maintenance tasks.
    
    Args:
        days_to_keep: Number of days to keep data
        
    Returns:
        TaskResult with cleanup statistics
    """
    try:
        logger.info(f"Cleaning up data older than {days_to_keep} days")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Clean up old task results
        from .celery_app import celery_app
        inspector = celery_app.control.inspect()
        
        # Get task statistics
        stats = {
            'cutoff_date': cutoff_date.isoformat(),
            'tasks_cleaned': 0,
            'logs_cleaned': 0,
            'cache_cleared': False,
            'processing_time': (datetime.utcnow() - self.start_time).total_seconds()
        }
        
        # Clear expired cache entries
        try:
            from .config import get_noaa_config
            noaa_config = get_noaa_config()
            if noaa_config.enable_caching:
                # Clear cache (this would be implemented in the NOAA integration)
                stats['cache_cleared'] = True
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
        
        # Clean up old logs (implement based on your logging setup)
        # This is a placeholder for log cleanup logic
        
        result_data = {
            'cleanup_stats': stats,
            'items_processed': stats['tasks_cleaned'] + stats['logs_cleaned'],
            'processing_time': stats['processing_time']
        }
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.SUCCESS,
            result=result_data,
            started_at=self.start_time,
            completed_at=datetime.utcnow(),
            metadata=result_data
        )
        
    except Exception as e:
        logger.error(f"Error in data cleanup: {e}")
        
        # Retry on transient errors
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < self.max_retries:
            raise self.retry(countdown=300)
        
        return TaskResult(
            task_id=self.request.id,
            task_name=self.name,
            state=TaskState.FAILURE,
            error=str(e),
            started_at=self.start_time,
            completed_at=datetime.utcnow()
        )

# Helper functions
def calculate_lead_score(lead_data: Dict[str, Any]) -> int:
    """
    Calculate lead score based on various factors.
    
    Args:
        lead_data: Enriched lead data
        
    Returns:
        Lead score (0-100)
    """
    score = 0
    
    # Base score from event severity
    severity = lead_data.get('severity', 'moderate')
    severity_scores = {
        'low': 10,
        'moderate': 25,
        'high': 50,
        'severe': 75,
        'extreme': 100
    }
    score += severity_scores.get(severity, 25)
    
    # Property value factor
    property_value = lead_data.get('property_value', 0)
    if property_value > 500000:
        score += 20
    elif property_value > 300000:
        score += 15
    elif property_value > 150000:
        score += 10
    
    # Hail size factor
    geometry = lead_data.get('geometry', {})
    magnitude = geometry.get('magnitude', 0)
    if magnitude >= 2.0:
        score += 25
    elif magnitude >= 1.5:
        score += 15
    elif magnitude >= 1.0:
        score += 10
    
    # Data quality factor
    data_quality = lead_data.get('data_quality_score', 0)
    score += int(data_quality * 0.2)  # Up to 20 points for data quality
    
    return min(100, score)

def convert_lead_to_contact(lead_data: Dict[str, Any]) -> Contact:
    """
    Convert lead data to GoHighLevel contact format.
    
    Args:
        lead_data: Enriched lead data
        
    Returns:
        GoHighLevel Contact object
    """
    geometry = lead_data.get('geometry', {})
    owner_info = lead_data.get('owner_info', {})
    
    # Extract phone number
    phone = None
    if owner_info.get('phone'):
        phone = owner_info['phone']
    elif geometry.get('county'):  # Fallback to county info
        # This would be replaced with actual phone lookup logic
        pass
    
    # Extract email
    email = owner_info.get('email')
    
    # Create contact
    contact = Contact(
        first_name=owner_info.get('first_name', 'Property'),
        last_name=owner_info.get('last_name', 'Owner'),
        email=email,
        phone=phone,
        address=owner_info.get('address'),
        city=owner_info.get('city'),
        state=owner_info.get('state'),
        zip_code=owner_info.get('zip_code'),
        lead_source='hail_hero',
        lead_score=lead_data.get('lead_score', 0),
        status=ContactStatus.NEW,
        pipeline_stage=PipelineStage.LEAD,
        tags=['hail_event', 'generated_lead'],
        custom_fields={
            'event_id': lead_data.get('event_id'),
            'hail_size': geometry.get('magnitude'),
            'event_date': geometry.get('begin_time'),
            'property_value': lead_data.get('property_value'),
            'severity': lead_data.get('severity'),
            'risk_score': lead_data.get('risk_score')
        }
    )
    
    return contact

# Periodic task implementations
@celery_app.task
def process_noaa_data_daily():
    """Daily NOAA data processing task."""
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    # Process yesterday's data
    process_noaa_data.delay(
        start_date=yesterday,
        end_date=today,
        target_states=['CO', 'TX', 'KS', 'NE', 'OK'],  # Target hail states
        min_hail_size=0.5
    )

@celery_app.task
def enrich_pending_leads():
    """Enrich leads that are pending enrichment."""
    # This would query your database for leads that need enrichment
    # For now, it's a placeholder
    logger.info("Enriching pending leads")
    pass

@celery_app.task
def sync_crm_data():
    """Sync data with CRM system."""
    # This would handle bidirectional sync with GoHighLevel
    logger.info("Syncing CRM data")
    pass

@celery_app.task
def cleanup_old_data():
    """Clean up old data."""
    cleanup_old_data.delay(days_to_keep=90)

@celery_app.task
def health_check():
    """Perform system health check."""
    logger.info("Performing health check")
    
    # Check database connectivity
    # Check Redis connectivity
    # Check external API connectivity
    # Check worker status
    
    health_status = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'healthy',
        'checks': {}
    }
    
    return health_status

# Utility functions for task management
def get_task_status(task_id: str) -> Optional[TaskResult]:
    """Get the status of a specific task."""
    result = celery_app.AsyncResult(task_id)
    
    if result.state == 'PENDING':
        return TaskResult(
            task_id=task_id,
            task_name=result.task,
            state=TaskState.PENDING
        )
    elif result.state == 'PROGRESS':
        return TaskResult(
            task_id=task_id,
            task_name=result.task,
            state=TaskState.RUNNING
        )
    elif result.state == 'SUCCESS':
        return TaskResult(
            task_id=task_id,
            task_name=result.task,
            state=TaskState.SUCCESS,
            result=result.result
        )
    elif result.state == 'FAILURE':
        return TaskResult(
            task_id=task_id,
            task_name=result.task,
            state=TaskState.FAILURE,
            error=str(result.result)
        )
    elif result.state == 'RETRY':
        return TaskResult(
            task_id=task_id,
            task_name=result.task,
            state=TaskState.RETRY
        )
    
    return None

def get_worker_stats() -> Dict[str, Any]:
    """Get worker statistics."""
    inspector = celery_app.control.inspect()
    
    stats = {
        'active_tasks': {},
        'scheduled_tasks': {},
        'worker_stats': {},
        'queue_lengths': {}
    }
    
    try:
        # Get active tasks
        active = inspector.active()
        if active:
            stats['active_tasks'] = active
        
        # Get scheduled tasks
        scheduled = inspector.scheduled()
        if scheduled:
            stats['scheduled_tasks'] = scheduled
        
        # Get worker stats
        worker_stats = inspector.stats()
        if worker_stats:
            stats['worker_stats'] = worker_stats
        
        # Get queue lengths (approximate)
        for queue_name in ['noaa', 'enrichment', 'messaging', 'crm', 'maintenance']:
            try:
                queue_length = len(celery_app.control.inspect().reserved())
                stats['queue_lengths'][queue_name] = queue_length
            except:
                stats['queue_lengths'][queue_name] = 0
        
    except Exception as e:
        logger.error(f"Error getting worker stats: {e}")
    
    return stats

def cancel_task(task_id: str) -> bool:
    """Cancel a running task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return True
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return False

if __name__ == '__main__':
    # Start Celery worker
    celery_app.start()