"""
Consent Management and DNC Compliance System for Hail Hero.

This module provides comprehensive consent management and Do Not Contact (DNC) compliance
functionality to ensure all outbound communications meet legal requirements.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date, timedelta
import json
import uuid

logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of consent that can be captured."""
    SMS = "sms"
    EMAIL = "email"
    PHONE = "phone"
    MARKETING = "marketing"
    SERVICE = "service"
    TRANSACTIONAL = "transactional"


class ConsentStatus(Enum):
    """Consent status values."""
    PENDING = "pending"
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    OPT_OUT = "opt_out"


class CommunicationChannel(Enum):
    """Communication channels for consent."""
    SMS = "sms"
    EMAIL = "email"
    PHONE_CALL = "phone_call"
    MAIL = "mail"
    IN_PERSON = "in_person"
    DIGITAL_ADVERTISING = "digital_advertising"


class ComplianceLevel(Enum):
    """Compliance levels for geographic regions."""
    FEDERAL = "federal"
    STATE = "state"
    INTERNATIONAL = "international"
    INDUSTRY = "industry"


@dataclass
class ConsentRecord:
    """Individual consent record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    contact_id: str = ""
    consent_type: ConsentType = ConsentType.SMS
    status: ConsentStatus = ConsentStatus.PENDING
    channel: CommunicationChannel = CommunicationChannel.SMS
    value: str = ""  # phone number, email address, etc.
    consent_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    revocation_date: Optional[datetime] = None
    consent_method: str = ""  # web_form, phone_call, email, etc.
    consent_source: str = ""  # website, mobile_app, etc.
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    geographic_location: Optional[str] = None
    language: str = "en"
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate and set default values."""
        if not self.consent_date and self.status == ConsentStatus.ACTIVE:
            self.consent_date = datetime.utcnow()
        if not self.revocation_date and self.status == ConsentStatus.REVOKED:
            self.revocation_date = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if consent is currently active."""
        if self.status != ConsentStatus.ACTIVE:
            return False
        
        if self.expiry_date and datetime.utcnow() > self.expiry_date:
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "contact_id": self.contact_id,
            "consent_type": self.consent_type.value,
            "status": self.status.value,
            "channel": self.channel.value,
            "value": self.value,
            "consent_date": self.consent_date.isoformat() if self.consent_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "revocation_date": self.revocation_date.isoformat() if self.revocation_date else None,
            "consent_method": self.consent_method,
            "consent_source": self.consent_source,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "geographic_location": self.geographic_location,
            "language": self.language,
            "custom_fields": self.custom_fields,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsentRecord":
        """Create from dictionary."""
        record = cls(
            id=data.get("id", str(uuid.uuid4())),
            contact_id=data.get("contact_id", ""),
            consent_type=ConsentType(data.get("consent_type", "sms")),
            status=ConsentStatus(data.get("status", "pending")),
            channel=CommunicationChannel(data.get("channel", "sms")),
            value=data.get("value", ""),
            consent_method=data.get("consent_method", ""),
            consent_source=data.get("consent_source", ""),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            geographic_location=data.get("geographic_location"),
            language=data.get("language", "en"),
            custom_fields=data.get("custom_fields", {})
        )
        
        # Parse datetime fields
        if data.get("consent_date"):
            record.consent_date = datetime.fromisoformat(data["consent_date"])
        if data.get("expiry_date"):
            record.expiry_date = datetime.fromisoformat(data["expiry_date"])
        if data.get("revocation_date"):
            record.revocation_date = datetime.fromisoformat(data["revocation_date"])
        if data.get("created_at"):
            record.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            record.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return record


@dataclass
class DNCRecord:
    """Do Not Contact record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    contact_id: str = ""
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None
    dnc_list_type: str = "federal"  # federal, state, internal
    dnc_source: str = ""  # ftc, state_agency, internal_opt_out
    registration_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate and set default values."""
        if not self.registration_date:
            self.registration_date = datetime.utcnow()
        if not self.effective_date:
            self.effective_date = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if DNC record is currently active."""
        if self.expiry_date and datetime.utcnow() > self.expiry_date:
            return False
        return True
    
    def matches_contact(self, phone_number: Optional[str] = None, email_address: Optional[str] = None) -> bool:
        """Check if this DNC record matches the provided contact info."""
        if phone_number and self.phone_number and self.phone_number == phone_number:
            return True
        if email_address and self.email_address and self.email_address == email_address:
            return True
        return False


@dataclass
class AuditLog:
    """Audit log entry for compliance activities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""  # consent_given, consent_revoked, dnc_check, etc.
    contact_id: Optional[str] = None
    consent_id: Optional[str] = None
    dnc_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    action_details: Dict[str, Any] = field(default_factory=dict)
    compliance_result: str = "success"  # success, failed, blocked
    reason: Optional[str] = None
    geographic_location: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "action_type": self.action_type,
            "contact_id": self.contact_id,
            "consent_id": self.consent_id,
            "dnc_id": self.dnc_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "action_details": self.action_details,
            "compliance_result": self.compliance_result,
            "reason": self.reason,
            "geographic_location": self.geographic_location,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ComplianceRule:
    """Compliance rule for geographic regions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    compliance_level: ComplianceLevel = ComplianceLevel.FEDERAL
    geographic_scope: List[str] = field(default_factory=list)  # states, countries
    channel: CommunicationChannel = CommunicationChannel.SMS
    consent_required: bool = True
    dnc_check_required: bool = True
    time_restrictions: Dict[str, Any] = field(default_factory=dict)  # time windows, days of week
    content_restrictions: List[str] = field(default_factory=list)
    age_requirement: Optional[int] = None
    parental_consent_required: bool = False
    record_retention_days: int = 365
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def applies_to_location(self, location: str) -> bool:
        """Check if this rule applies to the given location."""
        return location in self.geographic_scope


class ComplianceError(Exception):
    """Base exception for compliance-related errors."""
    pass


class ConsentRequiredError(ComplianceError):
    """Raised when consent is required but not found."""
    pass


class DNCViolationError(ComplianceError):
    """Raised when contact attempt violates DNC rules."""
    pass


class ComplianceConfigurationError(ComplianceError):
    """Raised when compliance configuration is invalid."""
    pass


class ConsentManager:
    """Manages consent records and consent lifecycle."""
    
    def __init__(self, storage_backend: Optional[str] = None):
        """Initialize consent manager with optional storage backend."""
        self.storage_backend = storage_backend
        self._consent_records: Dict[str, ConsentRecord] = {}
        self._contact_consents: Dict[str, List[str]] = {}  # contact_id -> consent_ids
        logger.info("ConsentManager initialized")
    
    def create_consent(self, contact_id: str, consent_type: ConsentType, 
                      channel: CommunicationChannel, value: str,
                      consent_method: str = "web_form", 
                      consent_source: str = "website",
                      expiry_days: Optional[int] = None,
                      **kwargs) -> ConsentRecord:
        """Create a new consent record."""
        consent = ConsentRecord(
            contact_id=contact_id,
            consent_type=consent_type,
            channel=channel,
            value=value,
            consent_method=consent_method,
            consent_source=consent_source,
            status=ConsentStatus.ACTIVE,
            **kwargs
        )
        
        if expiry_days:
            consent.expiry_date = datetime.utcnow() + timedelta(days=expiry_days)
        
        self._consent_records[consent.id] = consent
        
        if contact_id not in self._contact_consents:
            self._contact_consents[contact_id] = []
        self._contact_consents[contact_id].append(consent.id)
        
        logger.info(f"Created consent {consent.id} for contact {contact_id}")
        return consent
    
    def revoke_consent(self, consent_id: str, reason: Optional[str] = None) -> bool:
        """Revoke a consent record."""
        if consent_id not in self._consent_records:
            return False
        
        consent = self._consent_records[consent_id]
        consent.status = ConsentStatus.REVOKED
        consent.revocation_date = datetime.utcnow()
        consent.updated_at = datetime.utcnow()
        
        if reason:
            consent.custom_fields['revocation_reason'] = reason
        
        logger.info(f"Revoked consent {consent_id} for contact {consent.contact_id}")
        return True
    
    def get_active_consents(self, contact_id: Optional[str] = None, 
                           consent_type: Optional[ConsentType] = None,
                           channel: Optional[CommunicationChannel] = None) -> List[ConsentRecord]:
        """Get active consent records, optionally filtered."""
        consents = []
        
        for consent in self._consent_records.values():
            if not consent.is_active():
                continue
            
            if contact_id and consent.contact_id != contact_id:
                continue
            
            if consent_type and consent.consent_type != consent_type:
                continue
            
            if channel and consent.channel != channel:
                continue
            
            consents.append(consent)
        
        return consents
    
    def check_consent(self, contact_id: str, consent_type: ConsentType, 
                     channel: CommunicationChannel, value: str) -> bool:
        """Check if active consent exists for the given parameters."""
        active_consents = self.get_active_consents(
            contact_id=contact_id,
            consent_type=consent_type,
            channel=channel
        )
        
        for consent in active_consents:
            if consent.value == value:
                return True
        
        return False
    
    def update_consent_expiry(self, consent_id: str, expiry_date: datetime) -> bool:
        """Update consent expiry date."""
        if consent_id not in self._consent_records:
            return False
        
        consent = self._consent_records[consent_id]
        consent.expiry_date = expiry_date
        consent.updated_at = datetime.utcnow()
        
        logger.info(f"Updated expiry for consent {consent_id}")
        return True
    
    def get_consent_by_id(self, consent_id: str) -> Optional[ConsentRecord]:
        """Get consent record by ID."""
        return self._consent_records.get(consent_id)
    
    def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records and return count cleaned."""
        expired_count = 0
        current_time = datetime.utcnow()
        
        for consent_id, consent in list(self._consent_records.items()):
            if consent.expiry_date and current_time > consent.expiry_date:
                consent.status = ConsentStatus.EXPIRED
                consent.updated_at = current_time
                expired_count += 1
        
        logger.info(f"Cleaned up {expired_count} expired consents")
        return expired_count


class DNCManager:
    """Manages Do Not Contact compliance checking."""
    
    def __init__(self, storage_backend: Optional[str] = None):
        """Initialize DNC manager with optional storage backend."""
        self.storage_backend = storage_backend
        self._dnc_records: Dict[str, DNCRecord] = {}
        self._phone_index: Dict[str, List[str]] = {}  # phone_number -> dnc_ids
        self._email_index: Dict[str, List[str]] = {}  # email_address -> dnc_ids
        logger.info("DNCManager initialized")
    
    def add_dnc_record(self, phone_number: Optional[str] = None,
                      email_address: Optional[str] = None,
                      first_name: Optional[str] = None,
                      last_name: Optional[str] = None,
                      dnc_list_type: str = "federal",
                      dnc_source: str = "internal_opt_out",
                      **kwargs) -> DNCRecord:
        """Add a new DNC record."""
        dnc_record = DNCRecord(
            phone_number=phone_number,
            email_address=email_address,
            first_name=first_name,
            last_name=last_name,
            dnc_list_type=dnc_list_type,
            dnc_source=dnc_source,
            **kwargs
        )
        
        self._dnc_records[dnc_record.id] = dnc_record
        
        if phone_number:
            if phone_number not in self._phone_index:
                self._phone_index[phone_number] = []
            self._phone_index[phone_number].append(dnc_record.id)
        
        if email_address:
            if email_address not in self._email_index:
                self._email_index[email_address] = []
            self._email_index[email_address].append(dnc_record.id)
        
        logger.info(f"Added DNC record {dnc_record.id} for {phone_number or email_address}")
        return dnc_record
    
    def check_dnc_status(self, phone_number: Optional[str] = None,
                        email_address: Optional[str] = None) -> Dict[str, Any]:
        """Check DNC status for a phone number or email."""
        result = {
            "is_blocked": False,
            "dnc_records": [],
            "blocking_reasons": [],
            "list_types": []
        }
        
        # Check phone number
        if phone_number and phone_number in self._phone_index:
            for dnc_id in self._phone_index[phone_number]:
                dnc_record = self._dnc_records.get(dnc_id)
                if dnc_record and dnc_record.is_active():
                    result["is_blocked"] = True
                    result["dnc_records"].append(dnc_record.to_dict() if hasattr(dnc_record, 'to_dict') else str(dnc_record))
                    result["blocking_reasons"].append(f"DNC {dnc_record.dnc_list_type} list")
                    result["list_types"].append(dnc_record.dnc_list_type)
        
        # Check email address
        if email_address and email_address in self._email_index:
            for dnc_id in self._email_index[email_address]:
                dnc_record = self._dnc_records.get(dnc_id)
                if dnc_record and dnc_record.is_active():
                    result["is_blocked"] = True
                    result["dnc_records"].append(dnc_record.to_dict() if hasattr(dnc_record, 'to_dict') else str(dnc_record))
                    result["blocking_reasons"].append(f"DNC {dnc_record.dnc_list_type} list")
                    result["list_types"].append(dnc_record.dnc_list_type)
        
        return result
    
    def validate_contact_attempt(self, phone_number: Optional[str] = None,
                               email_address: Optional[str] = None,
                               consent_records: Optional[List[ConsentRecord]] = None) -> bool:
        """Validate if a contact attempt is compliant."""
        # Check DNC status
        dnc_result = self.check_dnc_status(phone_number, email_address)
        if dnc_result["is_blocked"]:
            logger.warning(f"Contact attempt blocked by DNC: {phone_number or email_address}")
            return False
        
        # Check if we have explicit consent
        if consent_records:
            for consent in consent_records:
                if consent.is_active():
                    if phone_number and consent.value == phone_number:
                        return True
                    if email_address and consent.value == email_address:
                        return True
        
        # If no explicit consent, check if this is allowed under existing business relationship
        # This is a simplified check - real implementation would be more complex
        return not dnc_result["is_blocked"]
    
    def import_dnc_list(self, dnc_data: List[Dict[str, Any]], 
                       list_type: str = "federal",
                       source: str = "import") -> int:
        """Import DNC records from a list of dictionaries."""
        imported_count = 0
        
        for record_data in dnc_data:
            try:
                dnc_record = DNCRecord(
                    phone_number=record_data.get("phone_number"),
                    email_address=record_data.get("email_address"),
                    first_name=record_data.get("first_name"),
                    last_name=record_data.get("last_name"),
                    dnc_list_type=list_type,
                    dnc_source=source,
                    registration_date=datetime.fromisoformat(record_data["registration_date"]) if record_data.get("registration_date") else None,
                    notes=record_data.get("notes")
                )
                
                self._dnc_records[dnc_record.id] = dnc_record
                
                if dnc_record.phone_number:
                    if dnc_record.phone_number not in self._phone_index:
                        self._phone_index[dnc_record.phone_number] = []
                    self._phone_index[dnc_record.phone_number].append(dnc_record.id)
                
                if dnc_record.email_address:
                    if dnc_record.email_address not in self._email_index:
                        self._email_index[dnc_record.email_address] = []
                    self._email_index[dnc_record.email_address].append(dnc_record.id)
                
                imported_count += 1
                
            except Exception as e:
                logger.error(f"Error importing DNC record: {e}")
                continue
        
        logger.info(f"Imported {imported_count} DNC records from {source}")
        return imported_count
    
    def cleanup_expired_dnc_records(self) -> int:
        """Clean up expired DNC records and return count cleaned."""
        expired_count = 0
        current_time = datetime.utcnow()
        
        for dnc_id, dnc_record in list(self._dnc_records.items()):
            if dnc_record.expiry_date and current_time > dnc_record.expiry_date:
                # Remove from indexes
                if dnc_record.phone_number and dnc_record.phone_number in self._phone_index:
                    self._phone_index[dnc_record.phone_number].remove(dnc_id)
                    if not self._phone_index[dnc_record.phone_number]:
                        del self._phone_index[dnc_record.phone_number]
                
                if dnc_record.email_address and dnc_record.email_address in self._email_index:
                    self._email_index[dnc_record.email_address].remove(dnc_id)
                    if not self._email_index[dnc_record.email_address]:
                        del self._email_index[dnc_record.email_address]
                
                # Remove from main storage
                del self._dnc_records[dnc_id]
                expired_count += 1
        
        logger.info(f"Cleaned up {expired_count} expired DNC records")
        return expired_count


class DataRetentionManager:
    """Manages data retention policies and enforcement."""
    
    def __init__(self, storage_backend: Optional[str] = None):
        """Initialize data retention manager."""
        self.storage_backend = storage_backend
        self._retention_policies: Dict[str, Dict[str, Any]] = {
            "consent_records": {"retention_days": 730, "action": "anonymize"},  # 2 years
            "dnc_records": {"retention_days": 3650, "action": "delete"},  # 10 years
            "audit_logs": {"retention_days": 1825, "action": "archive"},  # 5 years
            "contact_data": {"retention_days": 1095, "action": "anonymize"},  # 3 years
            "communication_logs": {"retention_days": 365, "action": "delete"}  # 1 year
        }
        self._data_stores: Dict[str, List[Any]] = {}
        logger.info("DataRetentionManager initialized")
    
    def set_retention_policy(self, data_type: str, retention_days: int, 
                           action: str = "delete", **kwargs) -> None:
        """Set retention policy for a data type."""
        self._retention_policies[data_type] = {
            "retention_days": retention_days,
            "action": action,
            **kwargs
        }
        logger.info(f"Set retention policy for {data_type}: {retention_days} days, action: {action}")
    
    def get_retention_policy(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get retention policy for a data type."""
        return self._retention_policies.get(data_type)
    
    def register_data_store(self, data_type: str, data_store: List[Any]) -> None:
        """Register a data store for retention management."""
        self._data_stores[data_type] = data_store
        logger.info(f"Registered data store for {data_type} with {len(data_store)} items")
    
    def enforce_retention_policies(self, data_type: Optional[str] = None) -> Dict[str, int]:
        """Enforce retention policies and return counts of affected records."""
        results = {}
        
        data_types_to_process = [data_type] if data_type else list(self._data_stores.keys())
        
        for dt in data_types_to_process:
            if dt not in self._data_stores:
                continue
            
            policy = self._retention_policies.get(dt)
            if not policy:
                continue
            
            results[dt] = self._process_data_retention(dt, policy)
        
        return results
    
    def _process_data_retention(self, data_type: str, policy: Dict[str, Any]) -> int:
        """Process retention for a specific data type."""
        retention_days = policy["retention_days"]
        action = policy["action"]
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        if data_type not in self._data_stores:
            return 0
        
        data_store = self._data_stores[data_type]
        processed_count = 0
        
        for item in list(data_store):
            if self._should_retain_item(item, cutoff_date):
                continue
            
            # Apply retention action
            if action == "delete":
                data_store.remove(item)
            elif action == "anonymize":
                self._anonymize_item(item)
            elif action == "archive":
                self._archive_item(item)
            
            processed_count += 1
        
        logger.info(f"Processed {processed_count} {data_type} items with action: {action}")
        return processed_count
    
    def _should_retain_item(self, item: Any, cutoff_date: datetime) -> bool:
        """Determine if an item should be retained based on its date."""
        # Check if item has a created_at or timestamp field
        if hasattr(item, 'created_at'):
            return item.created_at > cutoff_date
        elif hasattr(item, 'timestamp'):
            return item.timestamp > cutoff_date
        elif hasattr(item, 'consent_date'):
            return item.consent_date > cutoff_date
        elif hasattr(item, 'registration_date'):
            return item.registration_date > cutoff_date
        
        # If no date field, retain by default
        return True
    
    def _anonymize_item(self, item: Any) -> None:
        """Anonymize sensitive data in an item."""
        # Anonymize PII fields
        if hasattr(item, 'phone_number'):
            item.phone_number = self._anonymize_phone(item.phone_number)
        if hasattr(item, 'email_address'):
            item.email_address = self._anonymize_email(item.email_address)
        if hasattr(item, 'first_name'):
            item.first_name = "ANONYMIZED"
        if hasattr(item, 'last_name'):
            item.last_name = "ANONYMIZED"
        if hasattr(item, 'address'):
            item.address = "ANONYMIZED"
        if hasattr(item, 'ip_address'):
            item.ip_address = "0.0.0.0"
        
        # Add anonymization metadata
        if hasattr(item, 'custom_fields'):
            item.custom_fields['anonymized_date'] = datetime.utcnow().isoformat()
            item.custom_fields['anonymization_reason'] = 'retention_policy'
    
    def _anonymize_phone(self, phone: str) -> str:
        """Anonymize phone number."""
        if not phone or len(phone) < 4:
            return "ANONYMIZED"
        return phone[:2] + "*" * (len(phone) - 4) + phone[-2:]
    
    def _anonymize_email(self, email: str) -> str:
        """Anonymize email address."""
        if not email or '@' not in email:
            return "ANONYMIZED"
        local, domain = email.split('@', 1)
        if len(local) > 2:
            local = local[:2] + "*" * (len(local) - 2)
        return f"{local}@{domain}"
    
    def _archive_item(self, item: Any) -> None:
        """Archive an item (placeholder implementation)."""
        # In a real implementation, this would move the item to archive storage
        if hasattr(item, 'custom_fields'):
            item.custom_fields['archived_date'] = datetime.utcnow().isoformat()
            item.custom_fields['archive_reason'] = 'retention_policy'
    
    def generate_retention_report(self) -> Dict[str, Any]:
        """Generate a report on data retention status."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "policies": {},
            "data_stores": {},
            "recommendations": []
        }
        
        # Report on policies
        for data_type, policy in self._retention_policies.items():
            report["policies"][data_type] = {
                "retention_days": policy["retention_days"],
                "action": policy["action"],
                "cutoff_date": (datetime.utcnow() - timedelta(days=policy["retention_days"])).isoformat()
            }
        
        # Report on data stores
        for data_type, data_store in self._data_stores.items():
            total_items = len(data_store)
            if data_type in self._retention_policies:
                policy = self._retention_policies[data_type]
                cutoff_date = datetime.utcnow() - timedelta(days=policy["retention_days"])
                
                expired_items = sum(1 for item in data_store if not self._should_retain_item(item, cutoff_date))
                
                report["data_stores"][data_type] = {
                    "total_items": total_items,
                    "expired_items": expired_items,
                    "retention_percentage": ((total_items - expired_items) / total_items * 100) if total_items > 0 else 0
                }
                
                # Add recommendations
                if expired_items > 0:
                    report["recommendations"].append({
                        "data_type": data_type,
                        "issue": "expired_items",
                        "count": expired_items,
                        "recommendation": f"Run retention enforcement for {data_type}"
                    })
        
        return report


class ComplianceAuditor:
    """Manages audit logging and compliance reporting."""
    
    def __init__(self, storage_backend: Optional[str] = None):
        """Initialize compliance auditor."""
        self.storage_backend = storage_backend
        self._audit_logs: List[AuditLog] = []
        self._compliance_rules: Dict[str, ComplianceRule] = {}
        logger.info("ComplianceAuditor initialized")
    
    def log_action(self, action_type: str, contact_id: Optional[str] = None,
                  consent_id: Optional[str] = None, dnc_id: Optional[str] = None,
                  user_id: Optional[str] = None, ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None, action_details: Optional[Dict[str, Any]] = None,
                  compliance_result: str = "success", reason: Optional[str] = None,
                  geographic_location: Optional[str] = None) -> AuditLog:
        """Log a compliance-related action."""
        audit_log = AuditLog(
            action_type=action_type,
            contact_id=contact_id,
            consent_id=consent_id,
            dnc_id=dnc_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action_details=action_details or {},
            compliance_result=compliance_result,
            reason=reason,
            geographic_location=geographic_location
        )
        
        self._audit_logs.append(audit_log)
        logger.info(f"Logged audit action: {action_type} for contact {contact_id}")
        
        return audit_log
    
    def get_audit_logs(self, contact_id: Optional[str] = None,
                      action_type: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: Optional[int] = None) -> List[AuditLog]:
        """Get audit logs with optional filtering."""
        filtered_logs = []
        
        for log in self._audit_logs:
            if contact_id and log.contact_id != contact_id:
                continue
            if action_type and log.action_type != action_type:
                continue
            if start_date and log.timestamp < start_date:
                continue
            if end_date and log.timestamp > end_date:
                continue
            
            filtered_logs.append(log)
        
        # Sort by timestamp (newest first)
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            filtered_logs = filtered_logs[:limit]
        
        return filtered_logs
    
    def generate_compliance_report(self, start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a comprehensive compliance report."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get relevant audit logs
        audit_logs = self.get_audit_logs(start_date=start_date, end_date=end_date)
        
        # Analyze logs
        total_actions = len(audit_logs)
        successful_actions = sum(1 for log in audit_logs if log.compliance_result == "success")
        failed_actions = sum(1 for log in audit_logs if log.compliance_result == "failed")
        blocked_actions = sum(1 for log in audit_logs if log.compliance_result == "blocked")
        
        # Group by action type
        action_types = {}
        for log in audit_logs:
            action_type = log.action_type
            if action_type not in action_types:
                action_types[action_type] = {"total": 0, "success": 0, "failed": 0, "blocked": 0}
            action_types[action_type]["total"] += 1
            action_types[action_type][log.compliance_result] += 1
        
        # Identify issues
        issues = []
        if failed_actions > 0:
            issues.append({
                "type": "failed_actions",
                "count": failed_actions,
                "severity": "high",
                "description": f"{failed_actions} compliance actions failed"
            })
        
        if blocked_actions > 0:
            issues.append({
                "type": "blocked_actions",
                "count": blocked_actions,
                "severity": "medium",
                "description": f"{blocked_actions} contact attempts were blocked"
            })
        
        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "failed_actions": failed_actions,
                "blocked_actions": blocked_actions,
                "success_rate": (successful_actions / total_actions * 100) if total_actions > 0 else 0
            },
            "action_types": action_types,
            "issues": issues,
            "recommendations": self._generate_recommendations(audit_logs)
        }
        
        return report
    
    def _generate_recommendations(self, audit_logs: List[AuditLog]) -> List[Dict[str, Any]]:
        """Generate recommendations based on audit log analysis."""
        recommendations = []
        
        # Analyze failure patterns
        failed_logs = [log for log in audit_logs if log.compliance_result == "failed"]
        if failed_logs:
            failure_reasons = {}
            for log in failed_logs:
                reason = log.reason or "unknown"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            most_common_failure = max(failure_reasons.items(), key=lambda x: x[1])
            recommendations.append({
                "priority": "high",
                "category": "failure_reduction",
                "issue": f"Most common failure: {most_common_failure[0]} ({most_common_failure[1]} occurrences)",
                "recommendation": "Investigate and address the root cause of this failure type"
            })
        
        # Analyze blocked patterns
        blocked_logs = [log for log in audit_logs if log.compliance_result == "blocked"]
        if blocked_logs:
            recommendations.append({
                "priority": "medium",
                "category": "dnc_compliance",
                "issue": f"{len(blocked_logs)} contact attempts blocked by DNC rules",
                "recommendation": "Review DNC list update processes and ensure timely synchronization"
            })
        
        return recommendations
    
    def add_compliance_rule(self, rule: ComplianceRule) -> None:
        """Add a compliance rule."""
        self._compliance_rules[rule.id] = rule
        logger.info(f"Added compliance rule: {rule.name}")
    
    def check_compliance_rules(self, contact_info: Dict[str, Any], 
                             action_type: str) -> Dict[str, Any]:
        """Check if an action complies with all applicable rules."""
        location = contact_info.get("geographic_location", "US")
        channel = contact_info.get("channel", "sms")
        
        applicable_rules = []
        for rule in self._compliance_rules.values():
            if rule.is_active and rule.applies_to_location(location):
                if rule.channel.value == channel:
                    applicable_rules.append(rule)
        
        violations = []
        requirements = []
        
        for rule in applicable_rules:
            if rule.consent_required:
                requirements.append(f"Consent required for {rule.channel.value} in {rule.geographic_scope}")
            
            if rule.dnc_check_required:
                requirements.append(f"DNC check required for {rule.channel.value}")
            
            # Check for specific violations
            if action_type == "contact_attempt" and rule.consent_required:
                if not contact_info.get("has_consent", False):
                    violations.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "violation": "Consent required but not present",
                        "severity": "high"
                    })
            
            if action_type == "contact_attempt" and rule.dnc_check_required:
                if contact_info.get("dnc_status", False):
                    violations.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "violation": "Contact attempt violates DNC rules",
                        "severity": "high"
                    })
        
        return {
            "applicable_rules": [rule.to_dict() if hasattr(rule, 'to_dict') else str(rule) for rule in applicable_rules],
            "requirements": requirements,
            "violations": violations,
            "is_compliant": len(violations) == 0
        }