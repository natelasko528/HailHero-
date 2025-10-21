"""
API Documentation for Hail Hero using Flask-RESTX (Swagger/OpenAPI).

This module provides comprehensive API documentation accessible at /api/docs
"""

from flask import Blueprint
from flask_restx import Api, Resource, fields, Namespace

# Create blueprint for API
api_blueprint = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# Create API with Swagger UI
api = Api(
    api_blueprint,
    version='1.0',
    title='Hail Hero API',
    description='Lead Generation and CRM API for Roofing Insurance Claim Sales',
    doc='/docs',
    contact='HailHero Team',
    license='Proprietary',
    authorizations={
        'apikey': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-KEY'
        }
    }
)

# Create namespaces
leads_ns = Namespace('leads', description='Lead management operations')
inspections_ns = Namespace('inspections', description='Inspection operations')
events_ns = Namespace('events', description='Hail event operations')
contacts_ns = Namespace('contacts', description='Contact management')
photos_ns = Namespace('photos', description='Photo management')
health_ns = Namespace('health', description='Health check operations')

api.add_namespace(leads_ns)
api.add_namespace(inspections_ns)
api.add_namespace(events_ns)
api.add_namespace(contacts_ns)
api.add_namespace(photos_ns)
api.add_namespace(health_ns)

# ============================================================================
# Models (for request/response documentation)
# ============================================================================

# Property Model
property_model = api.model('Property', {
    'address': fields.String(description='Street address'),
    'city': fields.String(description='City'),
    'state': fields.String(description='State'),
    'zipcode': fields.String(description='ZIP code'),
    'latitude': fields.Float(description='Latitude'),
    'longitude': fields.Float(description='Longitude'),
})

# Event Model
event_model = api.model('Event', {
    'event_id': fields.String(description='Event identifier'),
    'event_type': fields.String(description='Event type (e.g., hail)'),
    'severity': fields.Float(description='Event severity'),
    'start_time': fields.DateTime(description='Event start time'),
    'location': fields.String(description='Location description'),
})

# Lead Model
lead_model = api.model('Lead', {
    'lead_id': fields.String(required=True, description='Unique lead identifier'),
    'status': fields.String(description='Lead status', enum=['new', 'contacted', 'scheduled', 'inspected', 'qualified', 'closed']),
    'score': fields.Float(description='Lead score (0-100)'),
    'property': fields.Nested(property_model, description='Property information'),
    'event': fields.Nested(event_model, description='Associated hail event'),
    'created_ts': fields.DateTime(description='Creation timestamp'),
    'updated_ts': fields.DateTime(description='Last update timestamp'),
})

# Lead List Model
lead_list_model = api.model('LeadList', {
    'leads': fields.List(fields.Nested(lead_model)),
    'total': fields.Integer(description='Total number of leads'),
    'page': fields.Integer(description='Current page'),
    'per_page': fields.Integer(description='Items per page'),
})

# Inspection Model
inspection_model = api.model('Inspection', {
    'id': fields.Integer(description='Inspection ID'),
    'lead_id': fields.String(required=True, description='Associated lead ID'),
    'inspector_id': fields.String(description='Inspector identifier'),
    'notes': fields.String(description='Inspection notes'),
    'photos': fields.List(fields.String, description='Photo IDs'),
    'gps_location': fields.String(description='GPS coordinates'),
    'timestamp': fields.DateTime(description='Inspection timestamp'),
    'sync_status': fields.String(description='Sync status'),
})

# Photo Model
photo_model = api.model('Photo', {
    'id': fields.Integer(description='Photo ID'),
    'lead_id': fields.String(description='Associated lead ID'),
    'filename': fields.String(description='Photo filename'),
    'filepath': fields.String(description='File path'),
    'file_size': fields.Integer(description='File size in bytes'),
    'mime_type': fields.String(description='MIME type'),
    'gps_latitude': fields.Float(description='GPS latitude'),
    'gps_longitude': fields.Float(description='GPS longitude'),
    'uploaded_at': fields.DateTime(description='Upload timestamp'),
})

# Contact Model
contact_model = api.model('Contact', {
    'contact_id': fields.String(description='Contact identifier'),
    'first_name': fields.String(description='First name'),
    'last_name': fields.String(description='Last name'),
    'phone': fields.String(description='Phone number'),
    'email': fields.String(description='Email address'),
    'consent_status': fields.String(description='Consent status'),
    'dnc_status': fields.Boolean(description='Do Not Contact flag'),
})

# Health Check Model
health_model = api.model('HealthCheck', {
    'status': fields.String(description='Overall system status'),
    'timestamp': fields.DateTime(description='Check timestamp'),
    'components': fields.Raw(description='Component health details'),
})

# Error Model
error_model = api.model('Error', {
    'error': fields.String(description='Error message'),
    'code': fields.Integer(description='Error code'),
    'details': fields.Raw(description='Additional error details'),
})

# ============================================================================
# API Endpoints
# ============================================================================

@health_ns.route('/')
class HealthCheck(Resource):
    @health_ns.doc('health_check')
    @health_ns.marshal_with(health_model)
    def get(self):
        """System health check"""
        from datetime import datetime
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow(),
            'components': {
                'database': 'healthy',
                'redis': 'healthy',
                'celery': 'healthy'
            }
        }


@leads_ns.route('/')
class LeadList(Resource):
    @leads_ns.doc('list_leads')
    @leads_ns.param('status', 'Filter by lead status')
    @leads_ns.param('page', 'Page number', type=int, default=1)
    @leads_ns.param('per_page', 'Items per page', type=int, default=20)
    @leads_ns.marshal_with(lead_list_model)
    def get(self):
        """List all leads with optional filtering"""
        # Implementation will use actual database
        return {
            'leads': [],
            'total': 0,
            'page': 1,
            'per_page': 20
        }

    @leads_ns.doc('create_lead')
    @leads_ns.expect(lead_model)
    @leads_ns.marshal_with(lead_model, code=201)
    @leads_ns.response(400, 'Validation Error', error_model)
    def post(self):
        """Create a new lead"""
        pass


@leads_ns.route('/<string:lead_id>')
@leads_ns.param('lead_id', 'Lead identifier')
class Lead(Resource):
    @leads_ns.doc('get_lead')
    @leads_ns.marshal_with(lead_model)
    @leads_ns.response(404, 'Lead not found', error_model)
    def get(self, lead_id):
        """Get a specific lead by ID"""
        pass

    @leads_ns.doc('update_lead')
    @leads_ns.expect(lead_model)
    @leads_ns.marshal_with(lead_model)
    @leads_ns.response(404, 'Lead not found', error_model)
    def put(self, lead_id):
        """Update a lead"""
        pass

    @leads_ns.doc('delete_lead')
    @leads_ns.response(204, 'Lead deleted')
    @leads_ns.response(404, 'Lead not found', error_model)
    def delete(self, lead_id):
        """Delete a lead"""
        pass


@inspections_ns.route('/')
class InspectionList(Resource):
    @inspections_ns.doc('list_inspections')
    @inspections_ns.marshal_list_with(inspection_model)
    def get(self):
        """List all inspections"""
        return []

    @inspections_ns.doc('create_inspection')
    @inspections_ns.expect(inspection_model)
    @inspections_ns.marshal_with(inspection_model, code=201)
    def post(self):
        """Create a new inspection"""
        pass


@inspections_ns.route('/<int:inspection_id>')
@inspections_ns.param('inspection_id', 'Inspection ID')
class Inspection(Resource):
    @inspections_ns.doc('get_inspection')
    @inspections_ns.marshal_with(inspection_model)
    @inspections_ns.response(404, 'Inspection not found', error_model)
    def get(self, inspection_id):
        """Get a specific inspection"""
        pass


@photos_ns.route('/')
class PhotoList(Resource):
    @photos_ns.doc('list_photos')
    @photos_ns.param('lead_id', 'Filter by lead ID')
    @photos_ns.marshal_list_with(photo_model)
    def get(self):
        """List all photos"""
        return []


@photos_ns.route('/<int:photo_id>')
@photos_ns.param('photo_id', 'Photo ID')
class Photo(Resource):
    @photos_ns.doc('get_photo')
    @photos_ns.marshal_with(photo_model)
    @photos_ns.response(404, 'Photo not found', error_model)
    def get(self, photo_id):
        """Get photo metadata"""
        pass

    @photos_ns.doc('download_photo')
    @photos_ns.produces(['image/jpeg', 'image/png'])
    @photos_ns.response(200, 'Photo file')
    @photos_ns.response(404, 'Photo not found')
    def head(self, photo_id):
        """Download photo file"""
        pass


@contacts_ns.route('/')
class ContactList(Resource):
    @contacts_ns.doc('list_contacts')
    @contacts_ns.marshal_list_with(contact_model)
    def get(self):
        """List all contacts"""
        return []

    @contacts_ns.doc('create_contact')
    @contacts_ns.expect(contact_model)
    @contacts_ns.marshal_with(contact_model, code=201)
    def post(self):
        """Create a new contact"""
        pass


@contacts_ns.route('/<string:contact_id>')
@contacts_ns.param('contact_id', 'Contact identifier')
class Contact(Resource):
    @contacts_ns.doc('get_contact')
    @contacts_ns.marshal_with(contact_model)
    @contacts_ns.response(404, 'Contact not found', error_model)
    def get(self, contact_id):
        """Get a specific contact"""
        pass


@events_ns.route('/')
class EventList(Resource):
    @events_ns.doc('list_events')
    @events_ns.param('start_date', 'Filter by start date (YYYY-MM-DD)')
    @events_ns.param('end_date', 'Filter by end date (YYYY-MM-DD)')
    @events_ns.marshal_list_with(event_model)
    def get(self):
        """List all hail events"""
        return []


# Helper function to register with main app
def init_api(app):
    """Register API blueprint with Flask app."""
    app.register_blueprint(api_blueprint)
    return api
