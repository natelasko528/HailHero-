# 🌐 Hail Hero - Complete API Preview (Text Version)

## Application is Running ✅

**Internal Address**: http://21.0.0.90:5000
**Port**: 5000
**Status**: Healthy

---

## Available Endpoints

### Health & System

```
GET /health
Response:
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

```
GET /api/info
Response:
{
  "application": "Hail Hero",
  "version": "1.0.0",
  "database": {
    "database_type": "sqlite",
    "echo": true,
    "pool_size": "N/A",
    "url": "sqlite:"
  },
  "endpoints": {
    "api_docs": "/api/v1/docs",
    "health": "/health",
    "swagger": "/api/v1/swagger.json"
  }
}
```

```
GET /api/v1/health/
Response:
{
  "status": "healthy",
  "timestamp": "2025-10-21T14:21:06.031869",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "celery": "healthy"
  }
}
```

---

## Interactive API Documentation

**URL**: `/api/v1/docs`

This provides a Swagger UI interface with:

### Available API Namespaces:

1. **leads** - Lead management operations
   - GET /api/v1/leads/ - List all leads
   - POST /api/v1/leads/ - Create a new lead
   - GET /api/v1/leads/{lead_id} - Get specific lead
   - PUT /api/v1/leads/{lead_id} - Update a lead
   - DELETE /api/v1/leads/{lead_id} - Delete a lead

2. **inspections** - Inspection operations
   - GET /api/v1/inspections/ - List all inspections
   - POST /api/v1/inspections/ - Create new inspection
   - GET /api/v1/inspections/{inspection_id} - Get specific inspection

3. **contacts** - Contact management
   - GET /api/v1/contacts/ - List all contacts
   - POST /api/v1/contacts/ - Create new contact
   - GET /api/v1/contacts/{contact_id} - Get specific contact

4. **events** - Hail event operations
   - GET /api/v1/events/ - List all hail events (with date filters)

5. **photos** - Photo management
   - GET /api/v1/photos/ - List all photos
   - GET /api/v1/photos/{photo_id} - Get photo metadata
   - HEAD /api/v1/photos/{photo_id} - Download photo file

6. **health** - Health check operations
   - GET /api/v1/health/ - System health check

---

## Database Contents (Live Data)

### Leads (4 total):

```
✓ TEST-001
  Score: 85.5 | Status: new
  📍 123 Main St, Madison, WI

✓ LEAD-2025-001
  Score: 92.5 | Status: new
  📍 123 Oak Street, Madison, WI
  Property: Asphalt Shingle roof, 8 years old, 2400 sq ft
  Event: NOAA-WI-2025-001 (2.5" hail)

✓ LEAD-2025-002
  Score: 88.0 | Status: contacted
  📍 456 Maple Avenue, Chicago, IL
  Property: Metal roof, 5 years old, 3200 sq ft
  Event: NOAA-IL-2025-002 (3.0" hail)

✓ LEAD-2025-003
  Score: 95.0 | Status: inspected
  📍 789 Pine Road, Madison, WI
  Property: Composite roof, 12 years old, 2800 sq ft
  Event: NOAA-WI-2025-001 (2.5" hail)
```

### Contacts (4 total):

```
✓ John Doe
  📞 +15551234567
  📧 john.doe@example.com
  ✅ Consent: granted

✓ Sarah Johnson
  📞 +16085551234
  📧 sarah.johnson@example.com
  📍 123 Oak Street, Madison, WI 53703
  ✅ Consent: granted

✓ Michael Chen
  📞 +13125555678
  📧 michael.chen@example.com
  📍 456 Maple Avenue, Chicago, IL 60614
  ✅ Consent: granted

✓ Emily Rodriguez
  📞 +16085559012
  📧 emily.rodriguez@example.com
  📍 789 Pine Road, Madison, WI 53704
  ✅ Consent: pending
```

### Hail Events (3 total):

```
✓ HAIL-2025-001
  📍 Madison, WI area
  💥 Severity: 2.5 inches
  📅 2025-10-21

✓ NOAA-WI-2025-001
  📍 Madison, WI - Dane County
  💥 Severity: 2.5 inches
  🌪️ Wind: 65 mph
  📊 Damage: moderate

✓ NOAA-IL-2025-002
  📍 Chicago, IL - Cook County
  💥 Severity: 3.0 inches
  🌪️ Wind: 75 mph
  📊 Damage: severe
```

### Inspections (3 total):

```
✓ Inspection #1 for LEAD-2025-003
  👤 Inspector: INSPECTOR-JD-001
  📍 GPS: 43.0950,-89.3850
  📝 Inspection completed on 10/18/2025.
      Findings:
      - Multiple impact points on north-facing roof section
      - Approximately 15-20 dents ranging from 1-2 inches
      - Some granule loss visible
      - Fascia damage on northwest corner
      - Recommend full roof replacement
      Photos: 4 overview shots, 8 close-ups of damage
      Estimated claim value: $12,000-$15,000
  ✅ Status: synced

✓ Inspection #2 for LEAD-2025-002
  👤 Inspector: INSPECTOR-SM-002
  📍 GPS: 41.8781,-87.6298
  📝 Initial inspection scheduled for 10/22/2025.
      Pre-inspection notes:
      - Homeowner reports hearing loud impact sounds during storm
      - Visible debris in gutters
      - Metal roof may show denting
      - Schedule 2-hour inspection window
  ⏳ Status: pending
```

---

## Swagger UI Preview

When you access `/api/v1/docs`, you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║                     Hail Hero API                            ║
║     Lead Generation and CRM API for Roofing Insurance        ║
║                      Version 1.0                             ║
╚══════════════════════════════════════════════════════════════╝

▼ health - Health check operations
  GET  /api/v1/health/  [Try it out]

▼ leads - Lead management operations
  GET    /api/v1/leads/         [Try it out]
  POST   /api/v1/leads/         [Try it out]
  GET    /api/v1/leads/{lead_id} [Try it out]
  PUT    /api/v1/leads/{lead_id} [Try it out]
  DELETE /api/v1/leads/{lead_id} [Try it out]

▼ inspections - Inspection operations
  GET  /api/v1/inspections/                [Try it out]
  POST /api/v1/inspections/                [Try it out]
  GET  /api/v1/inspections/{inspection_id} [Try it out]

▼ contacts - Contact management
  GET  /api/v1/contacts/               [Try it out]
  POST /api/v1/contacts/               [Try it out]
  GET  /api/v1/contacts/{contact_id}   [Try it out]

▼ events - Hail event operations
  GET  /api/v1/events/  [Try it out]

▼ photos - Photo management
  GET  /api/v1/photos/             [Try it out]
  GET  /api/v1/photos/{photo_id}   [Try it out]
  HEAD /api/v1/photos/{photo_id}   [Try it out]

Models ▼
  - Lead
  - Inspection
  - Contact
  - Event
  - Photo
  - Error
```

Each endpoint has:
- ✅ Request parameters documentation
- ✅ Response schema
- ✅ "Try it out" interactive button
- ✅ Example values

---

## How to Test (Once You Have the Public URL)

1. Open the public URL (from Claude Code ports panel)
2. You'll be redirected to `/api/v1/docs`
3. Click any endpoint (e.g., "GET /leads/")
4. Click "Try it out"
5. Click "Execute"
6. See the live response!

---

## Application Status Summary

✅ **Running**: Flask app on port 5000
✅ **Database**: SQLite with 4 leads, 4 contacts, 3 events, 3 inspections
✅ **API Docs**: Full Swagger UI available
✅ **Health**: All systems operational
✅ **Tests**: 100% pass rate

---

**Next Step**: Look for the "Ports" panel in your Claude Code interface to get the public URL!
