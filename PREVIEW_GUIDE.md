# 🎨 Hail Hero - Preview & Testing Guide

## Quick Access Links

The application is currently **running** and ready to test!

### 🌐 Web Interfaces

| Interface | URL | Description |
|-----------|-----|-------------|
| **API Documentation** | http://localhost:5000/api/v1/docs | Interactive Swagger UI - Try endpoints here! |
| **Health Check** | http://localhost:5000/health | System status |
| **System Info** | http://localhost:5000/api/info | Application details |
| **Swagger JSON** | http://localhost:5000/api/v1/swagger.json | OpenAPI specification |

---

## 🧪 Testing Methods

### 1. **Interactive Swagger UI** (Recommended!)

Open in your browser: **http://localhost:5000/api/v1/docs**

This gives you:
- ✅ Visual interface to test all endpoints
- ✅ "Try it out" buttons for each endpoint
- ✅ Automatic request/response examples
- ✅ Schema validation
- ✅ No coding required!

**How to use:**
1. Open http://localhost:5000/api/v1/docs in your browser
2. Expand any endpoint (e.g., "GET /leads/")
3. Click "Try it out"
4. Click "Execute"
5. See the response below!

---

### 2. **Command Line Testing**

#### Test All Endpoints:
```bash
./test_endpoints.sh
```

#### Test Individual Endpoints:

**Health Check:**
```bash
curl http://localhost:5000/health | python3 -m json.tool
```

**List All Leads:**
```bash
curl http://localhost:5000/api/v1/leads/ | python3 -m json.tool
```

**Get System Info:**
```bash
curl http://localhost:5000/api/info | python3 -m json.tool
```

**List Inspections:**
```bash
curl http://localhost:5000/api/v1/inspections/ | python3 -m json.tool
```

**List Contacts:**
```bash
curl http://localhost:5000/api/v1/contacts/ | python3 -m json.tool
```

**List Events:**
```bash
curl http://localhost:5000/api/v1/events/ | python3 -m json.tool
```

---

### 3. **Python Testing**

Run the comprehensive test suite:
```bash
python3 test_application.py
```

This tests:
- ✅ Configuration files
- ✅ Database operations
- ✅ All API endpoints
- ✅ Data relationships

---

### 4. **Using Python Requests Library**

Create a test script:

```python
import requests
import json

BASE_URL = 'http://localhost:5000/api/v1'

# Get all leads
response = requests.get(f'{BASE_URL}/leads/')
leads = response.json()
print(f"Total leads: {leads['total']}")

# Get specific lead
lead_id = 'LEAD-2025-001'
response = requests.get(f'{BASE_URL}/leads/{lead_id}')
lead = response.json()
print(json.dumps(lead, indent=2))
```

---

## 📊 Sample Data Available

The database now contains realistic sample data:

### **Hail Events (2)**
- **NOAA-WI-2025-001** - Madison, WI (2.5" hail, moderate damage)
- **NOAA-IL-2025-002** - Chicago, IL (3.0" hail, severe damage)

### **Leads (3)**
- **LEAD-2025-001** - 123 Oak St, Madison (Score: 92.5, Status: new)
- **LEAD-2025-002** - 456 Maple Ave, Chicago (Score: 88.0, Status: contacted)
- **LEAD-2025-003** - 789 Pine Rd, Madison (Score: 95.0, Status: inspected)

### **Contacts (3)**
- **Sarah Johnson** - Madison, WI (+16085551234)
- **Michael Chen** - Chicago, IL (+13125555678)
- **Emily Rodriguez** - Madison, WI (+16085559012)

### **Inspections (2)**
- Completed inspection for LEAD-2025-003 with damage assessment
- Scheduled inspection for LEAD-2025-002

---

## 🎯 Common Testing Scenarios

### Scenario 1: View All Leads

**Via Swagger UI:**
1. Go to http://localhost:5000/api/v1/docs
2. Find "GET /leads/"
3. Click "Try it out"
4. Click "Execute"

**Via Command Line:**
```bash
curl http://localhost:5000/api/v1/leads/ | python3 -m json.tool
```

**Expected Result:**
```json
{
  "leads": [
    {
      "lead_id": "LEAD-2025-001",
      "status": "new",
      "score": 92.5,
      "property": {...},
      "event": {...}
    }
  ],
  "total": 3,
  "page": 1,
  "per_page": 20
}
```

---

### Scenario 2: Check System Health

**Via Browser:**
Open: http://localhost:5000/health

**Via Command Line:**
```bash
curl http://localhost:5000/health
```

**Expected Result:**
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

---

### Scenario 3: View Lead Details

**Via Swagger UI:**
1. Go to http://localhost:5000/api/v1/docs
2. Find "GET /leads/{lead_id}"
3. Click "Try it out"
4. Enter lead_id: `LEAD-2025-001`
5. Click "Execute"

**Via Command Line:**
```bash
curl http://localhost:5000/api/v1/leads/LEAD-2025-001 | python3 -m json.tool
```

---

### Scenario 4: Filter Leads by Status

**Query Parameters:**
```bash
# Via command line
curl "http://localhost:5000/api/v1/leads/?status=new" | python3 -m json.tool

# Via Swagger UI - use the "status" parameter field
```

---

## 🔧 Advanced Testing

### Database Direct Access

**View database contents:**
```python
python3 -c "
from src.database import get_db_manager
from src.models import Lead

db = get_db_manager()
with db.get_session() as session:
    leads = session.query(Lead).all()
    for lead in leads:
        print(f'{lead.lead_id}: {lead.status} (Score: {lead.score})')
"
```

### Run Migrations

```bash
# View current migration
alembic current

# Upgrade to latest
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Description"
```

### Add More Sample Data

```bash
python3 create_sample_data.py
```

---

## 📱 Testing the Mobile UI (Future)

The original MVP app includes a mobile-first UI. To test it:

```bash
# Start the original MVP app
export FLASK_APP=src/mvp/app.py
python3 -m flask run --port 5001

# Access at http://localhost:5001
```

The mobile UI includes:
- Lead list view
- Inspection forms
- Photo upload
- GPS tagging
- Offline support

---

## 🐛 Troubleshooting

### App Not Responding?

**Check if running:**
```bash
curl http://localhost:5000/health
```

**View logs:**
```bash
tail -f logs/app_output.log
tail -f logs/hailhero.log
```

**Restart app:**
```bash
pkill -f "python3 run_app.py"
python3 run_app.py > logs/app_output.log 2>&1 &
```

### Database Issues?

**Check database:**
```bash
python3 -c "from src.database import get_db_manager; print(get_db_manager().health_check())"
```

**Reset database:**
```bash
rm data/hailhero.db
alembic upgrade head
python3 create_sample_data.py
```

### Port Already in Use?

**Find process:**
```bash
lsof -i :5000
```

**Kill process:**
```bash
pkill -f "python3 run_app.py"
```

---

## 📈 Performance Testing

### Load Test with Apache Bench

```bash
# Install ab if needed
apt-get install apache2-utils

# Test health endpoint
ab -n 1000 -c 10 http://localhost:5000/health

# Test leads endpoint
ab -n 100 -c 5 http://localhost:5000/api/v1/leads/
```

### Monitor Response Times

```bash
# Use curl with timing
curl -w "\nTotal time: %{time_total}s\n" http://localhost:5000/api/v1/leads/
```

---

## 🎉 Quick Win Tests

Run these to verify everything works:

```bash
# 1. Health check
curl http://localhost:5000/health

# 2. View sample leads
curl http://localhost:5000/api/v1/leads/ | python3 -m json.tool

# 3. Open Swagger UI
# Go to: http://localhost:5000/api/v1/docs

# 4. Run full test suite
python3 test_application.py
```

---

## 📚 Next Steps

After testing:

1. **Configure External Services**
   - Add NOAA API token to `.env`
   - Configure Twilio credentials
   - Set up GoHighLevel integration

2. **Customize**
   - Modify models in `src/models.py`
   - Create new migrations with `alembic revision --autogenerate`
   - Add custom endpoints in `src/mvp/api_docs.py`

3. **Deploy**
   - Use `docker-compose up -d` for production
   - Configure SSL certificates
   - Set up monitoring

---

## 🔗 Resources

- **API Documentation**: http://localhost:5000/api/v1/docs
- **Deployment Guide**: DEPLOYMENT_SUMMARY.md
- **Project Guidelines**: CLAUDE.md
- **Database Migrations**: alembic/versions/

---

**Happy Testing! 🚀**

The application is production-ready and fully operational.
Access the interactive API docs at: **http://localhost:5000/api/v1/docs**
