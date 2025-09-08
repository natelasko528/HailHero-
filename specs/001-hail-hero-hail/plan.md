# Implementation Plan (high level)

Goal: Deliver an MVP of Hail Hero that can detect hail events, generate candidate leads, allow mobile inspections, and integrate with GoHighLevel and Twilio for outreach.

Phases:

1) Discovery & Data Ingestion
- Implement NOAA storm events ingestion and geospatial intersection with OpenAddresses / OSM addresses.
- Build a connector framework that supports pluggable sources (NOAA, OpenAddresses, OSM, Hail Recon)

2) Lead Model & Enrichment
- Design lead scoring rules (event severity, roof area, prior claims, enrichment confidence).
- Implement enrichment pipeline that annotates source provenance and confidence.

3) Mobile Inspection MVP
- Mobile-first web app with offline-capable photo capture, GPS tagging, and measurement capture.
- Media storage with metadata and resumable uploads.

4) Integrations
- GoHighLevel: contact sync, tags, pipelines. Use webhooks to track status changes.
- Twilio: SMS/MMS outbound templates and inbound webhook handling for replies.

5) Claim Tracking & Reporting
- Simple claim entity and state machine (new, submitted, in-review, denied, closed).
- Dashboard for rep performance and lead conversion metrics.

Non-functional considerations:
- Start with a small footprint: serverless ingestion + a minimal backend API.
- Prioritize auditability and consent tracking.

Deliverables for MVP:
- NOAA ingestion worker and address intersection job
- Lead model and enrichment pipeline (open-source connectors + adapter interface)
- Mobile inspection UI and simple backend to receive uploads
- GoHighLevel + Twilio connector prototypes
- End-to-end demo using open data and a licensed enrichment sample (if available)

---

## MVP Implementation

An MVP has been implemented and is available in the `src/mvp` directory. See `specs/001-hail-hero-hail/README_MVP.md` for instructions on how to run it.

**MVP Status (2025-09-08)**:
- ✅ **MVP-1**: `src/mvp/runner.py` fetches real NOAA data (or synthetic fallback) and generates `leads.jsonl`
- ✅ **MVP-2**: `src/mvp/app.py` provides Flask web app to serve leads and inspection submission endpoint
- ✅ **MVP-3**: `specs/001-hail-hero-hail/README_MVP.md` with setup instructions
- ✅ **MVP-4**: Successfully tested with Playwright - all components working correctly
- ✅ **MVP-5**: Verified API endpoints, error handling, and web interface functionality

## Phase 1: Enhanced MVP Implementation

**Goal**: Transform basic MVP into a beautiful, functional application with real data integration and mobile-first inspection capabilities.

**Enhancement Components**:
1. **Beautiful Responsive UI**: Modern, mobile-first web interface with professional styling
2. **Real NOAA Integration**: Proper API token integration and live data fetching
3. **Address Enrichment**: OpenAddresses/OSM integration for property-level data
4. **Enhanced Lead Scoring**: Improved algorithm incorporating property data and event severity
5. **Mobile Inspection Interface**: Enhanced mobile-responsive inspection workflow
6. **Integration Prototypes**: GoHighLevel and Twilio API connector implementations
7. **Consent Management**: DNC compliance and consent capture features
