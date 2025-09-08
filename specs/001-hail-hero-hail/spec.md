# Feature Specification: Hail Hero — Hail Lead Gen CRM

**Feature Branch**: `001-hail-hero-hail`  
**Created**: 2025-09-08  
**Status**: MVP Complete, Enhanced Implementation in Progress  
**Input**: User description: "Hail Hero: hail lead gen CRM for roofing insurance claim sales. Combine best features from Hail Recon, Hail Trace, and Roofr into a single product called Hail Hero. Include inspections, mapping, lead capture, claim tracking, photo documentation, roof measurements, homeowner contact data ingestion and enrichment, integrations with GoHighLevel and Twilio, mobile-first field workflows, and prefer open data sources (OpenAddresses, OpenStreetMap/Nominatim, NOAA Storm Events API, OpenAerialMap). Do NOT use mock PII; prefer open/partner data sources and explicit consent flows for contacts."

## Purpose
Hail Hero is a lead-generation and CRM platform tailored for roofing insurance-claim sales teams. It combines inspection tooling, automated hail-event discovery, homeowner/contact enrichment, roof measurement, photo documentation, and flexible integrations (GoHighLevel, Twilio, CRMs) to streamline lead capture, qualification, and claims workflow for field sales reps.

## Execution flow (high level)
1. Ingest hail-event and address data from open data sources (NOAA, OpenAddresses, OpenStreetMap, OpenAerialMap) and partner datasets.
2. Identify candidate properties in affected areas and enrich records with available contact and property metadata via permitted sources / partner APIs.
3. Create lead records and present prioritized tasks to field reps (inspections, phone/SMS outreach).
4. Capture inspection data (photos, notes, roof measurements) in the mobile app; attach to lead and optional claim.
5. Push qualified leads to GoHighLevel (contact sync / funnel), and send notifications/2-way SMS via Twilio.
6. Track claim lifecycle and status; log interactions and outcomes for reporting and analytics.

---

## User Scenarios & Testing (mandatory)

### Primary User Story
As a field sales rep, I want to see newly available leads created from recent hail events, view property/location details and available contact info, perform a mobile inspection (photos + measurements), and push qualified leads to the sales funnel while sending the homeowner an SMS notification so that I can quickly convert inspections into claim opportunities.

### Acceptance Scenarios
1. Given a new NOAA hail event polygon, when the system ingests event geometry and intersects with addresses, then the system should create candidate leads for each affected property with a confidence score and available enrichment fields populated.
2. Given a candidate lead with available contact channels, when a rep marks the lead as "Schedule Inspection", then the system should create an inspection task, push a calendar invite or message through GoHighLevel/Twilio, and mark the lead status accordingly.
3. Given an inspection performed in the mobile app, when photos and measurements are uploaded and tied to the lead, then the system should persist media, extract key metadata (roof area, approximate pitch), and flag potential claimability for review.

### Edge Cases
- Properties with incomplete geocoding or no public address data should be flagged and the rep should be able to manually confirm via map pin-drop.
- Contacts flagged by external sources as Do Not Contact (DNC) or out-of-service must not be messaged; system must surface DNC status and require explicit override.
- Duplicate leads (same address) created from overlapping events must be deduplicated or merged with a single canonical lead.

---

## Requirements (mandatory)

### Functional Requirements
- FR-001: The system MUST ingest hail event data (geometry, timestamp, severity) from NOAA Storm Events API or equivalent open sources on a scheduled cadence.
- FR-002: The system MUST geocode or reverse-geocode addresses using OpenStreetMap/Nominatim and reconcile addresses with OpenAddresses when available.
- FR-003: The system MUST create candidate lead records for properties intersecting hail event areas and compute a lead score based on event severity, roof attributes, and proximity.
- FR-004: The system MUST attempt to enrich candidate leads with homeowner contact information using permitted/enriched sources. If proprietary sources (Hail Recon, Hail Trace) are used, the system MUST record provenance and require appropriate licensing/consent.
- FR-005: The system MUST provide a mobile inspection workflow to capture photos, notes, GPS location, and roof measurement metadata, and attach them to the lead record.
- FR-006: The system MUST send notifications and two-way SMS using Twilio and sync leads/contacts to GoHighLevel when a lead is qualified.
- FR-007: The system MUST store media (photos) and measurement artifacts with tamper-evident metadata (timestamp, GPS, uploader id) for claim evidence.
- FR-008: The system MUST support deduplication rules (address + parcel id + phone) and present a merge/resolve UI for duplicates.
- FR-009: The system MUST surface data quality indicators (geocode confidence, source provenance, enrichment confidence).
- FR-010: The system MUST implement consent capture for any contact used for outreach and store consent records linked to contact channels.

*Clarifying / Legal Requirements*
- FR-011: The system MUST NOT import or use any proprietary contact databases without explicit legal permission/licensing. [NEEDS CLARIFICATION: Access to Hail Recon / Hail Trace datasets and licensing terms]

---

## Key Entities
- Homeowner / Contact: {contact_id, full_name, phone_numbers[], emails[], preferred_channel, consent_records[], source_provenance}
- Property: {property_id, address, parcel_id (if available), lat, lon, building_area, roof_material, roof_pitch_estimate}
- HailEvent: {event_id, source, geometry, start_ts, end_ts, severity}
- Lead: {lead_id, property_id, contact_id?, score, status, created_ts, last_updated_ts, provenance}
- Inspection: {inspection_id, lead_id, inspector_id, photos[], measurements{}, notes, gps, ts}
- Claim: {claim_id, lead_id, insurer, claim_status, claim_number, notes}
- IntegrationConfig: {provider, credentials_ref, webhook_endpoints, enabled}

---

## Integrations & Data Sources (preferred, open-first)

Primary open and free sources to prefer (design for pluggable connectors):
- NOAA Storm Events API (hail event polygons / storm reports) — use for event detection and ingestion.
- OpenAddresses — canonical public address data where available.
- OpenStreetMap / Nominatim — geocoding and reverse geocoding.
- OpenAerialMap / public imagery sources — aerial imagery for visual confirmation and measurement.
- USGS / state parcel or assessor APIs (where available) — parcel and owner metadata.

Commercial / partner integrations (require credentials and legal review):
- Hail Recon / Hail Trace — known providers of hail-inspection and contact/enrichment data. [NEEDS CLARIFICATION: commercial access & licensing required before ingestion]

Messaging & CRM integrations:
- GoHighLevel — contact sync, funnel automation, webhook ingestion; map fields: {first_name, last_name, phone, email, address, lead_source, lead_score, tags}
- Twilio — SMS, MMS, voice; required flows: outbound SMS templates, two-way reply handling, and call tracking webhooks.

Integration design notes:
- Always store source_provenance (provider, timestamp, confidence) on enriched fields.
- Implement webhooks for inbound events (Twilio replies, GoHighLevel status changes) and idempotent processing.

---

## Data Privacy, Consent, and Compliance
- The system MUST capture and store consent records for any contact outreach (timestamp, method, consent_text, revocation_flag).
- Contacts flagged as DNC, spam, or legally protected must be excluded from automated outreach.
- Keep a clear audit trail for media and inspection artifacts for claims (who captured, when, device GPS).
- [NEEDS CLARIFICATION] Confirm acceptable geographic coverage and legal jurisdictions for contact enrichment (US-only? multi-country?).

---

## Sample Integration Payloads (field-level mapping examples — DO NOT CONTAIN REAL PII)

GoHighLevel contact create/update (example fields):
- payload: {"first_name":"<first>","last_name":"<last>","phone":"<e.164>","email":"<email>","address":"<street, city, state, zip>","lead_source":"noaa-hail-2025-09-08","lead_score":87}

Twilio SMS outbound (example):
- body: "Hello {{first_name}}, this is {{rep_name}} from Hail Hero — we observed hail damage in your area and can inspect your roof. Reply YES to schedule a free inspection."

Twilio webhook (inbound message) handling:
- on inbound SMS, match phone -> lead; create interaction event and optionally enqueue follow-up task for rep. Ensure idempotency by MessageSid.

---

## Data Acquisition & Enrichment Strategy
- Prefer open sources for bulk ingestion (NOAA + OpenAddresses + OSM) to identify candidate properties after a hail event.
- Where open data lacks contact channels, implement connectors to partner/commercial providers only after contractual permission. Record proof of permission in `IntegrationConfig`.
- Provide an operator UI to review and opt-in contacts prior to outreach in regions with strict privacy rules.
- If the user provides an allowed third-party dataset (e.g., licensed Hail Recon export), the system will ingest and tag each record with `source_provenance` and `license_reference`.

---

## Operational Requirements
- Schedule: nightly ingestion of NOAA events plus ad-hoc manual ingestion capability.
- Performance: initial MVP targets to process a region (city / county) within 15 minutes of ingestion for candidate lead generation.
- Storage: media storage with immutable metadata and retention policy configurable by workspace admin.

---

## Open Questions / NEEDS CLARIFICATION
- Q1: Do we have licensed access to Hail Recon or Hail Trace contact/enrichment data, or should the platform rely only on open sources and GoHighLevel-provided contacts? [REQUIRED]
- Q2: Geographic scope — US only, or international (affects data sources and legal compliance)? [REQUIRED]
- Q3: Acceptable outreach channels and templates (pre-approved SMS language, do-not-contact rules)? [REQUIRED]
- Q4: Preferred storage location for media evidence (S3-compatible, local, or 3rd-party)? [OPTIONAL]

---

## Review & Acceptance Checklist
- [ ] All mandatory sections completed
- [ ] No implementation details that violate the "WHAT not HOW" rule (implementation notes are marked and will be removed from final spec)
- [ ] Data provenance and consent flows described
- [ ] Open data sources documented and preferred over proprietary sources where feasible
- [ ] All Qs under "NEEDS CLARIFICATION" answered

---

## Execution Status
- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (see NEEDS CLARIFICATION)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed

---

## MVP Implementation

An MVP has been implemented and is available in the `src/mvp` directory. See `specs/001-hail-hero-hail/README_MVP.md` for instructions on how to run it.

The MVP consists of:
- ✅ A runner script to fetch real or synthetic hail event data
- ✅ A Flask web application to display leads and accept inspection submissions  
- ✅ Beautiful responsive UI with mobile-first design
- ✅ Enhanced lead scoring algorithm with multiple components
- ✅ Successful testing via Playwright automation

**Current Status**: All MVP components verified and functional. Ready for enhanced feature implementation.
