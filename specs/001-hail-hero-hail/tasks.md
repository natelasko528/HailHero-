# Implementation Tasks — Hail Hero (MVP)

Milestone 1 — Ingestion & Identification
- T1.1: Create NOAA ingestion worker (cron/job) that fetches recent hail events and stores event geometries.
- T1.2: Implement address intersection job using OpenAddresses and OSM data; output candidate property list.
- T1.3: Implement basic deduplication heuristics (address normalization + geometry proximity).

Milestone 2 — Enrichment & Lead Model
- T2.1: Build enrichment adapter interface and OSM/OpenAddresses enrichers.
- T2.2: Add partner enrichment adapter placeholder for Hail Recon / Hail Trace.
- T2.3: Implement lead scoring and provenance fields.

Milestone 3 — Mobile Inspection MVP
- T3.1: Create mobile web app skeleton with offline-capable photo capture.
- T3.2: Backend endpoints for upload, inspection metadata, and measurement attachment.
- T3.3: Implement media storage with metadata and tamper-evidence fields.

Milestone 4 — Integrations
- T4.1: Implement Twilio outbound SMS + inbound webhook handler prototype.
- T4.2: Implement GoHighLevel contact sync prototype (push leads, map fields).
- T4.3: Webhook handlers and idempotency logic for inbound events.

Milestone 5 — Claims & Reporting
- T5.1: Basic claim entity and life-cycle management.
- T5.2: Reporting dashboard with lead conversion and inspection KPIs.

Cross-cutting Tasks
- C1: Consent capture UI + consent record storage
- C2: Logging, monitoring, and error handling for ingestion jobs
- C3: Security review and data retention policy drafting

Estimate: 6–12 weeks for a small cross-functional team (2 backend, 1 mobile/frontend, 1 ops).

---

## MVP Implementation (2025-09-08)
- **MVP-1**: Implemented `src/mvp/runner.py` to fetch real NOAA data (or synthetic fallback) and generate `leads.jsonl`.
- **MVP-2**: Implemented `src/mvp/app.py` (minimal Flask app) to serve leads and provide an inspection submission endpoint.
- **MVP-3**: Added `specs/001-hail-hero-hail/README_MVP.md` with instructions.
- **Status**: MVP runner has been executed. `leads.jsonl` is populated. The Flask app is ready to be run.

## Next Phase Implementation (2025-09-08)
- **Address Enrichment**: Need to implement OpenAddresses/OSM integration for property address enrichment
- **Lead Scoring Enhancement**: Improve scoring algorithm with property data and event severity
- **Mobile Inspection UI**: Enhance web app for mobile-first inspection workflow
- **Integration Connectors**: Implement GoHighLevel and Twilio API connectors
- **Consent Management**: Add consent capture and DNC compliance features
