# Byterover Project Memory — Hail Hero (hail lead-gen CRM)

According to Byterover memory layer: extract project facts and coding practices for future agents working on this repository.

Project overview
- Name: Hail Hero — hail lead generation CRM for roofing insurance-claim sales.
- Branch: `001-hail-hero-hail` (feature draft created 2025-09-08).
- Purpose: ingest hail events, identify candidate properties, enrich leads, provide mobile inspection workflows, and integrate with GoHighLevel and Twilio.

Key requirements (high-level)
- Ingest NOAA storm events and intersect with address datasets (OpenAddresses/OSM).
- Geocode/reverse-geocode via Nominatim/OpenStreetMap.
- Create lead records with provenance, scoring, and deduplication.
- Mobile-first inspection capture: photos, measurements, GPS, tamper-evident metadata.
- Outbound messaging via Twilio and contact sync to GoHighLevel.
- Consent capture and DNC handling are mandatory before outreach.

Important files & locations
- `specs/001-hail-hero-hail/spec.md` — feature spec, entities, FRs, open questions.
- `specs/001-hail-hero-hail/plan.md` — high-level implementation phases and MVP deliverables.
- `specs/001-hail-hero-hail/research.md` — data sources, enrichment patterns, legal notes.
- `specs/001-hail-hero-hail/tasks.md` — task breakdown and milestones.
- `specs/001-hail-hero-hail/prototypes/` — example scripts (NOAA ingestion, Playwright downloader, Twilio webhook prototypes).
- `memory/` — workspace memory area for agent notes and project-level memory.

Operational constraints & preferences
- Prefer open data sources first (NOAA, OpenAddresses, OSM, OpenAerialMap) and only use commercial enrichment after legal review and licensing.
- Nightly scheduled ingestion with ad-hoc manual run capability.
- Rate limits and quotas must be respected; connectors must implement exponential backoff and batching.
- Initial MVP should be lightweight (serverless ingestion + minimal backend API).

Data & privacy rules (critical)
- Do not import/use proprietary contact databases without explicit licensing and proof of permission.
- Capture and store consent records for any outreach (timestamp, method, consent text, revocation flag).
- Exclude DNC or flagged contacts from automated outreach; surface DNC and require explicit override when applicable.
- Store `source_provenance` for enriched fields (provider, retrieved_ts, confidence_score, license_ref).

Coding practices and guidance for agents (project-specific)
- Keep connectors pluggable and small: implement an adapter interface for each source.
- Always annotate enriched fields with provenance and confidence.
- For experiments/prototypes under `specs/.../prototypes`, prefer clear, small scripts and diagnostic artifacts (screenshots/logs) rather than brittle automation.
- Tests: include at least one integration smoke test for connectors that can run offline (use recorded fixtures) and a unit test for lead scoring.
- Error handling: avoid catching broad `Exception`; prefer targeted exceptions and explicit logging of failure context.

Prototypes & diagnostics
- Playwright-based NOAA downloader: `specs/.../prototypes/noaa_playwright_downloader.py` — prototype automation; keep screenshots and downloaded CSVs under `specs/.../data/`.
- CSV ingestion & parsing: `noaa_csv_downloader.py` and `noaa_ingest.py` are good references for ingestion flow.

Open questions (persisted)
- Does the project have licensed access to Hail Recon / Hail Trace? (affects enrichment approach)
- Geographic scope (US-only vs international)
- Preferred media storage backend (S3-compatible vs local)
- Approved messaging templates and legal review for outreach language

How to use this memory
- Future agents should consult this file first for constraints, data sources, and legal requirements.
- Update `memory/byterover_project_memory.md` when licensing decisions, data sources, or operational choices change.

Created: 2025-09-08 by agent

