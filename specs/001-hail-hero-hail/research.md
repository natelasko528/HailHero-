# Research: Hail Hero integrations & data sources

This document summarizes recommended data sources, representative provider features (Hail Recon, Hail Trace, Roofr), and integration notes for Hail Hero.

1) Provider feature summary (quick take)
- Hail Recon: strong at property-level contact enrichment and lead lists derived from storm footprints; commercial data, often licensed per-search.
- Hail Trace: geospatial event correlation and claim readiness scoring; provides inspection-ready leads and measurement tooling.
- Roofr: automated roof measurements from aerial imagery and ease-of-use measurement/estimation tools; focuses on roof geometry extraction.

2) Open / free data sources to prefer (pluggable connectors)
- NOAA / NCEI Storm Events (event metadata; may require NCEI tokens for some endpoints).
- NOAA/NWS alerts (public alert feeds) — good for near-real-time event awareness.
- OpenAddresses — public address datasets for matching candidate properties.
- OpenStreetMap / Nominatim — geocoding and reverse-geocoding.
- OpenAerialMap — community aerial imagery for visual confirmation and manual measurements.
- USGS/state parcel / assessor APIs (where available) — authoritative parcel and owner metadata.

3) Data-enrichment and contact sources
- Open sources rarely contain up-to-date phone/email; commercial providers (Hail Recon / Hail Trace) fill gaps but require licensing and consent mechanisms.
- Preference: use open sources for identification and public metadata, then layer licensed enrichment only when legally approved.

4) Integration patterns & provenance
- All enrichment fields must include `source_provenance` (provider, retrieved_ts, confidence_score, license_ref).
- Support import modes: "watch-only" (index only), "opt-in outreach" (requires explicit consent), and "full-sync" (enrich and push to CRM).

5) Operational considerations
- Rate limits and API quotas vary; design connectors with backoff/retry and batch ingestion.
- For mobile-first: keep media uploads resumable and minimize upload size by performing client-side resizing.

6) Security & legal
- Do not store or process contact PII without a documented lawful basis and consent record. Store consent details alongside contact records.

7) Next research tasks
- Obtain sample license terms for Hail Recon / Hail Trace and confirm allowed uses.
- Prototype NOAA ingestion (confirm endpoint access using my environment's network and API tokens).
- Prototype roof measurement experiments using OpenAerialMap tiles and open measurement algorithms (Roofr-style).
