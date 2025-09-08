# Hail Hero Project Constitution

## Core Principles

### I. Compliance-First Development
All contact enrichment and data processing must prioritize legal compliance and consent management. System must track data provenance, store consent records, and respect Do Not Contact flags. No proprietary data ingestion without explicit licensing.

### II. Mobile-First Field Operations
The application is designed for field sales representatives working in mobile environments. All workflows must be optimized for mobile devices with offline capabilities, GPS integration, and efficient data capture.

### III. Open Data Preference
Prefer open data sources (NOAA, OpenAddresses, OpenStreetMap) over proprietary sources. All commercial integrations require proper licensing and legal review before implementation.

### IV. Production-Ready Architecture
Build with production deployment in mind from day one. Use Docker containerization, proper configuration management, health checks, and scalable architecture patterns.

### V. Test-Driven Development
Mandatory testing with comprehensive test coverage. Unit tests for business logic, integration tests for external APIs, and E2E tests for critical user workflows.

## Technical Standards

### Technology Stack
- **Backend**: Python 3.11+ with Flask
- **Database**: SQLite (development), PostgreSQL (production)
- **Caching/Queue**: Redis with Celery for async tasks
- **Geospatial**: GeoPandas, Shapely, Nominatim
- **External APIs**: NOAA Storm Events, Twilio, GoHighLevel
- **Containerization**: Multi-stage Docker builds
- **Testing**: pytest with Playwright for E2E testing

### Data Management
- All media uploads include tamper-evident metadata
- Data provenance tracking for all enriched fields
- Immutable audit trails for compliance requirements
- Configurable data retention policies

### Security & Compliance
- Environment-based configuration management
- API key security through environment variables
- Contact consent management and tracking
- Do Not Contact flag enforcement
- Data encryption for sensitive information

## Development Workflow

### Local Development
1. Use `docker-compose --profile development` for hot reload
2. Run enhanced data processing runner for lead generation
3. Test with both synthetic and real NOAA data
4. Validate mobile responsiveness across devices

### Testing Requirements
- Unit tests for all business logic (pytest)
- Integration tests for external API integrations
- E2E tests using Playwright for critical user journeys
- Performance testing for data ingestion workflows
- Security testing for compliance features

### Deployment Process
1. Build production Docker containers
2. Run database migrations
3. Configure environment-specific settings
4. Deploy with nginx reverse proxy
5. Monitor health checks and metrics

## Quality Gates

### Code Quality
- Black formatting for consistent code style
- Flake8 linting for code quality
- MyPy type checking for type safety
- Comprehensive code reviews for all changes

### Performance Standards
- Data ingestion within 15 minutes for regional processing
- Mobile page load times under 3 seconds
- API response times under 500ms
- Database query optimization for large datasets

### Documentation Requirements
- Up-to-date CLAUDE.md for development guidance
- API documentation for all endpoints
- User guides for field operations
- Compliance documentation for legal requirements

## Governance

### Change Management
- All changes must comply with constitution principles
- Breaking changes require migration plans and approval
- Compliance changes take precedence over feature development
- Security vulnerabilities must be addressed immediately

### Decision Making
- Technical decisions prioritize production readiness
- Data source choices favor open over proprietary
- Mobile experience is primary consideration
- Compliance requirements are non-negotiable

**Version**: 1.0.0 | **Ratified**: 2025-09-08 | **Last Amended**: 2025-09-08