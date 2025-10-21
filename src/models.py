"""
SQLAlchemy models for Hail Hero database.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from src.database import Base


class Lead(Base):
    """Lead model representing a potential customer."""

    __tablename__ = 'leads'

    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(String(50), default='new', index=True)
    score = Column(Float, default=0.0)
    property_data = Column(Text)  # JSON stored as text
    event_data = Column(Text)  # JSON stored as text
    scoring_details = Column(Text)  # JSON stored as text
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    inspections = relationship("Inspection", back_populates="lead", cascade="all, delete-orphan")
    photos = relationship("Photo", back_populates="lead", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Lead {self.lead_id} status={self.status} score={self.score}>"


class Inspection(Base):
    """Inspection model for property inspections."""

    __tablename__ = 'inspections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(String(255), ForeignKey('leads.lead_id'), nullable=False, index=True)
    inspector_id = Column(String(255))
    notes = Column(Text)
    photos = Column(Text)  # JSON array stored as text
    gps_location = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow)
    sync_status = Column(String(50), default='pending')

    # Relationships
    lead = relationship("Lead", back_populates="inspections")

    def __repr__(self):
        return f"<Inspection {self.id} lead={self.lead_id} status={self.sync_status}>"


class Photo(Base):
    """Photo model for inspection photos."""

    __tablename__ = 'photos'

    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(String(255), ForeignKey('leads.lead_id'), nullable=False, index=True)
    inspection_id = Column(Integer, ForeignKey('inspections.id'))
    filename = Column(String(500), nullable=False)
    filepath = Column(String(1000), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    gps_latitude = Column(Float)
    gps_longitude = Column(Float)
    photo_metadata = Column(Text)  # JSON stored as text
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    lead = relationship("Lead", back_populates="photos")

    def __repr__(self):
        return f"<Photo {self.filename} lead={self.lead_id}>"


class Event(Base):
    """Hail event model."""

    __tablename__ = 'events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(255), unique=True, nullable=False, index=True)
    event_type = Column(String(50), default='hail')
    source = Column(String(100))  # e.g., 'NOAA', 'manual'
    severity = Column(Float)
    geometry = Column(Text)  # GeoJSON stored as text
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    location_description = Column(Text)
    event_metadata = Column(Text)  # JSON stored as text
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)

    def __repr__(self):
        return f"<Event {self.event_id} type={self.event_type} severity={self.severity}>"


class Contact(Base):
    """Contact model for homeowners/customers."""

    __tablename__ = 'contacts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    contact_id = Column(String(255), unique=True, nullable=False, index=True)
    first_name = Column(String(255))
    last_name = Column(String(255))
    phone = Column(String(50), index=True)
    email = Column(String(255), index=True)
    address = Column(Text)
    consent_status = Column(String(50), default='pending')
    consent_timestamp = Column(DateTime)
    dnc_status = Column(Boolean, default=False)  # Do Not Contact
    source_provenance = Column(Text)  # JSON stored as text
    contact_metadata = Column(Text)  # JSON stored as text
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Contact {self.contact_id} {self.first_name} {self.last_name}>"
