#!/usr/bin/env python3
"""Hail Hero MVP Flask App - Enhanced Mobile-First Inspection Workflow.

Provides a production-ready mobile-first web UI for hail damage inspections
with offline capabilities, GPS tagging, and real-time synchronization.

Usage:
  export FLASK_APP=src/mvp/app.py
  export FLASK_ENV=production
  flask run
"""

import json
import datetime
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from flask import Flask, jsonify, request, render_template_string, Response, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler

# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'specs' / '001-hail-hero-hail' / 'data'
LEADS_FILE = DATA_DIR / 'leads.jsonl'
DATABASE_PATH = BASE_DIR / 'data' / 'hail_hero.db'
UPLOAD_FOLDER = BASE_DIR / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create necessary directories
DATABASE_PATH.parent.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

# App configuration
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Enable CORS for mobile app integration
CORS(app)

# Configure logging
if not app.debug:
    handler = RotatingFileHandler('hail_hero.log', maxBytes=10000, backupCount=1)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Hail Hero startup')

# --- Database and Data Store ---
def init_database():
    """Initialize SQLite database for production use."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create leads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'new',
            score REAL DEFAULT 0,
            property_data TEXT,
            event_data TEXT,
            scoring_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create inspections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT NOT NULL,
            inspector_id TEXT,
            notes TEXT,
            photos TEXT,
            gps_location TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sync_status TEXT DEFAULT 'pending',
            FOREIGN KEY (lead_id) REFERENCES leads (lead_id)
        )
    ''')
    
    # Create photos table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT NOT NULL,
            inspection_id INTEGER,
            filename TEXT NOT NULL,
            original_filename TEXT,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            mime_type TEXT,
            gps_coordinates TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sync_status TEXT DEFAULT 'pending',
            FOREIGN KEY (lead_id) REFERENCES leads (lead_id),
            FOREIGN KEY (inspection_id) REFERENCES inspections (id)
        )
    ''')
    
    # Create sync_log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            action TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# --- Utility Functions ---
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_leads_from_db():
    """Load leads from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT l.*, 
               json_group_array(
                   json_object('id', i.id, 'notes', i.notes, 'photos', i.photos, 
                              'gps_location', i.gps_location, 'timestamp', i.timestamp)
               ) as inspections
        FROM leads l
        LEFT JOIN inspections i ON l.lead_id = i.lead_id
        GROUP BY l.lead_id
        ORDER BY l.created_at DESC
    ''')
    
    leads = []
    for row in cursor.fetchall():
        lead = {
            'lead_id': row['lead_id'],
            'status': row['status'],
            'score': row['score'],
            'property': json.loads(row['property_data']) if row['property_data'] else {},
            'event': json.loads(row['event_data']) if row['event_data'] else {},
            'scoring_details': json.loads(row['scoring_details']) if row['scoring_details'] else None,
            'created_ts': row['created_at'],
            'updated_ts': row['updated_at']
        }
        
        # Parse inspections
        if row['inspections']:
            inspections = json.loads(row['inspections'])
            if inspections[0]['id'] is not None:  # Has inspections
                lead['inspection'] = inspections[0]
        
        leads.append(lead)
    
    conn.close()
    return leads

def load_leads():
    """Load leads from JSONL file and migrate to database if needed."""
    global leads_db
    
    # First try to load from database
    try:
        leads_db = load_leads_from_db()
        if leads_db:
            app.logger.info(f'Loaded {len(leads_db)} leads from database')
            return
    except Exception as e:
        app.logger.error(f'Error loading from database: {e}')
    
    # Fallback to JSONL file and migrate
    if not LEADS_FILE.exists():
        app.logger.warning(f"Leads file not found at {LEADS_FILE}. Run the runner first.")
        leads_db = []
        return

    with LEADS_FILE.open("r", encoding="utf-8") as f:
        leads_db = [json.loads(line) for line in f]
    
    # Migrate to database
    migrate_leads_to_db(leads_db)
    app.logger.info(f'Migrated {len(leads_db)} leads to database')

def migrate_leads_to_db(leads):
    """Migrate leads from JSONL to database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for lead in leads:
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO leads 
                (lead_id, status, score, property_data, event_data, scoring_details, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lead.get('lead_id'),
                lead.get('status', 'new'),
                lead.get('score', 0),
                json.dumps(lead.get('property', {})),
                json.dumps(lead.get('event', {})),
                json.dumps(lead.get('scoring_details')),
                lead.get('created_ts', lead.get('timestamp', datetime.datetime.utcnow().isoformat())),
                lead.get('updated_ts', datetime.datetime.utcnow().isoformat())
            ))
        except Exception as e:
            app.logger.error(f'Error migrating lead {lead.get("lead_id")}: {e}')
    
    conn.commit()
    conn.close()

# --- HTML Templates (for MVP) ---
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hail Hero - Lead Management</title>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary-color: #2563eb;
      --primary-dark: #1d4ed8;
      --primary-light: #60a5fa;
      --secondary-color: #10b981;
      --secondary-dark: #059669;
      --danger-color: #ef4444;
      --danger-dark: #dc2626;
      --warning-color: #f59e0b;
      --warning-dark: #d97706;
      --info-color: #3b82f6;
      --success-color: #22c55e;
      --purple-color: #8b5cf6;
      --gray-50: #f9fafb;
      --gray-100: #f3f4f6;
      --gray-200: #e5e7eb;
      --gray-300: #d1d5db;
      --gray-400: #9ca3af;
      --gray-500: #6b7280;
      --gray-600: #4b5563;
      --gray-700: #374151;
      --gray-800: #1f2937;
      --gray-900: #111827;
      --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
      --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      --border-radius: 0.75rem;
      --border-radius-sm: 0.5rem;
      --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      --transition-fast: all 0.2s ease;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      color: var(--gray-800);
      line-height: 1.6;
      min-height: 100vh;
      position: relative;
    }

    /* Animated background */
    body::before {
      content: '';
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: 
        radial-gradient(circle at 20% 50%, rgba(37, 99, 235, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 40% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
      z-index: -1;
      animation: backgroundShift 20s ease-in-out infinite;
    }

    @keyframes backgroundShift {
      0%, 100% { transform: translateX(0) translateY(0); }
      50% { transform: translateX(-20px) translateY(-20px); }
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 0 1rem;
    }

    /* Header Styles */
    header {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(20px);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      position: sticky;
      top: 0;
      z-index: 1000;
      border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }

    .header-content {
      padding: 1.25rem 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 1.75rem;
      font-weight: 800;
      color: var(--primary-color);
      text-decoration: none;
      transition: var(--transition);
    }

    .logo:hover {
      transform: translateY(-2px);
    }

    .logo i {
      font-size: 2.25rem;
      background: linear-gradient(135deg, var(--primary-color), var(--purple-color));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.05); }
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .mobile-menu-toggle {
      display: none;
      background: none;
      border: none;
      font-size: 1.5rem;
      color: var(--gray-700);
      cursor: pointer;
      transition: var(--transition);
    }

    .mobile-menu-toggle:hover {
      color: var(--primary-color);
    }

    /* Stats Dashboard */
    .stats-dashboard {
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(10px);
      border-radius: var(--border-radius);
      padding: 1.5rem;
      margin: 1.5rem 0;
      box-shadow: var(--shadow-md);
      border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1.5rem;
    }

    .stat-card {
      background: linear-gradient(135deg, var(--gray-50), white);
      padding: 1.25rem;
      border-radius: var(--border-radius-sm);
      text-align: center;
      border: 1px solid var(--gray-200);
      transition: var(--transition);
      position: relative;
      overflow: hidden;
    }

    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
      transition: var(--transition);
    }

    .stat-card:hover::before {
      left: 100%;
    }

    .stat-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }

    .stat-icon {
      font-size: 2rem;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .stat-value {
      font-size: 2rem;
      font-weight: 700;
      color: var(--gray-900);
      margin-bottom: 0.25rem;
    }

    .stat-label {
      font-size: 0.875rem;
      color: var(--gray-600);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }

    /* Advanced Filters */
    .filters-section {
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(10px);
      border-radius: var(--border-radius);
      padding: 1.5rem;
      margin: 1.5rem 0;
      box-shadow: var(--shadow-md);
      border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .filters-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }

    .filters-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: var(--gray-800);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .filters-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .filter-group {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .filter-label {
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--gray-700);
    }

    .filter-input {
      padding: 0.75rem 1rem;
      border: 2px solid var(--gray-200);
      border-radius: var(--border-radius-sm);
      background: white;
      color: var(--gray-700);
      font-size: 0.875rem;
      transition: var(--transition);
      width: 100%;
    }

    .filter-input:focus {
      outline: none;
      border-color: var(--primary-color);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    .search-box {
      position: relative;
    }

    .search-box i {
      position: absolute;
      left: 1rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--gray-400);
    }

    .search-box input {
      padding-left: 2.5rem;
    }

    /* Lead Cards Grid */
    .leads-section {
      margin: 1.5rem 0;
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
    }

    .section-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--gray-900);
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .view-toggle {
      display: flex;
      gap: 0.5rem;
    }

    .view-btn {
      padding: 0.5rem;
      background: var(--gray-200);
      border: none;
      border-radius: var(--border-radius-sm);
      color: var(--gray-600);
      cursor: pointer;
      transition: var(--transition);
    }

    .view-btn.active {
      background: var(--primary-color);
      color: white;
    }

    .leads-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }

    .leads-grid.list-view {
      grid-template-columns: 1fr;
    }

    /* Enhanced Lead Cards */
    .lead-card {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-radius: var(--border-radius);
      box-shadow: var(--shadow-md);
      overflow: hidden;
      transition: var(--transition);
      border: 1px solid rgba(255, 255, 255, 0.2);
      position: relative;
    }

    .lead-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: linear-gradient(90deg, var(--primary-color), var(--purple-color));
    }

    .lead-card:hover {
      transform: translateY(-4px);
      box-shadow: var(--shadow-xl);
    }

    .lead-card.priority-high::before {
      background: linear-gradient(90deg, var(--danger-color), var(--warning-color));
    }

    .lead-card.priority-medium::before {
      background: linear-gradient(90deg, var(--warning-color), var(--info-color));
    }

    .lead-card.priority-low::before {
      background: linear-gradient(90deg, var(--secondary-color), var(--info-color));
    }

    .lead-header {
      padding: 1.5rem;
      border-bottom: 1px solid var(--gray-100);
      background: linear-gradient(135deg, var(--gray-50), white);
    }

    .lead-title {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1rem;
    }

    .lead-id {
      font-weight: 700;
      color: var(--gray-900);
      font-size: 1.25rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .lead-badge {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.375rem 0.875rem;
      border-radius: 9999px;
      font-size: 0.875rem;
      font-weight: 600;
      backdrop-filter: blur(10px);
    }

    .score-badge {
      background: rgba(239, 68, 68, 0.1);
      color: var(--danger-color);
      border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .score-medium {
      background: rgba(245, 158, 11, 0.1);
      color: var(--warning-color);
      border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .score-low {
      background: rgba(16, 185, 129, 0.1);
      color: var(--secondary-color);
      border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .lead-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
    }

    .status-new {
      color: var(--info-color);
    }

    .status-inspected {
      color: var(--secondary-color);
    }

    .status-qualified {
      color: var(--purple-color);
    }

    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: currentColor;
      animation: statusPulse 2s ease-in-out infinite;
    }

    @keyframes statusPulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .lead-body {
      padding: 1.5rem;
    }

    .lead-section {
      margin-bottom: 1.25rem;
    }

    .lead-section:last-child {
      margin-bottom: 0;
    }

    .section-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--gray-600);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .section-content {
      color: var(--gray-800);
      font-size: 0.9375rem;
    }

    .event-info {
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .event-type {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.375rem 0.875rem;
      background: rgba(37, 99, 235, 0.1);
      color: var(--primary-color);
      border-radius: 0.5rem;
      font-weight: 500;
      font-size: 0.875rem;
      border: 1px solid rgba(37, 99, 235, 0.2);
    }

    .magnitude {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-weight: 600;
      color: var(--gray-700);
      padding: 0.375rem 0.875rem;
      background: var(--gray-100);
      border-radius: 0.5rem;
    }

    .location {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--gray-600);
    }

    .timestamp {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--gray-500);
      font-size: 0.875rem;
    }

    .inspection-data {
      background: linear-gradient(135deg, var(--gray-50), white);
      padding: 1.25rem;
      border-radius: var(--border-radius-sm);
      border-left: 4px solid var(--secondary-color);
      border: 1px solid var(--gray-200);
    }

    .inspection-notes {
      margin-bottom: 1rem;
      color: var(--gray-700);
      line-height: 1.5;
    }

    .inspection-photos {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .photo-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: var(--gray-100);
      border-radius: 0.5rem;
      font-size: 0.875rem;
      color: var(--gray-600);
      border: 1px solid var(--gray-200);
      transition: var(--transition);
    }

    .photo-item:hover {
      background: var(--gray-200);
    }

    .lead-actions {
      padding: 1.25rem 1.5rem;
      background: linear-gradient(135deg, var(--gray-50), white);
      border-top: 1px solid var(--gray-100);
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .btn {
      padding: 0.625rem 1.25rem;
      border: none;
      border-radius: var(--border-radius-sm);
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: var(--transition);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      text-decoration: none;
      position: relative;
      overflow: hidden;
    }

    .btn::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
      transition: var(--transition);
    }

    .btn:hover::before {
      left: 100%;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
      color: white;
      box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
    }

    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 12px rgba(37, 99, 235, 0.4);
    }

    .btn-secondary {
      background: var(--gray-200);
      color: var(--gray-700);
      border: 1px solid var(--gray-300);
    }

    .btn-secondary:hover {
      background: var(--gray-300);
      transform: translateY(-2px);
    }

    .btn-success {
      background: linear-gradient(135deg, var(--secondary-color), var(--secondary-dark));
      color: white;
      box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }

    .btn-success:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4);
    }

    /* Map Integration */
    .map-container {
      height: 300px;
      background: var(--gray-100);
      border-radius: var(--border-radius);
      margin: 1rem 0;
      position: relative;
      overflow: hidden;
      border: 2px solid var(--gray-200);
    }

    .map-placeholder {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: var(--gray-500);
      font-size: 1.125rem;
    }

    /* Photo Upload Interface */
    .photo-upload {
      margin: 1rem 0;
      padding: 1.5rem;
      background: linear-gradient(135deg, var(--gray-50), white);
      border-radius: var(--border-radius);
      border: 2px dashed var(--gray-300);
      text-align: center;
      transition: var(--transition);
    }

    .photo-upload:hover {
      border-color: var(--primary-color);
      background: rgba(37, 99, 235, 0.05);
    }

    .photo-upload.dragover {
      border-color: var(--primary-color);
      background: rgba(37, 99, 235, 0.1);
    }

    .upload-btn {
      padding: 0.75rem 1.5rem;
      background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
      color: white;
      border: none;
      border-radius: var(--border-radius);
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: var(--transition);
      margin-top: 1rem;
    }

    .upload-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 12px rgba(37, 99, 235, 0.4);
    }

    /* Progress Indicators */
    .progress-bar {
      width: 100%;
      height: 8px;
      background: var(--gray-200);
      border-radius: 4px;
      overflow: hidden;
      margin: 1rem 0;
    }

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
      border-radius: 4px;
      transition: width 0.3s ease;
    }

    /* Empty State */
    .empty-state {
      text-align: center;
      padding: 4rem 2rem;
      color: var(--gray-500);
    }

    .empty-state i {
      font-size: 4rem;
      margin-bottom: 1rem;
      color: var(--gray-300);
    }

    .empty-state h3 {
      font-size: 1.5rem;
      margin-bottom: 0.5rem;
      color: var(--gray-700);
    }

    /* Loading State */
    .loading {
      text-align: center;
      padding: 2rem;
      color: var(--gray-500);
    }

    .loading i {
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    /* Mobile-first responsive design */
    @media (max-width: 768px) {
      .header-content {
        flex-direction: column;
        text-align: center;
      }

      .mobile-menu-toggle {
        display: block;
      }

      .stats-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
      }

      .filters-grid {
        grid-template-columns: 1fr;
      }

      .leads-grid {
        grid-template-columns: 1fr;
      }

      .section-header {
        flex-direction: column;
        gap: 1rem;
        align-items: flex-start;
      }

      .event-info {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
      }

      .lead-actions {
        flex-direction: column;
      }

      .btn {
        justify-content: center;
        width: 100%;
      }
      
      /* Touch-friendly interface */
      .btn {
        min-height: 44px; /* iOS touch target minimum */
        font-size: 1rem;
        padding: 0.75rem 1rem;
      }
      
      .lead-card {
        margin-bottom: 1rem;
      }
      
      .filter-input {
        font-size: 1rem;
        padding: 0.875rem 1rem;
      }
      
      /* Mobile-specific camera button */
      .camera-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: var(--transition);
      }
      
      .camera-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
      }
      
      .camera-btn:active {
        transform: scale(0.95);
      }
      
      /* Offline indicator */
      .offline-indicator {
        position: fixed;
        top: 60px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--warning-color);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        z-index: 1000;
        display: none;
      }
      
      .offline-indicator.show {
        display: block;
      }
      
      /* GPS status indicator */
      .gps-status {
        position: fixed;
        top: 60px;
        right: 20px;
        background: var(--success-color);
        color: white;
        padding: 0.375rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        z-index: 1000;
        display: none;
      }
      
      .gps-status.active {
        display: block;
      }
      
      .gps-status.error {
        background: var(--danger-color);
      }
    }

    @media (max-width: 480px) {
      .container {
        padding: 0 0.75rem;
      }

      .header-content {
        padding: 1rem 0;
      }

      .logo {
        font-size: 1.5rem;
      }

      .logo i {
        font-size: 1.75rem;
      }

      .stats-grid {
        grid-template-columns: 1fr;
      }

      .stat-value {
        font-size: 1.5rem;
      }

      .stat-label {
        font-size: 0.8rem;
      }

      .leads-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
      }
    }

    /* Tablet specific styles */
    @media (min-width: 769px) and (max-width: 1024px) {
      .leads-grid {
        grid-template-columns: repeat(2, 1fr);
      }
      
      .stats-grid {
        grid-template-columns: repeat(3, 1fr);
      }
    }

    /* Print styles */
    @media print {
      header, .filters-section, .lead-actions {
        display: none;
      }

      .leads-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
      }

      .lead-card {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid var(--gray-300);
      }
    }

    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
      * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :root {
        --gray-50: #1f2937;
        --gray-100: #111827;
        --gray-200: #1f2937;
        --gray-300: #374151;
        --gray-400: #6b7280;
        --gray-500: #9ca3af;
        --gray-600: #d1d5db;
        --gray-700: #e5e7eb;
        --gray-800: #f3f4f6;
        --gray-900: #f9fafb;
      }

      body {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: var(--gray-100);
      }

      .lead-card {
        background: rgba(31, 41, 55, 0.9);
        border-color: rgba(75, 85, 99, 0.3);
      }

      .stat-card {
        background: rgba(31, 41, 55, 0.9);
        border-color: rgba(75, 85, 99, 0.3);
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="container">
      <div class="header-content">
        <a href="/" class="logo">
          <i class="fas fa-cloud-showers-heavy"></i>
          <span>Hail Hero</span>
        </a>
        <div class="header-actions">
          <div class="stats">
            <div class="stat-item">
              <div class="stat-value">{{ leads|length }}</div>
              <div class="stat-label">Total Leads</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">{{ leads|selectattr('status', 'equalto', 'new')|list|length }}</div>
              <div class="stat-label">New</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">{{ leads|selectattr('status', 'equalto', 'inspected')|list|length }}</div>
              <div class="stat-label">Inspected</div>
            </div>
          </div>
          <button class="mobile-menu-toggle" onclick="toggleMobileMenu()">
            <i class="fas fa-bars"></i>
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Mobile-specific elements -->
  <div class="offline-indicator" id="offlineIndicator">
    <i class="fas fa-wifi-slash"></i> Offline Mode
  </div>
  
  <div class="gps-status" id="gpsStatus">
    <i class="fas fa-map-marker-alt"></i> GPS Active
  </div>
  
  <!-- Mobile Camera Button -->
  <button class="camera-btn" id="cameraBtn" onclick="openMobileCamera()" style="display: none;">
    <i class="fas fa-camera"></i>
  </button>

  <main class="container">
    <!-- Enhanced Stats Dashboard -->
    <div class="stats-dashboard">
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon">
            <i class="fas fa-users"></i>
          </div>
          <div class="stat-value">{{ leads|length }}</div>
          <div class="stat-label">Total Leads</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <i class="fas fa-clock"></i>
          </div>
          <div class="stat-value">{{ leads|selectattr('status', 'equalto', 'new')|list|length }}</div>
          <div class="stat-label">New Leads</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <i class="fas fa-check-circle"></i>
          </div>
          <div class="stat-value">{{ leads|selectattr('status', 'equalto', 'inspected')|list|length }}</div>
          <div class="stat-label">Inspected</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <i class="fas fa-star"></i>
          </div>
          <div class="stat-value">{{ "%.1f"|format(leads|selectattr('score')|map(attribute='score')|list|average|round(1) if leads else 0) }}</div>
          <div class="stat-label">Avg Score</div>
        </div>
      </div>
    </div>

    <!-- Advanced Filters -->
    <div class="filters-section">
      <div class="filters-header">
        <div class="filters-title">
          <i class="fas fa-filter"></i>
          Filters & Search
        </div>
      </div>
      <div class="filters-grid">
        <div class="filter-group">
          <label class="filter-label">Search Leads</label>
          <div class="search-box">
            <i class="fas fa-search"></i>
            <input type="text" class="filter-input" id="searchInput" placeholder="Search by ID, location...">
          </div>
        </div>
        <div class="filter-group">
          <label class="filter-label">Status</label>
          <select class="filter-input" id="statusFilter">
            <option value="">All Statuses</option>
            <option value="new">New</option>
            <option value="inspected">Inspected</option>
            <option value="qualified">Qualified</option>
          </select>
        </div>
        <div class="filter-group">
          <label class="filter-label">Sort by</label>
          <select class="filter-input" id="sortBy">
            <option value="score">Score</option>
            <option value="created_ts">Date Created</option>
            <option value="lead_id">Lead ID</option>
            <option value="magnitude">Magnitude</option>
          </select>
        </div>
        <div class="filter-group">
          <label class="filter-label">Order</label>
          <select class="filter-input" id="sortOrder">
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Leads Section -->
    <div class="leads-section">
      <div class="section-header">
        <div class="section-title">
          <i class="fas fa-list-ul"></i>
          Leads ({{ leads|length }})
        </div>
        <div class="view-toggle">
          <button class="view-btn active" onclick="setViewMode('grid')">
            <i class="fas fa-th"></i>
          </button>
          <button class="view-btn" onclick="setViewMode('list')">
            <i class="fas fa-list"></i>
          </button>
        </div>
      </div>

      <div class="leads-grid" id="leadsGrid">
        {% for lead in leads %}
        {% set status = lead.get('status', 'new') %}
        {% set prop = lead.get('property', {}) %}
        {% set ev = lead.get('event', {}) %}
        {% set score = lead.get('score', 0) %}
        {% set magnitude = ev.get('MAGNITUDE') or ev.get('mag') or 0 %}
        
        <div class="lead-card {% if score >= 20 %}priority-high{% elif score >= 15 %}priority-medium{% else %}priority-low{% endif %}" 
             data-status="{{ status }}" 
             data-score="{{ score }}" 
             data-created="{{ lead.get('created_ts', '') }}"
             data-magnitude="{{ magnitude }}"
             data-lead-id="{{ lead.get('lead_id') }}"
             data-location="{{ "%.4f"|format(prop.get('lat', 0)) }}, {{ "%.4f"|format(prop.get('lon', 0)) }}">
          <div class="lead-header">
            <div class="lead-title">
              <div class="lead-id">
                <i class="fas fa-id-card"></i>
                {{ lead.get('lead_id') }}
              </div>
              <div class="lead-badge {% if score >= 20 %}score-high{% elif score >= 15 %}score-medium{% else %}score-low{% endif %}">
                <i class="fas fa-star"></i>
                {{ score }}
              </div>
            </div>
            <div class="lead-status status-{{ status }}">
              <div class="status-dot"></div>
              {{ status|title }}
            </div>
          </div>
          
          <div class="lead-body">
            <div class="lead-section">
              <div class="section-title">
                <i class="fas fa-cloud-bolt"></i>
                Event Information
              </div>
              <div class="section-content">
                <div class="event-info">
                  <div class="event-type">
                    <i class="fas fa-cloud-showers-heavy"></i>
                    {{ ev.get('EVENT_TYPE') or ev.get('eventType') or 'Unknown' }}
                  </div>
                  <div class="magnitude">
                    <i class="fas fa-gauge-high"></i>
                    Magnitude: {{ "%.1f"|format(magnitude) }}
                  </div>
                </div>
                {% if lead.get('scoring_details') %}
                <div class="progress-bar">
                  <div class="progress-fill" style="width: {{ (score / 30 * 100)|round }}%"></div>
                </div>
                {% endif %}
              </div>
            </div>

            <div class="lead-section">
              <div class="section-title">
                <i class="fas fa-map-marker-alt"></i>
                Location
              </div>
              <div class="section-content">
                <div class="location">
                  <i class="fas fa-map-pin"></i>
                  {{ "%.4f"|format(prop.get('lat', 0)) }}, {{ "%.4f"|format(prop.get('lon', 0)) }}
                </div>
                <div class="map-container">
                  <div class="map-placeholder">
                    <i class="fas fa-map"></i>
                    Map View Available
                  </div>
                </div>
              </div>
            </div>

            <div class="lead-section">
              <div class="section-title">
                <i class="fas fa-clock"></i>
                Timeline
              </div>
              <div class="section-content">
                <div class="timestamp">
                  <i class="fas fa-calendar-alt"></i>
                  Created: {{ lead.get('created_ts') or lead.get('timestamp') or 'N/A' }}
                </div>
                {% if lead.get('inspection') %}
                <div class="timestamp">
                  <i class="fas fa-check-circle"></i>
                  Inspected: {{ lead['inspection'].get('timestamp', 'N/A') }}
                </div>
                {% endif %}
              </div>
            </div>

            {% if lead.get('inspection') %}
            <div class="lead-section">
              <div class="section-title">
                <i class="fas fa-clipboard-check"></i>
                Inspection Details
              </div>
              <div class="section-content">
                <div class="inspection-data">
                  {% if lead['inspection'].get('notes') %}
                  <div class="inspection-notes">
                    <strong>Notes:</strong> {{ lead['inspection'].get('notes') }}
                  </div>
                  {% endif %}
                  {% if lead['inspection'].get('photos') %}
                  <div class="inspection-photos">
                    {% for photo in lead['inspection'].get('photos', []) %}
                    <div class="photo-item">
                      <i class="fas fa-camera"></i>
                      Photo {{ loop.index }}
                    </div>
                    {% endfor %}
                  </div>
                  {% endif %}
                </div>
              </div>
            </div>
            {% endif %}

            {% if status == 'new' %}
            <div class="lead-section">
              <div class="section-title">
                <i class="fas fa-camera"></i>
                Photo Upload
              </div>
              <div class="section-content">
                <div class="photo-upload" id="upload-{{ lead.get('lead_id') }}">
                  <i class="fas fa-cloud-upload-alt" style="font-size: 2rem; color: var(--gray-400); margin-bottom: 0.5rem;"></i>
                  <p>Drag & drop photos here or click to upload</p>
                  <button class="upload-btn" onclick="openCamera('{{ lead.get('lead_id') }}')">
                    <i class="fas fa-camera"></i>
                    Take Photo
                  </button>
                </div>
              </div>
            </div>
            {% endif %}
          </div>

          <div class="lead-actions">
            <button class="btn btn-primary" onclick="viewDetails('{{ lead.get('lead_id') }}')">
              <i class="fas fa-eye"></i>
              View Details
            </button>
            {% if status == 'new' %}
            <button class="btn btn-success" onclick="markInspected('{{ lead.get('lead_id') }}')">
              <i class="fas fa-check"></i>
              Mark Inspected
            </button>
            {% endif %}
            <button class="btn btn-secondary" onclick="showLocation('{{ prop.get('lat', 0) }}', '{{ prop.get('lon', 0) }}')">
              <i class="fas fa-map"></i>
              Show on Map
            </button>
            {% if status == 'new' %}
            <button class="btn btn-secondary" onclick="callLead('{{ lead.get('lead_id') }}')">
              <i class="fas fa-phone"></i>
              Call
            </button>
            {% endif %}
          </div>
        </div>
        {% endfor %}
      </div>

      {% if not leads %}
      <div class="empty-state">
        <i class="fas fa-cloud-showers-heavy"></i>
        <h3>No Leads Found</h3>
        <p>There are currently no leads available in the system.</p>
      </div>
      {% endif %}
    </div>
  </main>

  <script>
    // Mobile-first functionality
    class MobileApp {
      constructor() {
        this.isMobile = window.innerWidth <= 768;
        this.isOnline = navigator.onLine;
        this.currentPosition = null;
        this.watchId = null;
        this.offlineQueue = [];
        
        this.init();
      }
      
      init() {
        this.setupNetworkListeners();
        this.setupGPS();
        this.setupMobileUI();
        this.loadOfflineData();
        this.setupServiceWorker();
      }
      
      setupNetworkListeners() {
        window.addEventListener('online', () => {
          this.isOnline = true;
          this.hideOfflineIndicator();
          this.syncOfflineData();
        });
        
        window.addEventListener('offline', () => {
          this.isOnline = false;
          this.showOfflineIndicator();
        });
      }
      
      setupGPS() {
        if (navigator.geolocation) {
          this.watchId = navigator.geolocation.watchPosition(
            (position) => {
              this.currentPosition = {
                lat: position.coords.latitude,
                lon: position.coords.longitude,
                accuracy: position.coords.accuracy,
                timestamp: position.timestamp
              };
              this.showGPSStatus();
            },
            (error) => {
              this.showGPSError(error);
            },
            {
              enableHighAccuracy: true,
              timeout: 10000,
              maximumAge: 300000 // 5 minutes
            }
          );
        }
      }
      
      setupMobileUI() {
        // Show camera button on mobile
        const cameraBtn = document.getElementById('cameraBtn');
        if (this.isMobile && cameraBtn) {
          cameraBtn.style.display = 'flex';
        }
        
        // Touch-friendly interactions
        this.setupTouchInteractions();
        
        // Responsive handling
        window.addEventListener('resize', () => {
          this.isMobile = window.innerWidth <= 768;
          if (cameraBtn) {
            cameraBtn.style.display = this.isMobile ? 'flex' : 'none';
          }
        });
      }
      
      setupTouchInteractions() {
        // Add touch feedback to buttons
        document.querySelectorAll('.btn').forEach(btn => {
          btn.addEventListener('touchstart', () => {
            btn.style.transform = 'scale(0.95)';
          });
          
          btn.addEventListener('touchend', () => {
            btn.style.transform = 'scale(1)';
          });
        });
      }
      
      setupServiceWorker() {
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
              console.log('ServiceWorker registration successful');
            })
            .catch(error => {
              console.log('ServiceWorker registration failed:', error);
            });
        }
      }
      
      showOfflineIndicator() {
        const indicator = document.getElementById('offlineIndicator');
        if (indicator) {
          indicator.classList.add('show');
        }
      }
      
      hideOfflineIndicator() {
        const indicator = document.getElementById('offlineIndicator');
        if (indicator) {
          indicator.classList.remove('show');
        }
      }
      
      showGPSStatus() {
        const gpsStatus = document.getElementById('gpsStatus');
        if (gpsStatus) {
          gpsStatus.classList.add('active');
          gpsStatus.classList.remove('error');
        }
      }
      
      showGPSError(error) {
        const gpsStatus = document.getElementById('gpsStatus');
        if (gpsStatus) {
          gpsStatus.classList.add('active', 'error');
          gpsStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> GPS Error';
        }
      }
      
      async loadOfflineData() {
        // Load cached data from IndexedDB
        try {
          const cachedLeads = localStorage.getItem('cachedLeads');
          if (cachedLeads && !this.isOnline) {
            // Use cached data when offline
            console.log('Using cached leads data');
          }
        } catch (error) {
          console.error('Error loading offline data:', error);
        }
      }
      
      async syncOfflineData() {
        if (this.offlineQueue.length === 0) return;
        
        try {
          for (const item of this.offlineQueue) {
            await this.syncItem(item);
          }
          this.offlineQueue = [];
          showNotification('Offline data synced successfully', 'success');
        } catch (error) {
          console.error('Error syncing offline data:', error);
          showNotification('Error syncing offline data', 'error');
        }
      }
      
      async syncItem(item) {
        // Sync individual item to server
        const response = await fetch(item.url, {
          method: item.method,
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(item.data)
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }
      
      addToOfflineQueue(url, method, data) {
        this.offlineQueue.push({ url, method, data, timestamp: Date.now() });
        localStorage.setItem('offlineQueue', JSON.stringify(this.offlineQueue));
      }
      
      getCurrentLocation() {
        return new Promise((resolve, reject) => {
          if (!this.currentPosition) {
            reject(new Error('GPS location not available'));
            return;
          }
          
          resolve(this.currentPosition);
        });
      }
    }
    
    // Initialize mobile app
    const mobileApp = new MobileApp();
    
    // Enhanced filter and search functionality
    document.getElementById('searchInput').addEventListener('input', filterLeads);
    document.getElementById('statusFilter').addEventListener('change', filterLeads);
    document.getElementById('sortBy').addEventListener('change', filterLeads);
    document.getElementById('sortOrder').addEventListener('change', filterLeads);

    function filterLeads() {
      const searchTerm = document.getElementById('searchInput').value.toLowerCase();
      const statusFilter = document.getElementById('statusFilter').value;
      const sortBy = document.getElementById('sortBy').value;
      const sortOrder = document.getElementById('sortOrder').value;
      const leads = document.querySelectorAll('.lead-card');
      
      let filteredLeads = Array.from(leads);
      
      // Search functionality
      if (searchTerm) {
        filteredLeads = filteredLeads.filter(lead => {
          const leadId = lead.dataset.leadId.toLowerCase();
          const location = lead.dataset.location.toLowerCase();
          return leadId.includes(searchTerm) || location.includes(searchTerm);
        });
      }
      
      // Filter by status
      if (statusFilter) {
        filteredLeads = filteredLeads.filter(lead => lead.dataset.status === statusFilter);
      }
      
      // Sort leads
      filteredLeads.sort((a, b) => {
        let aValue, bValue;
        
        switch(sortBy) {
          case 'score':
            aValue = parseFloat(a.dataset.score);
            bValue = parseFloat(b.dataset.score);
            break;
          case 'magnitude':
            aValue = parseFloat(a.dataset.magnitude);
            bValue = parseFloat(b.dataset.magnitude);
            break;
          case 'created_ts':
            aValue = new Date(a.dataset.created);
            bValue = new Date(b.dataset.created);
            break;
          case 'lead_id':
            aValue = a.dataset.leadId;
            bValue = b.dataset.leadId;
            break;
        }
        
        if (sortOrder === 'asc') {
          return aValue > bValue ? 1 : -1;
        } else {
          return aValue < bValue ? 1 : -1;
        }
      });
      
      // Reorder leads in the grid
      const grid = document.getElementById('leadsGrid');
      filteredLeads.forEach(lead => grid.appendChild(lead));
      
      // Hide filtered out leads
      leads.forEach(lead => {
        const shouldShow = (!searchTerm || lead.dataset.leadId.toLowerCase().includes(searchTerm) || lead.dataset.location.toLowerCase().includes(searchTerm)) &&
                         (!statusFilter || lead.dataset.status === statusFilter);
        lead.style.display = shouldShow ? 'block' : 'none';
      });
    }

    // View mode toggle
    function setViewMode(mode) {
      const grid = document.getElementById('leadsGrid');
      const buttons = document.querySelectorAll('.view-btn');
      
      buttons.forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');
      
      if (mode === 'list') {
        grid.classList.add('list-view');
      } else {
        grid.classList.remove('list-view');
      }
    }

    // Lead action functions
    function viewDetails(leadId) {
      // Create modal instead of alert
      const modal = document.createElement('div');
      modal.style.cssText = `
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.5); display: flex; align-items: center;
        justify-content: center; z-index: 10000;
      `;
      modal.innerHTML = `
        <div style="background: white; padding: 2rem; border-radius: 1rem; max-width: 500px; width: 90%;">
          <h3 style="margin-bottom: 1rem;">Lead Details</h3>
          <p>Viewing detailed information for lead: <strong>${leadId}</strong></p>
          <div style="margin-top: 1.5rem; display: flex; gap: 1rem;">
            <button onclick="this.closest('div[style*=fixed]').remove()" 
                    style="padding: 0.5rem 1rem; background: #2563eb; color: white; border: none; border-radius: 0.5rem; cursor: pointer;">
              Close
            </button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
    }

    function markInspected(leadId) {
      if (confirm('Are you sure you want to mark this lead as inspected?')) {
        // Show loading state
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        btn.disabled = true;
        
        fetch('/api/inspection', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            lead_id: leadId,
            notes: 'Inspected via web interface',
            photos: []
          })
        })
        .then(response => response.json())
        .then(data => {
          showNotification('Lead marked as inspected successfully!', 'success');
          setTimeout(() => location.reload(), 1000);
        })
        .catch(error => {
          console.error('Error:', error);
          showNotification('Error marking lead as inspected', 'error');
          btn.innerHTML = originalText;
          btn.disabled = false;
        });
      }
    }

    function showLocation(lat, lon) {
      // Open Google Maps in a new tab
      window.open(`https://www.google.com/maps?q=${lat},${lon}`, '_blank');
    }

    function callLead(leadId) {
      // In a real app, this would integrate with a calling system
      showNotification('Calling feature would be integrated here', 'info');
    }

    function openCamera(leadId) {
      // Check if camera API is available
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(function(stream) {
            // Camera access granted
            showNotification('Camera access granted. Photo capture interface would open here.', 'success');
            // In a real app, this would open a camera interface
          })
          .catch(function(err) {
            showNotification('Camera access denied or not available', 'error');
          });
      } else {
        showNotification('Camera not available on this device', 'error');
      }
    }

    function showNotification(message, type = 'info') {
      const notification = document.createElement('div');
      notification.style.cssText = `
        position: fixed; top: 20px; right: 20px; padding: 1rem 1.5rem;
        border-radius: 0.5rem; color: white; z-index: 10000;
        max-width: 300px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease;
      `;
      
      const colors = {
        success: '#10b981',
        error: '#ef4444',
        info: '#3b82f6',
        warning: '#f59e0b'
      };
      
      notification.style.background = colors[type] || colors.info;
      notification.textContent = message;
      
      document.body.appendChild(notification);
      
      setTimeout(() => {
        notification.remove();
      }, 3000);
    }

    function toggleMobileMenu() {
      // Mobile menu toggle functionality
      showNotification('Mobile menu would open here', 'info');
    }

    // Photo upload drag and drop
    document.addEventListener('DOMContentLoaded', function() {
      const uploadAreas = document.querySelectorAll('.photo-upload');
      
      uploadAreas.forEach(area => {
        area.addEventListener('dragover', function(e) {
          e.preventDefault();
          this.classList.add('dragover');
        });
        
        area.addEventListener('dragleave', function(e) {
          e.preventDefault();
          this.classList.remove('dragover');
        });
        
        area.addEventListener('drop', function(e) {
          e.preventDefault();
          this.classList.remove('dragover');
          
          const files = e.dataTransfer.files;
          if (files.length > 0) {
            showNotification(`${files.length} file(s) uploaded successfully`, 'success');
          }
        });
      });
    });

    // Auto-refresh functionality (optional)
    setInterval(() => {
      // Uncomment to enable auto-refresh
      // location.reload();
    }, 30000); // Refresh every 30 seconds

    // Add CSS animation for notifications
    const style = document.createElement('style');
    style.textContent = `
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
    `;
    document.head.appendChild(style);
  </script>
</body>
</html>
"""

# --- API Endpoints ---
@app.route('/')
def list_leads():
    """Display the list of leads with modern UI."""
    sorted_leads = sorted(leads_db, key=lambda l: l.get('timestamp', ''), reverse=True)
    return render_template_string(HTML_TEMPLATE, leads=sorted_leads)

@app.route('/classic')
def classic_ui():
    """Display the classic UI for comparison."""
    sorted_leads = sorted(leads_db, key=lambda l: l.get('timestamp', ''), reverse=True)
    return render_template_string(HTML_TEMPLATE, leads=sorted_leads)

@app.route('/api/leads', methods=['GET'])
def get_leads():
    """Return all leads as JSON."""
    return jsonify(leads_db)

@app.route('/api/inspection', methods=['POST'])
def submit_inspection():
    """Webhook to accept an inspection submission."""
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    lead_id = data.get('lead_id')
    if not lead_id or lead_id not in [lead['lead_id'] for lead in leads_db]:
        return jsonify({'error': 'Invalid lead_id'}), 404

    for lead in leads_db:
        if lead['lead_id'] == lead_id:
            lead['status'] = 'inspected'
            lead['inspection'] = {
                'notes': data.get('notes'),
                'photos': data.get('photos', []),
                'timestamp': datetime.datetime.utcnow().isoformat(),
            }
            break

    # In a real app, you'd save this back to a persistent store
    return jsonify(next(lead for lead in leads_db if lead['lead_id'] == lead_id))

@app.route('/twilio/sms', methods=['POST'])
def handle_twilio_sms():
    """Placeholder for Twilio SMS webhook."""
    # from_number = request.form['From']
    # body = request.form['Body']
    # In a real app, match number to lead and handle reply
    return "<Response></Response>", 200, {'Content-Type': 'application/xml'}

# Load leads at app startup
load_leads()

if __name__ == '__main__':
    app.run(debug=True)
