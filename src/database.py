"""
Database abstraction layer for Hail Hero.

Provides a unified interface for both SQLite (development) and PostgreSQL (production)
with support for SQLAlchemy ORM and raw SQL queries.
"""

import os
import logging
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.pool import StaticPool, NullPool
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Create declarative base for models
Base = declarative_base()


class DatabaseManager:
    """
    Manages database connections and provides a unified interface
    for both SQLite and PostgreSQL databases.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            database_url: Database connection string.
                         If None, uses DATABASE_URL environment variable.
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'sqlite:///data/hailhero.db'
        )
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._setup_engine()

    def _setup_engine(self):
        """Configure SQLAlchemy engine based on database type."""
        logger.info(f"Setting up database engine: {self.database_url.split('@')[0]}...")

        # Configure engine based on database type
        if self.database_url.startswith('sqlite'):
            # SQLite configuration
            # Extract database path and ensure directory exists
            db_path = self.database_url.replace('sqlite:///', '')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=os.getenv('DEBUG', 'false').lower() == 'true'
            )

            # Enable foreign keys for SQLite
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        elif self.database_url.startswith('postgresql'):
            # PostgreSQL configuration
            self.engine = create_engine(
                self.database_url,
                poolclass=NullPool,
                pool_pre_ping=True,
                echo=os.getenv('DEBUG', 'false').lower() == 'true'
            )
        else:
            raise ValueError(f"Unsupported database type: {self.database_url}")

        # Create session factory
        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        )

        logger.info("Database engine configured successfully")

    def create_tables(self):
        """Create all tables defined in models."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.

        Yields:
            Session: SQLAlchemy session object

        Example:
            with db_manager.get_session() as session:
                user = session.query(User).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def execute_raw(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query and return results.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List of dictionaries containing query results
        """
        with self.engine.connect() as connection:
            result = connection.execute(text(query), params or {})
            if result.returns_rows:
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
            return []

    def execute_script(self, script_path: str):
        """
        Execute SQL script from file.

        Args:
            script_path: Path to SQL script file
        """
        logger.info(f"Executing SQL script: {script_path}")

        with open(script_path, 'r') as f:
            script = f.read()

        with self.engine.connect() as connection:
            # Split on semicolons and execute each statement
            for statement in script.split(';'):
                statement = statement.strip()
                if statement:
                    connection.execute(text(statement))
                    connection.commit()

        logger.info(f"SQL script executed successfully: {script_path}")

    def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get database connection information.

        Returns:
            Dictionary with connection details
        """
        return {
            'database_type': 'sqlite' if 'sqlite' in self.database_url else 'postgresql',
            'url': self.database_url.split('@')[0] if '@' in self.database_url else self.database_url.split('///')[0],
            'pool_size': self.engine.pool.size() if hasattr(self.engine.pool, 'size') else 'N/A',
            'echo': self.engine.echo,
        }

    def close(self):
        """Close database connections."""
        if self.SessionLocal:
            self.SessionLocal.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get global database manager instance (singleton pattern).

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_db(database_url: Optional[str] = None):
    """
    Initialize database with tables.

    Args:
        database_url: Optional database URL override
    """
    db_manager = DatabaseManager(database_url) if database_url else get_db_manager()
    db_manager.create_tables()
    return db_manager


def get_session():
    """
    Get database session (for use with dependency injection).

    Yields:
        Session: SQLAlchemy session
    """
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        yield session
