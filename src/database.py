import time
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import settings
from src.models import Base

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(max_retries: int = 10, delay_seconds: int = 3):
    """Wait for database availability and ensure tables are created."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to database (attempt {attempt}/{max_retries})...")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables verified/created successfully.")
            return True
        except Exception as e:
            logger.warning(f"Database connection failed on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)
            else:
                raise RuntimeError(f"Could not connect to database after {max_retries} attempts.") from e
