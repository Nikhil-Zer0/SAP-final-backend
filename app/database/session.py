import duckdb
from typing import Generator
from app.config import settings

def get_db() -> Generator:
    """Create and yield a database connection"""
    try:
        # Connect to DuckDB (creates file if it doesn't exist)
        conn = duckdb.connect(settings.DATABASE_URL.replace("duckdb:", ""))
        conn.execute("PRAGMA threads=4")  # Use multiple threads for analytics
        conn.execute("PRAGMA enable_progress_bar")  # Show progress for long queries
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the database with required tables"""
    conn = duckdb.connect(settings.DATABASE_URL.replace("duckdb:", ""))
    
    # Create bias detection tables
    conn.execute("""
    CREATE TABLE IF NOT EXISTS bias_metrics (
        id INTEGER PRIMARY KEY,
        model_name VARCHAR,
        model_version VARCHAR,
        dataset_name VARCHAR,
        sensitive_attribute VARCHAR,
        disparate_impact DOUBLE,
        statistical_parity DOUBLE,
        equal_opportunity DOUBLE,
        detection_timestamp TIMESTAMP,
        audit_status VARCHAR
    )
    """)
    
    # Create compliance reports table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS compliance_reports (
        id INTEGER PRIMARY KEY,
        report_name VARCHAR,
        report_type VARCHAR,
        model_name VARCHAR,
        generation_timestamp TIMESTAMP,
        regulatory_framework VARCHAR,
        pdf_path VARCHAR,
        status VARCHAR
    )
    """)
    
    # Create model registry table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY,
        model_name VARCHAR UNIQUE,
        model_version VARCHAR,
        description VARCHAR,
        created_at TIMESTAMP,
        trained_dataset VARCHAR,
        fairness_score DOUBLE,
        is_active BOOLEAN
    )
    """)
    
    conn.close()
    print("Database initialized successfully with all required tables")