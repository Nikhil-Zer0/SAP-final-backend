import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Ethical AI Auditor"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "duckdb:ai_auditor.duckdb")
    
    # Model configuration
    FAIRNESS_THRESHOLD: float = 0.8  # EEOC 80% rule
    SENSITIVE_ATTRIBUTES: list = ["gender", "age", "ethnicity"]
    
    # AI model paths
    DEFAULT_MODEL_PATH: str = "models/default_model.pkl"

settings = Settings()