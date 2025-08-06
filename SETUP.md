# Ethical AI Auditor - Setup Guide

## Prerequisites

- Python 3.8+ (recommended: Python 3.11+)
- pip package manager

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd "SAP new"
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize database** (if needed):
   ```bash
   python init_db.py
   ```

5. **Run the application**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Dependencies Overview

### Core Framework
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation and settings management

### AI/ML Libraries
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **SHAP**: Model explainability
- **AIF360**: AI fairness toolkit
- **LIME**: Local interpretable model-agnostic explanations

### Database & Storage
- **DuckDB**: In-process analytical database

### Additional Tools
- **MLflow**: ML lifecycle management
- **HTTPX/Requests**: HTTP client libraries
- **FPDF2**: PDF generation for reports
- **python-dotenv**: Environment variable management

## API Endpoints

Once running, the API will be available at:
- **API Documentation**: http://localhost:8000/docs
- **Bias Detection**: POST `/api/v1/bias/detect`
- **Model Explanation**: POST `/api/v1/explain`
- **Compliance Reports**: POST `/api/v1/compliance/generate-report`

## Project Structure

```
SAP new/
├── app/
│   ├── config.py              # Configuration settings
│   ├── main.py               # FastAPI application
│   ├── routers/              # API route handlers
│   ├── services/             # Business logic
│   ├── models/               # Pydantic models
│   ├── utils/                # Utility functions
│   └── database/             # Database connections
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
└── hiring_data.csv          # Sample dataset
```

## Development

For development, you may want to install additional packages:
```bash
pip install pytest pytest-asyncio black flake8
```

## Troubleshooting

If you encounter dependency issues:
1. Ensure you're using Python 3.8+
2. Try upgrading pip: `pip install --upgrade pip`
3. Clear pip cache: `pip cache purge`
4. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## SAP Integration Notes

This project is designed to be compatible with:
- SAP Business Technology Platform (BTP)
- SAP AI Core for model deployment
- SAP Analytics Cloud for dashboard integration
- Alternative to SAP Joule for AI explainability