from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import bias, explain, compliance, dashboard
from app.config import settings



app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="API for Ethical AI Auditor - Ensuring fair, transparent, and compliant AI systems",
    debug=True
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(dashboard.router, prefix=settings.API_V1_STR)
app.include_router(bias.router, prefix=settings.API_V1_STR)
app.include_router(explain.router, prefix=settings.API_V1_STR)
app.include_router(compliance.router, prefix=settings.API_V1_STR)
app.include_router(dashboard.router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {
        "message": "Ethical AI Auditor API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "bias_detection": f"{settings.API_V1_STR}/bias/detect",
            "explanation": f"{settings.API_V1_STR}/explain"
        }
    }