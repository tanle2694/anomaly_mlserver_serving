from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import infer, models

app = FastAPI(title="Anomaly Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(models.router, prefix="/api")
app.include_router(infer.router, prefix="/api")
