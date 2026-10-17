"""Run: python -m uvicorn backend.app:app --reload --port 8000."""
import asyncio
import os
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

from .core import ApiError, GitHub, analyze, load_model, parse_repository


def load_environment():
    """Read only supported settings; never execute or interpolate .env content."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key in {"GITHUB_TOKEN", "CORS_ORIGINS"} and key not in os.environ:
            if len(value) > 1 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


load_environment()


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository: str = Field(validation_alias=AliasChoices("repository", "url"), max_length=250)

    @field_validator("repository")
    @classmethod
    def valid_repository(cls, value):
        return parse_repository(value)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repositories: list[str] = Field(validation_alias=AliasChoices("repositories", "urls"), min_length=2, max_length=2)

    @field_validator("repositories")
    @classmethod
    def valid_repositories(cls, values):
        return [parse_repository(value) for value in values]


class Prediction(BaseModel):
    probability: float = Field(ge=0, le=1)
    risk: Literal["Low", "Medium", "High"]
    model_version: str
    predicted_at: str
    explanation: str
    drivers: list[dict]


class AnalysisResponse(BaseModel):
    repository: dict
    prediction: Prediction | None
    prediction_error: str | None
    activity: dict
    insights: list[dict]
    features: dict[str, float | None]
    coverage: dict


class CompareResponse(BaseModel):
    repositories: list[AnalysisResponse] = Field(min_length=2, max_length=2)


app = FastAPI(title="RepoPulse", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(ApiError)
async def api_error(_request, error):
    headers = {"Retry-After": str(error.retry_after)} if error.retry_after else None
    return JSONResponse(error.payload(), status_code=error.status, headers=headers)


@app.get("/health")
@app.get("/api/health")
async def health():
    forest, _ = load_model()
    return {"status": "ok", "model_version": forest["model_version"], "runtime": "python", "github_token_configured": bool(os.getenv("GITHUB_TOKEN"))}


@app.get("/model-info")
@app.get("/api/model-info")
async def model_info():
    return load_model()[1]


async def body(request, schema):
    raw = await request.body()
    if len(raw) > 4096:
        raise ApiError("invalid_request", "Request body is too large.", 413)
    try:
        return schema.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ApiError("invalid_request", f"Invalid request: {first['msg']}.", 422) from exc


@app.post("/analyze", response_model=AnalysisResponse)
@app.post("/api/analyze", response_model=AnalysisResponse)
@app.post("/predict", response_model=AnalysisResponse)
@app.post("/api/predict", response_model=AnalysisResponse)
async def analyze_repository(request: Request):
    payload = await body(request, AnalyzeRequest)
    name = payload.repository
    async with httpx.AsyncClient() as client:
        return await analyze(name, GitHub(client, os.getenv("GITHUB_TOKEN")))


@app.post("/compare", response_model=CompareResponse)
@app.post("/api/compare", response_model=CompareResponse)
async def compare(request: Request):
    payload = await body(request, CompareRequest)
    names = payload.repositories
    async with httpx.AsyncClient() as client:
        github = GitHub(client, os.getenv("GITHUB_TOKEN"))
        return {"repositories": await asyncio.gather(*(analyze(name, github) for name in names))}
