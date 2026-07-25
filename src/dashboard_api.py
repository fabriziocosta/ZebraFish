"""Local read-only FastAPI service for the mission-control dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard_data import build_investigation, find_entity, find_observation


ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="ZebraFish Mission Control", version="1.0", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _build(campaign: str, **kwargs: Any) -> dict[str, Any]:
    try:
        return build_investigation(ROOT, campaign, **kwargs)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "zebrafish-mission-control", "root": str(ROOT)}


@app.get("/api/investigation/{campaign}")
def investigation(
    campaign: str,
    view: str = Query("current", pattern="^(current|history|full)$"),
    level: int = Query(3, ge=0, le=5),
    relation_depth: int = Query(1, ge=0, le=5),
    entity_type: str | None = None,
    relation_type: str | None = None,
    confidence_min: float | None = Query(None, ge=0.0, le=1.0),
    time_from: str | None = None,
    time_to: str | None = None,
    active_only: bool = False,
) -> dict[str, Any]:
    # The adapter accepts the filters now and applies the graph filters. The
    # other filters remain part of the stable API contract for future views.
    return _build(
        campaign,
        view=view,
        level=level,
        relation_depth=relation_depth,
        filters={
            "entity_type": entity_type,
            "relation_type": relation_type,
            "confidence_min": confidence_min,
            "time_from": time_from,
            "time_to": time_to,
            "active_only": active_only,
        },
    )


@app.get("/api/investigation/{campaign}/evidence/{observation_id}")
def evidence(campaign: str, observation_id: str) -> dict[str, Any]:
    result = find_observation(ROOT, campaign, observation_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Observation not found: {observation_id}")
    return result


@app.get("/api/investigation/{campaign}/entities/{entity_id}")
def entity(campaign: str, entity_id: str, collection: str | None = None) -> dict[str, Any]:
    result = find_entity(ROOT, campaign, entity_id, collection)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")
    return result


@app.get("/api/investigation/{campaign}/graph")
def graph(
    campaign: str,
    level: int = Query(3, ge=0, le=5),
    relation_depth: int = Query(1, ge=0, le=5),
    entity_type: str | None = None,
    relation_type: str | None = None,
) -> dict[str, Any]:
    payload = _build(
        campaign,
        view="current",
        level=level,
        relation_depth=relation_depth,
        filters={"entity_type": entity_type, "relation_type": relation_type},
    )
    return payload["graph"]


DIST = ROOT / "dashboard" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="dashboard-assets")


@app.get("/", include_in_schema=False)
def dashboard_root():
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {
            "service": "zebrafish-mission-control",
            "message": "Dashboard frontend is not built. Run `cd dashboard && npm install && npm run build`.",
            "api": "/api/docs",
        }
    )
