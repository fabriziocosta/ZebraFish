"""Local read-only FastAPI service for the mission-control dashboard."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from io import StringIO
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard_data import build_investigation, find_entity, find_observation, resolve_campaign
from src.agent_campaign_loop import load_campaign_config, campaign_live_status, terminate_campaign
from src.meta_controller import _pid_running, stop_loop


ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="ZebraFish Mission Control", version="1.0", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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


def _campaign_config(campaign: str) -> tuple[str, dict[str, Any]]:
    alias, config_path = resolve_campaign(ROOT, campaign)
    return alias, load_campaign_config(config_path)


def _spawn(command: list[str]) -> dict[str, Any]:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"status": "started", "pid": process.pid, "command": command}


@app.post("/api/investigation/{campaign}/control/{controller}/{action}")
def control(campaign: str, controller: str, action: str) -> dict[str, Any]:
    if controller not in {"meta", "campaign"}:
        raise HTTPException(status_code=400, detail="controller must be meta or campaign")
    if action not in {"start", "stop", "continue"}:
        raise HTTPException(status_code=400, detail="action must be start, stop, or continue")
    try:
        alias, config = _campaign_config(campaign)
        campaign_id = str(config["campaign"]["id"])
        if controller == "meta":
            if action == "stop":
                stopped = stop_loop(ROOT, config, reason="dashboard stop requested")
                return {"status": "stopping" if stopped else "stopped", "controller": "meta", "campaign": campaign_id}
            state_path = ROOT / str(config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
            if state_path.exists():
                from src.scientific_state import load_state
                meta_state = load_state(state_path).get("controller_state", {}).get("meta_controller", {})
                if isinstance(meta_state, dict) and _pid_running(meta_state.get("pid")):
                    return {"status": "already_running", "controller": "meta", "campaign": campaign_id, "pid": meta_state.get("pid")}
            return _spawn([sys.executable, str(ROOT / "run_campaign.py"), "meta-controller", alias, "--start" if action == "start" else "--continue"])
        live = campaign_live_status(config)
        if action == "stop":
            output = StringIO()
            code = terminate_campaign(config, reason="dashboard stop requested", force_after=float(config.get("meta_controller", {}).get("stop_grace_seconds", 5)), stream=output)
            return {"status": "stopped" if code == 0 else "not_stopped", "controller": "campaign", "campaign": campaign_id, "detail": output.getvalue()}
        if live.get("running"):
            return {"status": "already_running", "controller": "campaign", "campaign": campaign_id, "pid": live.get("pid")}
        return _spawn([sys.executable, str(ROOT / "run_campaign.py"), alias])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
