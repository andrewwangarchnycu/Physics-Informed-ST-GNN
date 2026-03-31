"""
06_deployment/app.py
════════════════════════════════════════════════════════════════
PIN-ST-GNN FastAPI + WebSocket [REMOVED_ZH:5]

[REMOVED_ZH:4]：
  cd urban-thermal-gnn/06_deployment
  conda activate PytorchGPU
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

WebSocket [REMOVED_ZH:2]：ws://localhost:8000/ws

[REMOVED_ZH:4]（JSON）：
  Client → Server:
    { "action": "predict",  "id": "uuid", "data": { ...geometry... } }
    { "action": "optimize", "id": "uuid", "data": { ...ga_config... } }
    { "action": "cancel",   "id": "uuid" }
    { "action": "ping" }

  Server → Client:
    { "action": "predict_result",    "id": "...", "data": {...} }
    { "action": "optimize_progress", "id": "...", "data": {...} }
    { "action": "optimize_result",   "id": "...", "data": {...} }
    { "action": "error",             "id": "...", "message": "..." }
    { "action": "pong" }
"""
from __future__ import annotations
import sys, json, asyncio, logging, time, uuid, pickle, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set

# ── [REMOVED_ZH:4] ──────────────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent
_ROOT   = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "02_graph_construction"))
sys.path.insert(0, str(_ROOT / "03_model"))
sys.path.insert(0, str(_ROOT / "04_training"))
sys.path.insert(0, str(_ROOT / "07_optimization"))
sys.path.insert(0, str(_HERE))

import numpy as np
import h5py
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("uvicorn")

# ── model paths (prefer V2 checkpoint if available) ─────────────────────────
_CKPT_V2     = _ROOT / "checkpoints_v2" / "best_model.pt"
_CKPT_V1     = _ROOT / "04_training" / "checkpoints" / "best_model.pt"
CKPT_PATH    = _CKPT_V2 if _CKPT_V2.exists() else _CKPT_V1
H5_PATH      = _ROOT / "01_data_generation/outputs/raw_simulations/ground_truth.h5"
EPW_PKL_PATH = _ROOT / "01_data_generation/outputs/raw_simulations/epw_data.pkl"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════════════
# Global singletons (initialised at startup, thread-safe read-only after)
# ════════════════════════════════════════════════════════════════
_model:      object = None
_norm_stats: dict   = {}
_epw_data:   object = None
_dim_air:    int    = 8         # auto-detected from checkpoint at startup
_executor            = ThreadPoolExecutor(max_workers=4)
_active_jobs: Dict[str, "NSGA2Optimizer"] = {}

# ════════════════════════════════════════════════════════════════
# FastAPI application
# ════════════════════════════════════════════════════════════════
app = FastAPI(title="PIN-ST-GNN Deployment API",
              description="Real-time UTCI prediction + GA optimization for Rhino 8 Grasshopper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global _model, _norm_stats, _epw_data

    log.info(f"[startup] device={DEVICE}")

    # ── load normalisation statistics from HDF5 ───────────────────────
    with h5py.File(H5_PATH, "r") as hf:
        _norm_stats = {}
        for field in hf["normalization"].keys():
            grp = hf[f"normalization/{field}"]
            _norm_stats[field] = {
                "mean": float(grp.attrs["mean"]),
                "std":  float(grp.attrs["std"]),
            }
    log.info(f"[startup] norm_stats loaded: {list(_norm_stats.keys())}")

    # ── load EPW climate data (register types for pickle deserialisation) ──
    import __main__
    from shared import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC
    __main__.EPWData       = _ED
    with open(EPW_PKL_PATH, "rb") as f:
        _epw_data = pickle.load(f)
    log.info("[startup] EPW loaded")

    # ── load GNN model ────────────────────────────────────────────────
    global _dim_air
    from urbangraph import build_model
    ckpt_label = "V2" if CKPT_PATH == _CKPT_V2 else "V1"
    log.info(f"[startup] loading {ckpt_label} checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    # Auto-detect dim_air from checkpoint weights:
    #   air_encoder.net.0.weight has shape (hidden_dim, dim_air)
    state = ckpt["model_state"]
    air_enc_key = "air_encoder.net.0.weight"
    if air_enc_key in state:
        _dim_air = state[air_enc_key].shape[1]
    else:
        _dim_air = 8  # safe default for V1 checkpoints
    log.info(f"[startup] auto-detected dim_air={_dim_air}")

    cfg  = {"model": {"out_timesteps": 11, "dim_air": _dim_air}}
    _model = build_model(cfg).to(DEVICE)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    log.info(f"[startup] model loaded  epoch={ckpt.get('epoch','?')}  "
             f"val_R²={ckpt.get('val_r2', '?')}")


@app.get("/health")
def health():
    return {
        "status":   "ok",
        "device":   DEVICE,
        "model":    _model is not None,
        "epw":      _epw_data is not None,
        "dim_air":  _dim_air,
    }


@app.get("/norm_stats")
def norm_stats():
    return _norm_stats


# ════════════════════════════════════════════════════════════════
# WebSocket endpoint
# ════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client = ws.client.host if ws.client else "unknown"
    log.info(f"[WS] connected: {client}")

    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"action": "error", "message": "invalid JSON"})
                continue

            action = msg.get("action", "")
            msg_id = msg.get("id", str(uuid.uuid4()))

            if action == "ping":
                await ws.send_json({"action": "pong"})

            elif action == "predict":
                await _handle_predict(ws, msg_id, msg.get("data", {}), loop)

            elif action == "optimize":
                asyncio.create_task(
                    _handle_optimize(ws, msg_id, msg.get("data", {}), loop))

            elif action == "cancel":
                _cancel_job(msg_id)
                await ws.send_json({"action": "cancelled", "id": msg_id})

            else:
                await ws.send_json({
                    "action":  "error",
                    "id":      msg_id,
                    "message": f"unknown action: {action}",
                })

    except WebSocketDisconnect:
        log.info(f"[WS] disconnected: {client}")
    except Exception as e:
        log.error(f"[WS] error: {e}\n{traceback.format_exc()}")
        try:
            await ws.send_json({"action": "error", "message": str(e)})
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
# Predict handler
# ════════════════════════════════════════════════════════════════

async def _handle_predict(ws: WebSocket, msg_id: str,
                           data: dict, loop) -> None:
    """
    Run UTCI prediction. Offloads GNN inference to ThreadPoolExecutor
    to avoid blocking the async event loop.
    """
    try:
        t0 = time.perf_counter()
        result = await loop.run_in_executor(
            _executor, _predict_sync, data)
        elapsed = (time.perf_counter() - t0) * 1000

        result["inference_ms"] = round(elapsed, 1)
        await ws.send_json({
            "action": "predict_result",
            "id":     msg_id,
            "data":   result,
        })

    except Exception as e:
        log.error(f"[predict] {e}\n{traceback.format_exc()}")
        await ws.send_json({
            "action":  "error",
            "id":      msg_id,
            "message": str(e),
        })


def _predict_sync(data: dict) -> dict:
    """Synchronous GNN inference — runs inside executor thread."""
    from geometry_converter import GNNInputBuilder
    from train import build_env_time_seq

    builder   = GNNInputBuilder(_norm_stats, _epw_data, dim_air=_dim_air)
    gnn_in    = builder.build(data)

    if gnn_in is None:
        raise ValueError("sensor grid empty — check site boundary / building positions")

    sensor_pts  = gnn_in["sensor_pts"]
    obj_feat    = torch.from_numpy(gnn_in["obj_feat"]).to(DEVICE)
    air_feat    = torch.from_numpy(gnn_in["air_feat"]).to(DEVICE)
    static_edges = {
        k: torch.from_numpy(v).to(DEVICE)
        for k, v in gnn_in["static_edges"].items()
    }
    dynamic_edges = [{}] * air_feat.shape[1]

    env_seq, time_seq = build_env_time_seq(_epw_data, list(range(8, 19)), month=7)
    env_seq   = env_seq.to(DEVICE)
    time_seq  = time_seq.to(DEVICE)

    with torch.no_grad():
        pred = _model(
            obj_feat      = obj_feat,
            air_feat      = air_feat,
            dynamic_edges = dynamic_edges,
            static_edges  = static_edges,
            env_seq       = env_seq,
            time_seq      = time_seq,
        )  # (N_air, T)

    mu  = _norm_stats["utci"]["mean"]
    std = _norm_stats["utci"]["std"]
    utci = (pred.cpu().numpy() * std + mu).astype(float)  # (N_air, T)

    # hourly summary statistics
    hours = list(range(8, 19))
    hourly = {
        str(h): {
            "mean": round(float(utci[:, i].mean()), 2),
            "max":  round(float(utci[:, i].max()),  2),
            "min":  round(float(utci[:, i].min()),  2),
        }
        for i, h in enumerate(hours)
    }

    # thermal stress classification
    utci_peak = utci.mean(axis=1)  # (N_air,) — time-averaged per node
    classes   = _utci_class(utci_peak)

    return {
        "sensor_pts":   sensor_pts.tolist(),       # [[x,y],...]
        "utci_all":     utci.tolist(),             # (N_air, T)
        "utci_mean":    utci_peak.tolist(),        # (N_air,) time-mean
        "utci_class":   classes.tolist(),          # (N_air,) 0-5
        "hourly_stats": hourly,
        "n_sensors":    int(sensor_pts.shape[0]),
        "summary": {
            "mean_utci": round(float(utci.mean()), 2),
            "max_utci":  round(float(utci.max()),  2),
            "min_utci":  round(float(utci.min()),  2),
        },
    }


# ════════════════════════════════════════════════════════════════
# Optimize handler
# ════════════════════════════════════════════════════════════════

async def _handle_optimize(ws: WebSocket, job_id: str,
                            data: dict, loop) -> None:
    """Run NSGA-II optimisation and stream progress over WebSocket."""
    try:
        from chromosome import ChromosomeConfig
        from constraints import ConstraintChecker
        from fitness import FitnessEvaluator
        from nsga2_engine import NSGA2Optimizer

        cfg_dict  = data.get("chromosome_config", {})
        site_pts  = data.get("site_boundary", [[0,0],[80,0],[80,80],[0,80]])
        setback   = float(data.get("setback",    3.0))
        far_max   = float(data.get("far_max",    3.0))
        bcr_max   = float(data.get("bcr_max",    0.6))
        pop_size  = int(data.get("pop_size",    40))
        n_gen     = int(data.get("n_gen",       50))
        floor_h   = float(data.get("floor_height", 4.5))

        cfg     = ChromosomeConfig.from_dict(cfg_dict) if cfg_dict \
                  else ChromosomeConfig(
                      site_bbox   = _site_bbox(site_pts),
                      floor_height = floor_h)
        checker = ConstraintChecker(site_pts, setback, far_max, bcr_max, floor_h)
        evaluator = FitnessEvaluator(
            model       = _model,
            norm_stats  = _norm_stats,
            epw_data    = _epw_data,
            site_pts    = site_pts,
            cfg         = cfg,
            checker     = checker,
            device      = DEVICE,
        )
        optimizer = NSGA2Optimizer(
            evaluator  = evaluator,
            cfg        = cfg,
            pop_size   = pop_size,
            n_gen      = n_gen,
        )
        _active_jobs[job_id] = optimizer

        await ws.send_json({
            "action": "optimize_started",
            "id":     job_id,
            "data":   {"pop_size": pop_size, "n_gen": n_gen},
        })

        # progress queue — GA thread pushes, event loop reads
        queue: asyncio.Queue = asyncio.Queue()

        ga_future = asyncio.ensure_future(
            optimizer.run_async(loop, queue, _executor))

        # stream progress to client
        while not ga_future.done():
            try:
                info = await asyncio.wait_for(queue.get(), timeout=1.0)
                await ws.send_json({
                    "action": "optimize_progress",
                    "id":     job_id,
                    "data":   info,
                })
            except asyncio.TimeoutError:
                pass

        result = await ga_future
        _active_jobs.pop(job_id, None)

        await ws.send_json({
            "action": "optimize_result",
            "id":     job_id,
            "data":   result,
        })

    except WebSocketDisconnect:
        _cancel_job(job_id)
    except Exception as e:
        log.error(f"[optimize] {e}\n{traceback.format_exc()}")
        _active_jobs.pop(job_id, None)
        try:
            await ws.send_json({
                "action":  "error",
                "id":      job_id,
                "message": str(e),
            })
        except Exception:
            pass


def _cancel_job(job_id: str):
    if job_id in _active_jobs:
        _active_jobs[job_id].cancel()


# ════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════

_UTCI_THRESHOLDS = [-40, 9, 26, 32, 38, 46, 200]

def _utci_class(utci: np.ndarray) -> np.ndarray:
    c = np.zeros_like(utci, dtype=int)
    for i, (lo, hi) in enumerate(zip(_UTCI_THRESHOLDS[:-1], _UTCI_THRESHOLDS[1:])):
        c[(utci >= lo) & (utci < hi)] = i
    return c


def _site_bbox(pts):
    arr = np.array(pts)
    return (float(arr[:,0].min()), float(arr[:,1].min()),
            float(arr[:,0].max()), float(arr[:,1].max()))


# ════════════════════════════════════════════════════════════════
# Direct run
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")
