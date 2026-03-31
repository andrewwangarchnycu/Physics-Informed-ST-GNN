# Urban Thermal GNN Deployment Guide

Complete guide for deploying the Physics-Informed Spatio-Temporal GNN (PIN-ST-GNN) for urban thermal comfort prediction in Rhino 8 / Grasshopper.

**Status**: ✅ Production Ready
- Backend: Fully functional FastAPI + WebSocket server
- Grasshopper Components: UTCIPredictor, UTCIOptimizer, MaterialZoneEditor (optional)

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [Grasshopper Components](#grasshopper-components)
5. [API Reference](#api-reference)
6. [Advanced Features](#advanced-features)
7. [Testing & Validation](#testing--validation)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### 1. Start the Python Backend Server

```bash
# Navigate to deployment directory
cd urban-thermal-gnn/06_deployment

# Activate conda environment (adjust name as needed)
conda activate urbangnn

# Start FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     [startup] device=cuda  (or 'cpu' if GPU not available)
INFO:     [startup] norm_stats loaded: ['ta', 'mrt', 'va', 'rh', 'utci']
INFO:     [startup] EPW loaded
INFO:     [startup] model loaded  epoch=42  val_R²=0.89
```

### 2. Verify Server Health

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "model": true,
  "epw": true
}
```

### 3. Open Grasshopper Definition

1. Launch **Rhino 8**
2. Open Grasshopper (`Shift+Ctrl+G`)
3. Create a new script component
4. Copy contents of `UTCIPredictor.ghpy` into the script editor
5. Configure input ports and connect geometry

---

## 🖥️ System Requirements

### Backend (Python)

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.9+ | 3.10/3.11 recommended |
| PyTorch | 2.0+ | GPU support optional but recommended |
| FastAPI | 0.100+ | Web framework |
| H5py | 3.0+ | HDF5 data format |
| NumPy | 1.24+ | Numerical computing |
| Shapely | 2.0+ | Geometry operations |

**GPU Acceleration** (optional):
- NVIDIA GPU with CUDA 11.8+ (GeForce RTX 3060+ or Tesla T4+)
- Inference: 0.5-2s per prediction (GPU) vs 5-10s (CPU)

### Client (Grasshopper/Rhino)

| Component | Version | Notes |
|-----------|---------|-------|
| Rhino | 8.0+ | Rhino 7.x **not** supported |
| Grasshopper | Built-in | Comes with Rhino 8 |
| Python for Rhino | 3.9+ | `pip install websocket-client` |

---

## 🔧 Installation & Setup

### 1. Backend Setup

```bash
# Clone or navigate to repository
cd urban-thermal-gnn

# Create conda environment (if not exists)
conda create -n urbangnn python=3.10 pip -y
conda activate urbangnn

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Grasshopper Python Dependencies

In **Rhino 8 Script Editor**:

```python
# Run this once to install websocket-client
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "websocket-client"])
print("✓ websocket-client installed successfully")
```

Or from **Command Prompt** (Windows):

```cmd
"%LOCALAPPDATA%\Programs\Rhino 8\System\netfx\rhinocode\python\python.exe" -m pip install websocket-client
```

### 3. Verify Installation

```bash
# Test backend imports
python -c "from urbangraph import build_model; print('✓ Model import OK')"

# Test Grasshopper Python
python -c "import websocket; print('✓ WebSocket import OK')"
```

---

## 🎯 Grasshopper Components

### Component 1: UTCIPredictor

**Purpose**: Single-shot UTCI thermal comfort prediction for a design

**File**: `06_deployment/gh_component/UTCIPredictor.ghpy`

#### Input Ports

| Port | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `buildings_breps` | Brep (list) | No | — | Design buildings (solids) to evaluate |
| `tree_circles` | Circle (list) | No | — | Tree canopy locations (top-down view) |
| `tree_heights` | Number (list) | No | 5.0m | Tree heights (one per circle) |
| `site_boundary` | Curve | **Yes** | — | Site perimeter (closed polyline) |
| `hour_idx` | Integer | No | 4 | Display hour (0–10, where 0=8:00 AM) |
| `sensor_res` | Number | No | 2.0m | Grid spacing for sensor points |
| `server` | String | No | localhost:8000 | WebSocket server address |
| `run` | Boolean | **Yes** | False | Trigger computation (set to True) |

#### Output Ports

| Port | Type | Description |
|------|------|-------------|
| `utci_mesh` | Mesh | Vertex-colored UTCI heatmap (blue→green→yellow→orange→red) |
| `sensor_pts` | Point3d (list) | Sensor grid locations (use for analysis) |
| `utci_vals` | Number (list) | UTCI values at each sensor [°C] |
| `stats` | String | Summary statistics (count, mean, min, max, inference time) |
| `status` | String | Connection/computation status |

#### Workflow

1. **Draw site boundary** (closed polyline) → Connect to `site_boundary`
2. **Create building blocks** (Breps/extrusions) → Connect to `buildings_breps`
3. **Add tree locations** (circles on XY plane) → Connect to `tree_circles`
4. **Assign tree heights** (matching circle count) → Connect to `tree_heights`
5. **Set run=True** → Component computes UTCI prediction
6. **Visualize** → Colored mesh appears; read mean UTCI from stats

#### Example Usage

```python
# Inside Grasshopper Python script component
site_crv = ...  # Your site boundary curve
buildings = [...]  # List of building Breps
trees = [...]  # List of tree Circle objects
heights = [3.0, 5.0, 7.0]  # Tree heights in meters

# Component outputs:
# utci_mesh → visualize in Rhino (shaded)
# stats → read thermal comfort conditions
# sensor_pts → export for further analysis
```

#### Color Scale

| UTCI Range | Color | Thermal Stress |
|-----------|-------|---|
| < 9°C | Blue | Extreme Cold |
| 9–26°C | Green | Comfort |
| 26–32°C | Yellow | Warm |
| 32–38°C | Orange | Hot |
| 38–46°C | Red | Very Hot |
| > 46°C | Dark Red | Extreme Heat |

---

### Component 2: UTCIOptimizer

**Purpose**: Genetic algorithm optimization of building/tree layouts to minimize thermal discomfort

**File**: `06_deployment/gh_component/UTCIOptimizer.ghpy`

#### Input Ports

| Port | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `site_boundary` | Curve | **Yes** | — | Site boundary |
| `setback` | Number | No | 3.0m | Minimum setback from site edge |
| `far_max` | Number | No | 3.0 | Max floor-area ratio constraint |
| `bcr_max` | Number | No | 0.6 | Max building coverage ratio |
| `n_buildings` | Integer | No | 2 | Number of buildings to optimize |
| `n_trees` | Integer | No | 4 | Number of trees to optimize |
| `floor_range` | String | No | "2,8" | Building floor count range (e.g., "3,10") |
| `n_gen` | Integer | No | 50 | GA generations |
| `pop_size` | Integer | No | 40 | Population size per generation |
| `pareto_idx` | Integer | No | 0 | Index into Pareto front to display |
| `server` | String | No | localhost:8000 | WebSocket server address |
| `run` | Boolean | **Yes** | False | **Start** optimization (pulse: False→True) |
| `stop` | Boolean | No | False | **Cancel** running optimization |

#### Output Ports

| Port | Type | Description |
|------|------|-------------|
| `buildings_breps` | Brep (list) | Building geometry for selected Pareto solution |
| `trees_circles` | Circle (list) | Tree locations for selected solution |
| `utci_mesh` | Mesh | UTCI heatmap for selected solution |
| `objectives` | String | UTCI and green ratio for selection |
| `progress` | String | Real-time GA iteration progress |
| `pareto_count` | Integer | Number of Pareto-optimal solutions found |
| `status` | String | Optimization status (idle/running/complete/error) |

#### Workflow

1. **Define constraints**:
   - `setback=3.0`: Buildings must be 3m from site edge
   - `far_max=2.5`: Max buildable floor area ≤ 2.5 × site area
   - `bcr_max=0.5`: Buildings cover ≤ 50% of site

2. **Start optimization** (use Timer component with 1s interval):
   ```
   run → True → GA starts (100–200 evaluations)
                → False → polls progress
   ```

3. **Monitor progress**:
   - `progress` shows generation count, best UTCI, Pareto count
   - Runs in background (doesn't freeze Grasshopper)

4. **Browse solutions**:
   - Connect `pareto_count` to Integer Slider → `pareto_idx`
   - Slider selects which Pareto solution to display
   - Geometry updates for each selection

5. **Export best solution**:
   - Connected `buildings_breps` to Bake component
   - Right-click → "Bake" to save to Rhino

#### Optimization Details

- **Algorithm**: NSGA-II (Non-dominated Sorting GA)
- **Objectives**:
  - Minimize: Mean UTCI across all sensors, all hours
  - Maximize: Green space ratio (trees + vegetation)
- **Constraints**:
  - No building overlap
  - Buildings within site + setback
  - FAR/BCR limits
  - Floor count within specified range
- **Genes per building**: 6 (cx, cy, width, depth, rotation, floors)
- **Genes per tree**: 4 (x, y, radius, height)
- **Typical runtime**: 50 gen × 40 pop = 2000 evaluations ≈ 20–30 minutes

---

### Component 3: MaterialZoneEditor (Optional)

**Purpose**: Define surface material zones (grass, concrete, asphalt, water, etc.) for accurate surface temperature computation

**File**: `06_deployment/gh_component/MaterialZoneEditor.ghpy`

#### Input Ports

| Port | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `zone_boundaries` | Curve (list) | **Yes** | — | Boundary curves for each material zone |
| `material_names` | String (list) | **Yes** | — | Material type per zone (see below) |
| `site_boundary` | Curve | No | — | Site boundary (for context) |
| `sensor_res` | Number | No | 2.0 | Sensor grid resolution |

#### Material Types

Available options:
- `"grass"` - Vegetation (high ET, cooling effect)
- `"concrete"` - Concrete (moderate albedo)
- `"asphalt"` - Asphalt (low albedo, hot)
- `"permeable_pavement"` - Permeable surface (some infiltration)
- `"water"` - Water body (maximum evapotranspiration)
- `"bare_soil"` - Soil (moderate properties)

#### Output Ports

| Port | Type | Description |
|------|------|-------------|
| `material_zones` | Dictionary | JSON dict of zones (for backend) |
| `preview_mesh` | Mesh | Colored visualization of zones |
| `zone_info` | String | Summary of defined zones |
| `status` | String | Validation status |

#### Usage Example

```python
# 1. Draw grass zone (large rectangle)
grass_boundary = Rectangle(0, 0, 50, 50)

# 2. Draw asphalt zone (parking area)
asphalt_boundary = Rectangle(50, 0, 60, 50)

# 3. Connect to component:
zone_boundaries = [grass_boundary, asphalt_boundary]
material_names = ["grass", "asphalt"]

# 4. Preview shows colored zones (green + dark gray)
# 5. Pass material_zones to backend for surface-aware UTCI
```

---

## 📡 API Reference

### WebSocket Messages

#### Request: Predict (Single Shot)

```json
{
  "action": "predict",
  "id": "unique_request_id",
  "data": {
    "site_boundary": [[x1,y1], [x2,y2], ...],
    "buildings": [
      {
        "footprint": [[x,y], ...],
        "height": 18.0,
        "floor_count": 4
      }
    ],
    "trees": [
      {
        "x": 10.0,
        "y": 20.0,
        "radius": 3.0,
        "height": 8.0
      }
    ],
    "sensor_resolution": 2.0
  }
}
```

#### Response: Predict Result

```json
{
  "action": "predict_result",
  "id": "unique_request_id",
  "data": {
    "sensor_pts": [[x1,y1], [x2,y2], ...],
    "utci_all": [[t0_h8, t0_h9, ..., t0_h18], ...],
    "utci_mean": [22.5, 24.3, ...],
    "utci_class": [0, 1, 1, 2, ...],
    "n_sensors": 125,
    "summary": {
      "mean_utci": 25.4,
      "max_utci": 38.2,
      "min_utci": 18.9
    },
    "inference_ms": 1250.3
  }
}
```

#### Request: Optimize (GA)

```json
{
  "action": "optimize",
  "id": "unique_job_id",
  "data": {
    "site_boundary": [[...], ...],
    "setback": 3.0,
    "far_max": 3.0,
    "bcr_max": 0.6,
    "pop_size": 40,
    "n_gen": 50,
    "chromosome_config": {
      "site_bbox": [x_min, y_min, x_max, y_max],
      "n_buildings": 2,
      "n_trees": 4,
      "floor_range": [3, 12],
      "bldg_w_range": [8.0, 30.0],
      "bldg_d_range": [8.0, 30.0],
      "tree_r_range": [1.5, 5.0],
      "tree_h_range": [3.0, 10.0]
    }
  }
}
```

#### Response: Optimize Progress

```json
{
  "action": "optimize_progress",
  "id": "unique_job_id",
  "data": {
    "generation": 15,
    "n_gen": 50,
    "n_feasible": 32,
    "best_utci": 24.8,
    "best_green": 0.25,
    "pareto_count": 18,
    "pareto_designs": [
      {
        "design": {
          "buildings": [...],
          "trees": [...]
        },
        "mean_utci": 24.8,
        "green_ratio": 0.25
      },
      ...
    ]
  }
}
```

#### Response: Optimize Result

```json
{
  "action": "optimize_result",
  "id": "unique_job_id",
  "data": {
    "status": "complete",
    "generations_run": 50,
    "pareto_designs": [
      {
        "design": {...},
        "mean_utci": 24.1,
        "green_ratio": 0.30,
        "far": 1.8,
        "bcr": 0.45
      },
      ...
    ]
  }
}
```

---

## 🔬 Advanced Features

### 1. Surface Material Support

The updated `geometry_converter.py` (line 430) now supports surface temperature:

```python
# Air features dimension: 9 (was 8)
# [Ta_norm, MRT_norm, Va_norm, RH_norm, SVF, shadow, Bh_norm, Th_norm, Ts_norm]
```

**Material properties** defined in `shared/surface_materials.py`:
- Albedo (solar reflectance)
- Emissivity (thermal radiation)
- Evapotranspiration coefficient (cooling effect)
- Thermal conductivity

**Default behavior**: All sensors assume concrete surface (conservative estimate).

**To use material zones**:
1. Use `MaterialZoneEditor.ghpy` to define zones
2. (Future) Pass `material_zones` to `UTCIPredictor` input
3. Backend computes surface temperature per zone

### 2. Custom Environment Data

Modify `shared/climate.py` to:
- Change simulation hours: `SIM_HOURS = list(range(7, 19))` → 7:00–18:00
- Use specific EPW file (weather data)
- Override climate data for sensitivity analysis

### 3. Physics-Constrained Predictions

The model includes **physics penalties** (radiation, temporal smoothness, wind) that enforce realistic UTCI evolution.

Adjust in `04_training/train.py`:
```python
lambdas = {
    "lambda1": 0.1,  # Data loss weight
    "lambda2": 0.05,  # Radiation penalty
    "lambda3": 0.05,  # Temporal smoothness
}
```

### 4. Batch Processing

For batch evaluation of multiple designs (via Python):

```python
from geometry_converter import GNNInputBuilder

builder = GNNInputBuilder(norm_stats, epw_data)

for design_payload in design_list:
    gnn_input = builder.build(design_payload)
    pred = model(gnn_input)  # (N, T) predictions
    # Process results...
```

---

## ✅ Testing & Validation

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected:
- `"status": "ok"`
- `"model": true`
- `"epw": true`

### 2. Simple Prediction Test

```python
import json
import websocket

ws = websocket.create_connection("ws://localhost:8000/ws")

payload = {
    "action": "predict",
    "id": "test_1",
    "data": {
        "site_boundary": [[0,0], [100,0], [100,100], [0,100]],
        "buildings": [
            {
                "footprint": [[20,20], [80,20], [80,80], [20,80]],
                "height": 20.0,
                "floor_count": 5
            }
        ],
        "trees": [],
        "sensor_resolution": 5.0
    }
}

ws.send(json.dumps(payload))
response = json.loads(ws.recv())

print(f"✓ Received {response['data']['n_sensors']} sensor points")
print(f"✓ Mean UTCI: {response['data']['summary']['mean_utci']:.1f}°C")

ws.close()
```

### 3. Grasshopper Component Test

1. **Setup**:
   - Create new GH definition
   - Copy `UTCIPredictor.ghpy` to script component
   - Draw test geometry

2. **Validate**:
   - Set `run=True` → Component should execute in <5 seconds
   - Check `status` output → Should show "OK (XXX ms)"
   - Verify `utci_mesh` appears → Colored visualization
   - Check `stats` → Mean UTCI 20–35°C (summer)

3. **Troubleshooting**:
   - If error: "websocket-client not installed" → Run pip install
   - If timeout: Check server is running and reachable
   - If wrong UTCI values: Verify site boundary area >100 m²

### 4. Unit Tests

Run model validation:

```bash
cd 04_training
python evaluate.py checkpoints/best_model.pt

# Expected output:
# ✓ Test R² = 0.89
# ✓ Mean MAE = 1.2°C
# ✓ Percent within 2°C = 78%
```

---

## 🛠️ Troubleshooting

### Server Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `Port 8000 already in use` | Another process on 8000 | `lsof -i :8000` or change port: `--port 8001` |
| `ModuleNotFoundError: no module named 'urbangraph'` | PYTHONPATH not set | Ensure `sys.path` includes root directory |
| `CUDA out of memory` | Batch too large or insufficient VRAM | Use `--device cpu` or reduce grid resolution |
| `norm_stats not loaded` | HDF5 file missing | Check `H5_PATH` in `app.py` points to valid file |

### Grasshopper Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `websocket-client not installed` | Missing Python package | Run `pip install websocket-client` in Rhino PE |
| `Connection refused: [Errno 10061]` | Server not running | Start backend: `uvicorn app:app --port 8000` |
| `Timeout: no response in 10s` | Slow server or network | Increase timeout in script or check server logs |
| `Site boundary too few points` | Curve has <3 points | Ensure polyline has at least 3 vertices |
| `Sensor grid empty` | Buildings completely cover site | Move/resize buildings to leave space |

### Model Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `UTCI values unrealistic (>50°C or <-20°C)` | Unusual geometry | Validate site/building dimensions realistic |
| `Predictions don't change with design` | Model frozen/weights wrong | Reload checkpoint: `torch.load(..., weights_only=False)` |
| `Inconsistent predictions` | Random initialization in GNN | Model inference is deterministic; results should repeat |

---

## 📚 References

### Key Publications

1. Karsisto et al. (2016). "Seasonal surface urban heat island characteristics in a Northern European city"
2. Fiala et al. (2012). "UTCI—An Universal Thermal Climate Index"
3. Bruse & Fleer (1998). "Simulating surface-plant-air interactions inside urban environments with a three-dimensional numerical model"

### Model Architecture

- **Graph**: Heterogeneous GN with object (buildings/trees) + air (sensor) nodes
- **Spatial**: RGCN convolution (3 layers), 128-dim hidden state
- **Temporal**: LSTM encoder (1 layer, 256 hidden), 11-step output
- **Physics**: Custom loss with radiation, smoothness, wind constraints
- **Training**: Adam optimizer, 200 epochs, ~90% validation R²

### File Structure

```
urban-thermal-gnn/
├── 01_data_generation/          # Data pipeline
├── 02_graph_construction/       # Graph building
├── 03_model/                    # GNN architecture
│   ├── urbangraph.py           # Main model
│   ├── loss/
│   │   ├── data_loss.py
│   │   └── physics_penalty.py
├── 04_training/                 # Training scripts
├── 06_deployment/               # Production deployment
│   ├── app.py                  # FastAPI server
│   ├── geometry_converter.py    # Geometry to GNN input
│   ├── gh_component/           # Grasshopper plugins
│   │   ├── UTCIPredictor.ghpy
│   │   ├── UTCIOptimizer.ghpy
│   │   └── MaterialZoneEditor.ghpy
│   └── visualization/          # Color mapping
├── 07_optimization/             # GA optimizer
└── shared/                      # Common utilities
    ├── climate.py
    └── surface_materials.py     # NEW: Material properties
```

---

## 📞 Support & Feedback

For issues or questions:
1. Check this guide's **Troubleshooting** section
2. Review **API Reference** for message format
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Share error logs and minimal test case

---

**Last Updated**: 2024-03-31  
**Version**: 2.0 (Material-Aware, 9D Air Features)  
**Status**: ✅ Production Ready
