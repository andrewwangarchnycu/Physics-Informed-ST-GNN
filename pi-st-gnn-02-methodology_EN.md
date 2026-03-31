# Building Morphology and Urban Open Space Optimization: A Physics-Informed Graph Neural Network Approach

## 2. METHODOLOGY

### 2.1 Framework Overview

The methodology of this study is built on an integrated workflow composed of data generation, graph-based representation, physics-guided spatio-temporal prediction, and design optimization. Based on the actual program implementation, the entire system can be divided into five connected modules: (1) parametric urban morphology sampling under site regulatory constraints; (2) construction of a high-resolution thermal environment dataset combining EPW forcing, simulation outputs, and observational calibration; (3) conversion of buildings and open-space sensing locations into a heterogeneous graph; (4) UTCI prediction using a Physics-Informed Spatio-Temporal Graph Neural Network (PIN-ST-GNN); and (5) embedding the trained model into a multi-objective search process to identify design solutions that jointly improve thermal comfort and open-space greening performance.

Compared with conventional grid-based models that directly feed regular matrices into convolutional networks, this study places greater emphasis on the coupling between morphological semantics and microclimatic temporal dynamics. In the codebase, building masses are represented as object nodes, while evaluation positions in open spaces are represented as air nodes; the environmental climate sequence is introduced separately as a global context vector. This design allows local shading, sky view factor, nearby building height, and hourly external weather forcing to act simultaneously within the same predictive framework. In other words, the proposed approach does not merely fit UTCI through deep learning, but instead constructs prediction on the basis of structured urban geometry, explicit climatic conditions, and physics-informed constraints.

### 2.2 Calibrated Synthetic Dataset Generation

According to `site_constraints.yaml` and the data generation scripts, the study first constructs parametric scenarios within an 80 m x 80 m site. The site location corresponds to the Hsinchu area, and the regulatory settings are defined as FAR = 2.5, BCR = 0.60, with a minimum setback of 3 m, building heights ranging from 3 to 12 floors, and a minimum building footprint of 80 m2. The geometry sampler (`02_geometry_sampler.py`) randomly generates 2 to 5 buildings in each scenario. Building plans may be either rectangular or L-shaped; L-shaped buildings are introduced with a 30% probability when the area threshold is satisfied in order to increase morphological diversity. Building height is derived by multiplying the number of floors by a floor-to-floor height of 3.6 m. At the same time, the pipeline records each building footprint, coverage, gross floor area (GFA), centroid, and the geometry of the remaining open space. To preserve the potential for green interventions in urban open space, the script also places 2 to 5 trees in the available open area, with tree height ranging from 4 to 12 m and crown radius estimated proportionally from height.

The climatic boundary condition is not specified as a single fixed value. Instead, an EPW parsing process extracts hourly forcing conditions from the EnergyPlus weather file. The program reads the EPW file and selects the hottest typical day in July, using the 11 hourly steps from 8:00 to 18:00 to define a consistent temporal window for simulation and model training. Beyond EPW forcing, the v2 pipeline also incorporates observational calibration using data from the Central Weather Administration (CWA/CWB) and the MOENV IoT sensor network. As shown in `04_sensing_calibration.py`, the calibration parameters include `roughness_length`, `albedo_road`, and `ta_bias_offset`, optimized by differential evolution or L-BFGS-B to minimize the proxy loss:

$$
\mathcal{L}_{calib}=2.0\cdot RMSE(T_a^{pred},T_a^{obs})+0.5\cdot RMSE(MRT^{proxy}),
$$

where proxy air temperature is derived from EPW temperature with roughness and bias correction, and proxy MRT is estimated using simplified shortwave, longwave, and ground-reflection terms. The purpose of this step is not to replace high-fidelity simulation, but to make the subsequent batch-generated thermal fields more consistent with real summer urban microclimatic conditions.

For dataset construction, the study adopts a two-stage strategy. In the v1 stage, the pipeline generates the original scenarios and simulation outputs. In the v2 stage, the data are upgraded to twice the spatial resolution and adjusted using real-weather calibration. According to `dataset_summary_v2.json` and `TRAINING_PROGRESS_v2.txt`, the final dataset contains 300 v2 scenarios. All scenarios are resampled to a 1.0 m grid, each containing a fixed 6,241 air nodes and 11 time steps. These scenarios are aggregated into `ground_truth_v2.h5` and split into 205 training, 41 validation, and 54 test cases. The corresponding full-dataset normalization statistics are: air temperature mean/std = 30.8255/1.1458, MRT = 55.1912/11.7281, wind speed = 4.1134/0.7666, relative humidity = 64.9091/2.1086, and UTCI = 35.6146/2.9841. These statistics are written directly into the HDF5 normalization groups and are reused consistently during data loading and deployment.

For clarity, the core dataset settings are summarized in Table 1.

| Item | Implementation in this study |
| --- | --- |
| Site size | 80 m x 80 m |
| Regulatory constraints | FAR 2.5, BCR 0.60, setback 3 m |
| Number of buildings | 2-5 per scenario |
| Number of trees | 2-5 per scenario |
| Temporal resolution | 8:00-18:00, 11 time steps |
| Spatial resolution | v2 uses 1.0 m grid |
| Number of nodes | 6,241 air nodes per scenario |
| Data split | train/val/test = 205/41/54 |

### 2.3 Heterogeneous Graph Representation of Urban Morphology and Open Space

At the graph construction stage, the urban thermal environment is reformulated as a heterogeneous graph rather than flattened into a single regular matrix. According to `dataset.py`, the graph contains at least two node types. The first is the object node, which represents buildings. Its feature dimension is 7, including normalized height, number of floors, footprint area, centroid coordinates, GFA, and an L-shape indicator. The second is the air node, which represents sensing locations and thermal comfort evaluation positions within open space. In the v2 setting, the air-node feature dimension is 9, in the following order: normalized air temperature, MRT, wind speed, relative humidity, SVF, time-varying shadow state, nearest building height, nearest tree height, and an additional surface temperature feature.

The importance of this representation lies in the fact that an air node is not just a geometric point. It simultaneously carries local microclimatic information, geometric obstruction, and neighborhood morphology. SVF is estimated from multi-directional obstruction relationships; the shadow state is computed from solar altitude and azimuth together with building shadow projection; and the nearest building and tree heights are used to reflect the regulating effect of the surrounding three-dimensional environment. Thus, when the model processes the thermal field in open space, it is effectively processing environment points constrained by urban morphology.

Edge definitions are divided into static relations and a reserved dynamic interface. In the current dataset implementation, the static edges include: (1) `semantic`, i.e., fully connected object-to-object edges, which model the semantic and geometric relationships among different building masses; and (2) `contiguity`, i.e., KNN-based air-to-air edges, with default `k=8`, which characterize local continuity in the open-space thermal field. The program architecture also preserves a `dynamic_edges` interface, allowing future expansion toward explicitly time-varying relations such as shading, convection, or vegetation evapotranspiration. However, in the present training data pipeline, `dynamic_edges` is an empty list of dictionaries. Therefore, the graph actually implemented in this study is primarily defined by building semantic relations and spatial adjacency among air nodes.

Formally, the site graph can be expressed as:

$$
G=(V_{obj}\cup V_{air}, E_{semantic}\cup E_{contiguity}),
$$

where $V_{obj}$ denotes the set of building nodes and $V_{air}$ denotes the set of open-space nodes; $E_{semantic}$ captures relationships among buildings, and $E_{contiguity}$ captures local propagation relationships among sensing locations. This representation enables urban morphology and open space to be treated not as separate descriptions, but as an organized relational system that can be directly aggregated by a graph neural network.

### 2.4 PIN-ST-GNN Architecture

The core model is implemented in `urbangraph.py`, and its structure can be summarized as Input MLP -> RGCN x 3 -> Global Context Fusion -> Temporal LSTM -> Output MLP. First, object nodes and air nodes are separately encoded by two input encoders and projected into a shared 128-dimensional latent space. Then, at each time step, the model concatenates the object embeddings with the air embeddings of that time step into a single node tensor and passes it through a three-layer RGCN for relational message passing. In `RGCNBlock`, each relation type is assigned its own linear transformation matrix, and aggregation is normalized by the destination node degree. Accordingly, a single-layer update can be written as:

$$
h_i^{(l+1)}=\sigma \left( W_{self}h_i^{(l)} + \sum_{r\in \mathcal{R}} \frac{1}{c_{i,r}} \sum_{j\in \mathcal{N}_i^r} W_r h_j^{(l)} \right),
$$

where $\mathcal{R}$ is the set of relations and $c_{i,r}$ is the number of neighbors of node $i$ under relation $r$. The implementation further includes residual connections, LayerNorm, PReLU, and dropout to stabilize deep relational aggregation.

After spatial encoding, the model does not directly regress UTCI step by step. Instead, the spatial representation of each air node is fused with global climatic context. The function `build_env_time_seq()` extracts from the hottest typical July day a 7-dimensional environmental vector `[Ta, RH, WS, WDsin, WDcos, GHI, SolAlt]` and a 2-dimensional time encoding `[sin(hour), cos(hour)]`. These are projected by `GlobalContextMLP` and concatenated with the spatial hidden states, then transformed by a fusion MLP into the LSTM input representation.

The temporal module uses a single-layer LSTM with hidden size 256. Importantly, the program does not initialize the LSTM with zero vectors. Instead, it uses the RGCN-processed air-node representation at the first time step, followed by `h0_proj` and warm-up linear layers, to generate the initial hidden state and cell state. This allows the temporal model to start from a spatially informed state rather than from an empty initialization. Finally, the model takes only the last hidden state of the LSTM and uses a two-layer MLP to decode all 11 UTCI values at once, producing an output tensor of shape $(N_{air}, T)$. This means the model learns the daytime UTCI trajectory for each node, rather than performing independent hour-by-hour regression.

### 2.5 Physics-Informed Objective Function

The training objective in this study consists of data loss, physics-based penalties, and optional sensor supervision. According to `compute_loss()`, the total loss is defined as:

$$
\mathcal{L}_{total}=\mathcal{L}_{data}+\mathcal{L}_{physics}+\lambda_{sense}\mathcal{L}_{sensor},
$$

where `L_data` is computed as MSE between predicted UTCI and normalized HDF5 ground truth; `L_sensor` is an optional term that applies masked supervision when sensor-based UTCI is available for specific nodes and time steps. The main feature of the training objective lies in `L_physics`, which is composed of three terms:

$$
\mathcal{L}_{physics}=L_{rad}+L_{temp}+L_{wind}.
$$

First, `radiation_penalty` enforces that, under sunlit conditions, nodes with higher SVF and no shading should on average exhibit higher UTCI than shaded nodes. If the model predicts shaded regions to be hotter than sun-exposed regions beyond a margin, a quadratic penalty is applied. Second, `temporal_smoothness_penalty` limits the change in UTCI between adjacent time steps. The program uses a default threshold `max_delta = 0.625` in normalized space, corresponding in the code comments to an approximate hourly limit of 5°C, thereby preventing unrealistic temporal oscillations. Third, `wind_obstruction_penalty` imposes an upper-bound constraint on nodes near tall buildings; when the UTCI of these nodes exceeds the global mean plus 1.5 standard deviations, an additional penalty is added, reflecting the expectation that the thermal load around leeward high-rise conditions should not diverge without bound.

It should be noted that these physics penalties are not direct discretizations of full governing equations. Rather, they translate fundamental physical knowledge of urban outdoor thermal environments into differentiable soft constraints. Their role is to shrink the feasible space of a purely data-driven model so that predictions remain statistically close to the data while also behaving consistently with basic principles of shading, thermal inertia, and wind obstruction. This setting is especially suitable for the surrogate modeling task addressed here, where prediction speed must be maintained without allowing physically implausible outputs to distort design decisions.

### 2.6 Deployment-Oriented Inference and Urban Open Space Optimization

The proposed methodology does not stop at offline prediction. It is further integrated into a deployable design workflow. `geometry_converter.py` transforms Rhino/Grasshopper inputs, including site boundary, building footprints, tree layouts, and sensor resolution, into `sensor_pts`, `obj_feat`, `air_feat`, and `static_edges` that are fully consistent with the training pipeline. This means the surrogate model developed in the study can directly accept design proposals as input and rapidly return UTCI spatial distributions and summary statistics over the site.

More importantly, the trained PIN-ST-GNN is embedded into the NSGA-II optimization framework in `07_optimization`. The chromosome is represented as continuous variables in `[0,1]^n`. Each building is encoded by 6 genes corresponding to centroid location, width, depth, rotation, and number of floors; each tree is encoded by 4 genes corresponding to position, radius, and height. After decoding, each design is first checked by `ConstraintChecker` against five classes of constraints: FAR, BCR, setback, site containment, and building overlap. Then `FitnessEvaluator` calls the GNN to estimate mean UTCI and computes canopy coverage as the green ratio. The dual-objective problem can be written as:

$$
\min f_1=\overline{UTCI}, \qquad \max f_2=Green\ Ratio.
$$

In the implementation, the first objective is minimized directly, while the second is converted to minimization by taking its negative value. NSGA-II uses feasibility-first non-dominated sorting, crowding distance, SBX crossover, and polynomial mutation to retain Pareto solutions that balance thermal comfort and open-space greening potential across generations. In this sense, the methodology ultimately forms a closed loop of morphology generation -> microclimate surrogate prediction -> constrained multi-objective optimization, enabling Building Morphology and Urban Open Space design to move beyond one-off simulation comparison toward systematic search supported by a physics-informed surrogate model.
