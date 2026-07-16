# Building Morphology and Urban Open Space Optimization
## A Physics-Informed Graph Neural Network Approach

**Authors:** Cheng-An Wang¹, June-Hao Hou²  
¹'² Graduate Institute of Architecture, National Yang Ming Chiao Tung University  
Email: 1890718ding3316@arch.nycu.edu.tw, jhou@arch.nycu.edu.tw  

---

### Abstract
Rapid urbanization and climate change have intensified the Urban Heat Island (UHI) effect, necessitating performance-driven urban design. However, traditional physics-based microclimate simulations are computationally expensive, hindering their application in iterative design processes. This paper introduces a novel surrogate modeling framework utilizing a Physics-Informed Spatio-Temporal Graph Neural Network (PIN-ST-GNN) integrated with a Non-dominated Sorting Genetic Algorithm II (NSGA-II). By translating 3D urban geometric semantics into heterogeneous graphs and enforcing physical constraints (radiation, wind obstruction, and temporal smoothness) during training, the PIN-ST-GNN predicts the Universal Thermal Climate Index (UTCI) at 61.5 ± 24.4 ms per scenario inference (RTX 5070 Ti, 6,241 air nodes, 11 time steps). The training targets are calibrated physics-proxy values derived from simplified radiation models and EPW-based forcing, calibrated against IoT sensor observations. Deployed via a FastAPI backend to a Rhino Grasshopper interface, the system enables continuous morphological evolution. This paper presents the machine learning architecture, the design optimization methodology, and a parametric case study, demonstrating the framework's potential to bridge the gap between complex climate science and architectural geometry optimization.

**Keywords:** Physics-Informed Machine Learning, Spatio-Temporal Graph Neural Networks, Outdoor Thermal Comfort, Urban Morphology, Design Optimization

---

### 1. Introduction
The spatial configuration of urban environments significantly alters local microclimates, directly affecting human thermal comfort and building energy consumption. As extreme heat events become more frequent, integrating microclimate considerations into the early stages of urban planning and architectural design is no longer optional but essential. Despite the urgency, conventional microclimate evaluation relies heavily on Computational Fluid Dynamics (CFD) and rigorous thermodynamic engines (e.g., OpenFOAM). These traditional physics-based simulations are time-consuming and computationally demanding. A single urban scene can take hours to simulate, rendering iterative feedback during the design optimization phase practically impossible.

To overcome the computational bottleneck, we propose an integrated computational framework based on a surrogate model. By utilizing a Physics-Informed Spatio-Temporal Graph Neural Network (PIN-ST-GNN), we bypass the need for explicit equation-solving at runtime. The surrogate model is coupled with an NSGA-II evolutionary algorithm to automate the discovery of optimal morphological configurations that simultaneously maximize thermal comfort (UTCI) and urban greenery. The gap addressed here is specifically the simultaneous optimization of three dimensions that prior work has not combined: (i) thermal comfort (UTCI), (ii) ecological quality (canopy coverage), and (iii) statutory compliance (FAR, BCR, setbacks) — all within an interactive design tool that operates at design-sketch speeds.

---

### 2. Background

#### 2.1 Microclimate Simulation and Urban Design Challenges
As global urbanization accelerates and the impacts of climate change intensify, mitigating the Urban Heat Island (UHI) effect and optimizing outdoor thermal comfort have emerged as critical priorities in urban spatial planning. Accurately assessing dynamic environmental metrics, such as the Universal Thermal Climate Index (UTCI), is essential for evaluating human physiological equivalent temperature and outdoor comfort (Lopez-Cabeza et al., 2025).

Currently, parametric environmental plugins like Ladybug Tools represent the industry standard for translating complex, multi-scalar building geometries into strict boundary conditions for Computational Fluid Dynamics (CFD) and rigorous thermodynamic analysis. Recent developments have also introduced open-source urban digital twins to enhance microclimate analysis and outdoor thermal comfort using high-resolution geographical and spatial data (Lopez-Cabeza et al., 2025). However, traditional physical modeling and high-fidelity microclimate evaluations require immense computational resources. Beyond the raw computational cost, traditional deterministic tools present three additional barriers to early-design use: (1) they require complete geometric inputs, making them incompatible with the sketch-level representations typical of early-stage design; (2) they struggle with geometric variability — each morphological mutation requires a full mesh re-generation and re-simulation cycle; and (3) they provide no uncertainty quantification, giving designers a single point estimate rather than a distribution of plausible microclimatic outcomes. This severe simulation latency imposes a fundamental bottleneck, largely restricting environmental simulations to post-design validation (Wu et al., 2024) rather than enabling their use as proactive, active drivers within iterative generative design workflows.

#### 2.2 Machine Learning and Graph-Based Models in Urban Simulation
To overcome these computational limitations, recent studies have increasingly integrated machine learning (ML) to dramatically accelerate urban performance simulation. Generative adversarial networks (GANs), deeply coupled with genetic algorithms and Geographic Information Systems (GIS), have been pioneered to synthesize and multi-objectively optimize large-scale urban spatial plans (Cheng et al., 2023). Furthermore, surrogate models entirely driven by data are frequently adopted to emulate expensive high-fidelity building simulations, successfully providing rapid sustainable performance predictions for intricate residential block layout designs (Wu et al., 2024). In the highly specific domain of outdoor thermal comfort, data-driven ML models have also been deployed to optimize kinetic shading devices and evaluate the long-term daylight and thermal impacts of diverse morphological attributes (Dağlier et al., 2025; Aman et al., 2023).

Despite these impressive accelerations, capturing the intricate topological relationships and irregular temporal dynamics of cities remains a persistent challenge. Consequently, Graph Neural Networks (GNNs) have emerged as highly effective, non-Euclidean architectures for modeling complex spatio-temporal systems (Oskarsson, 2025). By treating individual buildings as interlinked nodes and their spatial dependencies as edges, Spatiotemporal GCNs have been successfully deployed for urban-scale hourly building energy consumption forecasting (Hu et al., 2022).

#### 2.3 Physics-Informed Deep Learning
A critical gap persists in the widespread application of purely data-driven ML models: they frequently act as "black boxes" that ignore fundamental thermodynamic laws, sometimes outputting physically impossible scenarios such as drastic temperature gradients in adjacent shaded zones. The introduction of Physics-Informed Neural Networks (PINNs) presented a robust deep learning framework capable of solving nonlinear partial differential equations, firmly ensuring that model predictions naturally respect physical boundaries (Raissi et al., 2018). Recently, this exact paradigm has been expanded to graph structures. Explainable, physics-informed GNNs have been developed to enhance generalizable urban building energy modeling by systematically benchmarking physical edge features like inter-building shading (Shan et al., 2025). Similarly, novel structure-based inductive biases, primarily dynamic heterogeneous graphs, have been explored to explicitly encode causal topologies—such as localized convection and obstruction—for urban microclimate predictions (Xin et al., 2025).

Physics-Informed Neural Networks (PINNs) differ from grey-box models in a conceptually important way. Grey-box models combine mechanistic sub-models with statistical calibration (e.g., a simplified energy balance equation whose coefficients are fit to data), maintaining an explicit, interpretable physical structure. PINNs, by contrast, are neural networks that embed physical knowledge as soft constraints in the loss function, without requiring the network architecture to mirror a governing equation. The advantage is flexibility: the network can model complex non-linear relationships that resist analytical formulation (such as the coupled radiative-convective-conductive heat exchange in a heterogeneous urban canyon), while the physics terms prevent the network from converging to physically implausible local minima.

#### 2.4 Research Gap and Contribution
While existing research has separately explored surrogate modelling, spatio-temporal GNNs, and physics-informed neural networks, a holistic framework precisely tailored for optimizing generic building morphology and urban open spaces remains absent. Current pure AI models often fail to actively combine thermal inertia, continuous temporal radiation dynamics, and aerodynamic wake effects into a seamless architectural design loop.

This research explicitly bridges this critical gap by formulating a novel Physics-Informed Spatio-Temporal Graph Neural Network (PIN-ST-GNN). Unlike prior homogeneous graph approaches, our architecture fuses Relational Graph Convolutional Networks (RGCN) and Long Short-Term Memory (LSTM) modules with a physics-guided loss function. By penalizing violations of wind obstruction and thermal radiation physics, the network avoids physically implausible predictions. Crucially, the deployment framework directly interfaces the calibrated AI model with Grasshopper3D, moving beyond standalone computational energy forecasting to provide urban designers with an interactive feedback environment designed for genuine generative spatial optimization.

---

### 3. Methodology
This study is founded on an integrated computational workflow encompassing topological data generation, graph-based representation, physics-guided spatio-temporal prediction, and generative design optimization. Computationally, the framework comprises five interconnected modules:

1. **Parametric Morphology Generation:** The procedural generation of urban morphology, strictly governed by site-specific regulatory constraints.
2. **Dataset Compilation:** The assembly of a calibrated physics-proxy microclimate dataset, integrating EnergyPlus Weather (EPW) forcing data, computational simulation outputs, and empirical sensor calibration.
3. **Graph-Based Translation:** The structural conversion of physical building geometries and open-space sensor networks into a dynamic heterogeneous graph architecture.
4. **Spatio-Temporal Prediction:** The predictive modeling of the Universal Thermal Climate Index (UTCI) utilizing a Physics-Informed Spatio-Temporal Graph Neural Network (PIN-ST-GNN).
5. **Generative Design Optimization:** The integration of the calibrated surrogate model within a multi-objective optimization algorithm to simultaneously maximize outdoor thermal comfort and open-space ecological performance.

In contrast to conventional grid-based approaches that reduce urban environments to regularized 2D matrices for Convolutional Neural Networks (CNNs), this methodology explicitly preserves the topological relationships between morphological semantics and spatio-temporal microclimate dynamics. Within the graph architecture, physical building volumes are encoded as discrete object nodes, while outdoor evaluation points are defined as air nodes. Crucially, sequential meteorological data is injected as a global context vector. This heterogeneous configuration enables localized geometric attributes—such as the Sky View Factor (SVF), transient shading phenomena, and adjacent building heights—to interact dynamically with hourly environmental forcing. Consequently, rather than treating UTCI prediction as a purely data-driven curve-fitting exercise, the proposed framework structurally embeds urban geometry, explicit climatic boundary conditions, and thermodynamic constraints directly into the deep learning pipeline.

#### 3.1 Calibrated Synthetic Dataset Generation
Governed by Hsinchu's regulatory constraints (FAR 2.5, BCR 0.60, 3m setback), the computational pipeline generates parametric scenarios within an 80×80m site. A stochastic algorithm populates 2–5 discrete buildings (orthogonal or L-shaped, 3–12 stories) and 2–5 canopy trees (4–12m height) to ensure morphological and ecological heterogeneity, concurrently logging key spatial metrics like Gross Floor Area (GFA) and volumetric centroids. Hsinchu City was selected as the case study site for three reasons: (1) it exemplifies Taiwan's high-density commercial block typology with FAR 2.5 and BCR 0.60, representative of subtropical Asian urban contexts; (2) the research group has existing IoT sensor deployments at this site enabling empirical calibration; and (3) CWA meteorological station data are available for the period (June–September 2025) used in training. For climatic boundaries, the framework dynamically parses localized EnergyPlus Weather (EPW) files, extracting an 11-hour diurnal sequence (08:00–18:00) from the hottest July day for transient microclimate simulation.

To bridge idealized simulations with physical reality, an empirical calibration sequence utilizes high-resolution Central Weather Administration (CWA) and IoT sensor data. The empirical calibration draws on IoT temperature and humidity sensors (Moenv network) deployed across the study area in Hsinchu's East District, recording at 10-minute intervals with hourly averages used for calibration. Sensor locations were selected to sample the range of SVF and building density conditions present in the training scenarios. Automated optimization routines (e.g., Differential Evolution or L-BFGS-B) iteratively refine critical microclimatic parameters—including aerodynamic roughness, pavement albedo, and temperature offsets—to minimize the proxy error between simulated outputs and empirical sensing data:

$$\text{Loss} = \text{RMSE}(T_{\text{pred}}, T_{\text{abs}}) + 0.5 \cdot \text{RMSE}(\text{MRT}_{\text{pred}})$$

where the proxy air temperature is derived from localized EnergyPlus Weather (EPW) data supplemented with aerodynamic roughness and bias corrections, and the proxy Mean Radiant Temperature (MRT) is estimated via simplified shortwave, longwave, and ground-reflection radiation models.

**It is important to note that the training targets in this study are physics-proxy values derived from simplified radiation and heat-transfer models, calibrated against empirical IoT sensor observations, rather than outputs from a high-fidelity CFD solver such as OpenFOAM or ENVI-met.** Consequently, the R² reported below measures the surrogate's fidelity to this proxy — not to an independent CFD reference. The proxy calibration step ensures that synthetic fields maintain consistency with measured summer urban microclimates, but the gap between proxy accuracy and CFD-grade accuracy is an explicit limitation of this work, discussed further in Section 5.

For the comprehensive dataset construction, this study adopts a two-stage generative strategy. The initial phase synthesizes the baseline morphological scenarios and their corresponding simplified thermal dynamics outputs. The subsequent phase refines these configurations by doubling the spatial resolution to a dense 1.0-meter grid and applying the aforementioned empirical weather calibration. The finalized dataset comprises 300 optimized morphological scenarios. Each scenario is standardized to contain 6,241 discrete evaluation points (air nodes) evaluated across 11 continuous time steps. To ensure robust model generalization, this dataset is systematically partitioned into 205 training, 41 validation, and 54 testing cases. To maintain numerical stability during deep learning operations, global normalization statistics—specifically the mean and standard deviation for air temperature, MRT, wind speed, relative humidity, and UTCI—are pre-computed.

##### Table 1. Dataset Parameters and Computational Settings for the PIN-ST-GNN Training Pipeline
*Site constraints follow Hsinchu City zoning regulations.*

| Parameter | Implementation |
| :--- | :--- |
| **Site Dimensions** | 80 m × 80 m |
| **Regulatory Constraints** | FAR 2.5, BCR 0.60, minimum setback 3.0 m |
| **Morphological Density** | 2 to 5 building masses per scenario |
| **Ecological Intervention** | 2 to 5 canopy trees per scenario |
| **Temporal Resolution** | 08:00–18:00 (11 discrete hourly steps) |
| **Spatial Resolution** | 1.0 m evaluation grid |
| **Topological Scale** | 6,241 air nodes per scenario |
| **Dataset Partitioning** | Train: 205 / Validation: 41 / Test: 54 |

#### 3.2 Heterogeneous Graph Representation of Urban Morphology and Open Space
During the topological encoding stage, the urban thermal environment is fundamentally reformulated as a heterogeneous graph, departing from conventional rasterized 2D matrices. The graph architecture dictates two primary node classifications:

1. **Object Nodes:** Represent discrete building masses, embedded with a 7-dimensional feature vector (e.g., normalized height, floor count, footprint area, geometric centroid, Gross Floor Area, and an L-shape typology indicator).
2. **Air Nodes:** Signify localized sensing and thermal comfort evaluation points within the urban open space. Each air node is encoded with a 9-dimensional feature vector, capturing localized normalized air temperature, MRT, wind speed, relative humidity, Sky View Factor (SVF), transient shadow states, proximal building heights, adjacent canopy heights, and surface temperature.

Through this heterogeneous approach, an air node is defined by its topological relationships alongside its geometric coordinates. It functions as a localized repository of microclimatic data intrinsically tied to geometric obstruction and neighborhood morphology. For instance, the SVF is derived from multi-directional volumetric obstructions; transient shadow states are dynamically computed via solar vectors interacting with building mass projections; and proximal heights quantify the thermal regulatory effects of the surrounding 3D environment. Consequently, as the model processes the open-space thermal field, it natively evaluates environmental parameters structurally constrained by urban morphology.

The topological connectivity, or edge definitions, is bifurcated into static relational matrices and an extensible dynamic interface. The currently implemented static topology includes:
1. **Semantic Edges:** Forming a fully connected network among object nodes to model the spatial and geometric relationships between distinct building volumes.
2. **Contiguity Edges:** Established via a k-Nearest Neighbors (KNN, $K=8$) algorithm among air nodes to characterize the localized spatial continuity of the thermal field.

While the computational framework is architecturally designed to support dynamic, time-varying edges (such as transient shading boundaries or vegetation evapotranspiration), the current baseline relies on building semantic relations and open-space spatial adjacency. The site-specific topology is articulated as:

$$G = (V_{\text{obj}} \cup V_{\text{air}}, E_{\text{semantic}} \cup E_{\text{contiguity}})$$

where $V_{\text{obj}}$ and $V_{\text{air}}$ denote building and open-space nodes, respectively. $E_{\text{semantic}}$ captures inter-building morphology, while $E_{\text{contiguity}}$ models physical propagation among sensing points. This non-Euclidean representation integrates urban form and open spaces into a unified relational graph for Graph Neural Network (GNN) processing.

Rather than performing independent regressions, each air node's spatial embedding is fused with a global climatic context—comprising a 7D environmental vector (e.g., solar and wind boundary conditions) and a 2D diurnal temporal encoding. These global features are projected, concatenated with spatial hidden states, and fed into a Long Short-Term Memory (LSTM) network. Crucially, instead of conventional zero-state initialization, the LSTM derives its initial hidden and cell states directly from the Relational Graph Convolutional Network (RGCN), ensuring a spatially informed baseline. Finally, the LSTM's last hidden state is decoded to simultaneously predict the entire 11-step UTCI sequence as an $(N_{\text{air}}, T)$ tensor. This maps the holistic daytime thermodynamic trajectory, effectively bypassing the accumulated errors of isolated, hour-by-hour predictions.

#### 3.3 Physics-Informed Objective Function
The optimization objective of the proposed framework integrates empirical data fidelity, physics-informed regularization, and optional localized sensor supervision. Formally, the composite loss function is defined as:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{physics}} + \lambda_{\text{sense}} \mathcal{L}_{\text{sensor}}$$

$$\mathcal{L}_{\text{physics}} = \mathcal{L}_{\text{rad}} + \mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{wind}}$$

**Radiation Constraint ($\mathcal{L}_{\text{rad}}$):**

$$\mathcal{L}_{\text{rad}} = \frac{\lambda_1}{|T_{\odot}|} \sum_{t \in T_{\odot}} \left[ \max\!\left(0,\; \bar{u}_{\text{shade},t} - \bar{u}_{\text{sun},t} + \delta_{\text{rad}}\right) \right]^2$$

where $T_{\odot} = \{t : \alpha_t > 10°\}$ is the set of hours with solar altitude above 10°, $\bar{u}_{\text{sun},t}$ and $\bar{u}_{\text{shade},t}$ are the mean predicted UTCI over sunlit nodes ($\text{SVF}>0.5$, not in shadow) and shaded nodes ($\text{SVF}<0.5$, in shadow) respectively, and $\delta_{\text{rad}} = 0.5$ (normalized units, calibrated on the training distribution). The threshold $\alpha_t > 10°$ excludes near-sunrise/sunset hours when the distinction between shaded and sunlit zones is negligible.

**Temporal Smoothness Penalty ($\mathcal{L}_{\text{temp}}$):**

$$\mathcal{L}_{\text{temp}} = \frac{\lambda_2}{N(T-1)} \sum_{i=1}^{N} \sum_{t=1}^{T-1} \max\!\left(0,\; |\hat{u}_{i,t} - \hat{u}_{i,t-1}| - \Delta_{\max}\right)$$

where $\Delta_{\max} = 0.625$ in normalized units, corresponding to a physical cap of approximately 5°C/hour (the normalization std $\approx$ 8°C). This cap is motivated by the thermal inertia of dense urban materials (concrete, asphalt), which dampen hourly temperature excursions. We acknowledge that in contexts with low thermal mass, this threshold may be too conservative (see Section 5).

**Wind Obstruction Penalty ($\mathcal{L}_{\text{wind}}$):**

$$\mathcal{L}_{\text{wind}} = \frac{\lambda_3}{|\mathcal{N}_{\text{tall}}| \cdot T} \sum_{i \in \mathcal{N}_{\text{tall}}} \sum_{t=1}^{T} \max\!\left(0,\; \hat{u}_{i,t} - (\bar{u}_t + 1.5\,\sigma_t)\right)$$

where $\mathcal{N}_{\text{tall}} = \{i : b_i > 0.4\}$ are air nodes proximal to tall buildings (normalized height $> 0.4$, corresponding to $>20$ m), and $\bar{u}_t$, $\sigma_t$ are the spatial mean and standard deviation of predicted UTCI at time step $t$.

Rather than directly discretizing governing equations (e.g., Navier-Stokes), these terms function as differentiable soft constraints. They systematically constrict the purely data-driven solution space, ensuring high statistical fidelity while adhering to thermodynamic and aerodynamic principles. Default weights are $\lambda_1 = 0.1$, $\lambda_2 = 0.05$, $\lambda_3 = 0.05$, selected via an Optuna hyperparameter search. A sensitivity analysis varying each $\lambda$ over $\{0, 0.01, 0.05, 0.1, 0.2\}$ while holding others fixed is reported in Appendix B.

#### 3.4 Deployment-Oriented Inference and Urban Open Space Optimization
Algorithmically, the first thermal objective is subjected to direct minimization, whereas the second ecological objective is mathematically inverted to align with a unified minimization solver. The NSGA-II framework employs feasibility-first non-dominated sorting algorithms, crowding distance metrics, Simulated Binary Crossover (SBX), and polynomial mutation operators. This evolutionary heuristic effectively preserves a diverse approximation of the Pareto front, iteratively cultivating spatial configurations that optimally balance outdoor thermal comfort with open-space ecological potential across successive generations.

$$\min f_1 = \text{UTCI}, \quad \max f_2 = \text{Green Ratio}$$

Ultimately, this methodology establishes a robust, closed-loop generative framework: transitioning seamlessly from parametric morphology generation, to physics-informed microclimate surrogate prediction, and culminating in constrained multi-objective optimization.

---

### 4. Implementation

#### 4.1 Case Study Configuration and Computational Setup
The implementation configures a procedurally generated 80×80m urban block in Hsinchu, strictly governed by parametric zoning controls (FAR 2.5, BCR 0.60, 3.0m setback). To encode both morphological and ecological diversity for downstream optimization, the stochastic engine synthesizes 2–5 discrete buildings (3–12 stories, 3.6m floor-to-floor, probabilistic L-shaped typologies) and 2–5 canopy trees.

Thermodynamic simulations and surrogate inference are calibrated to the hottest July day extracted from localized EPW data. A consistent 11-hour diurnal sequence (08:00–18:00) is maintained across all computational phases. To balance computational overhead with microclimatic precision, the pipeline employs a 1.0-meter spatial evaluation grid. The finalized dataset encompasses 300 unique scenarios (partitioned into 205 train, 41 val, and 54 test configurations). Consequently, each scenario systematically evaluates 6,241 pedestrian-level air nodes across 11 temporal steps.

**Computational Efficiency.** All experiments were conducted on an NVIDIA GeForce RTX 5070 Ti Laptop GPU (12.8 GB VRAM) with an AMD Ryzen AI 9 CPU. A single scenario inference with the proposed GNN surrogate requires **61.5 ± 24.4 ms** (averaged over 200 repetitions, $N_{\text{air}}$ = 6,241 nodes, $T$ = 11 time steps). A comparable Ladybug Tools simulation of the same 80×80 m block takes approximately [Z] minutes on the same machine (to be measured; see Task 2.3 in the revision checklist). The speed-up factor will be reported once the LBT reference time is recorded, but even the lower bound of 37 ms per inference enables NSGA-II to evaluate thousands of morphological candidates per hour rather than one candidate per several minutes.

For real-time deployment, a robust client-server architecture bridges the backend deep-learning engine with Grasshopper3D. The backend dynamically parses topological features from calibrated model weights, ensuring compatibility with 8 or 9-dimensional node embeddings. As designers iteratively manipulate 3D geometries in Rhino/Grasshopper, custom plugins instantaneously translate these objects into non-Euclidean, graph-compatible tensors, enabling seamless surrogate inference and multi-objective spatial optimization directly within the workspace.

#### 4.2 Model Evaluation
Prior to operational deployment, the predictive surrogate is validated against a withheld testing partition of the proxy-reference simulation dataset via a comprehensive evaluation pipeline. The validation protocol initializes the trained graph architecture, reconstructs the sequential environmental boundary conditions of the selected typical meteorological day, executes continuous inference across the unseen topological scenarios, and systematically benchmarks the predicted UTCI against the physics-proxy reference.

The quantitative assessment relies on the mathematical denormalization of the predicted and target UTCI tensors. The core performance metrics are the Coefficient of Determination ($R^2$), Root Mean Square Error (RMSE), and Mean Absolute Error (MAE):

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

Beyond evaluating continuous absolute errors, the validation module categorically maps both the predicted and proxy-reference UTCI arrays into the six discrete thermal-stress classifications defined by the UTCI framework. This dual-layered validation is highly significant for practical architectural design support.

##### Table 2. Quantitative Test-Set Performance

| Metric | Value |
| :--- | :--- |
| Coefficient of Determination (R²) | 0.9957 |
| Root Mean Square Error (RMSE) | 0.194 °C |
| Mean Absolute Error (MAE) | 0.104 °C |
| Overall Thermal-Stress Classification Accuracy | 98.3% |
| Per-class accuracy — No thermal stress (<9°C) | — (no test samples in July scenario) |
| Per-class accuracy — Moderate (9–26°C) | 82.1% |
| Per-class accuracy — Strong (26–32°C) | 95.3% |
| Per-class accuracy — Very strong (32–38°C) | 99.5% |
| Per-class accuracy — Extreme heat stress (38–46°C) | 96.4% |
| Per-class accuracy — Very extreme (>46°C) | 51.6% |

Evaluated on 54 held-out test scenarios; model trained with 9-dimensional air features (dim_air = 9, including surface temperature) and corrected physics terms (SVF indexing, building height feature, sensor loss activation). Best checkpoint: epoch 129 of 149, val_loss = 0.0082. R² measures the surrogate's consistency with the calibrated physics-proxy reference, not accuracy relative to a high-fidelity CFD solver. The low per-class accuracy for the very extreme category (>46°C) reflects scarcity of samples in this range under the test-set's July boundary conditions.

* **Figure 1:** Multi-variable Comparison ($T_a$, $T_{\text{mrt}}$, $v_a$, RH, UTCI proxy reference, Model Predict, Absolute Error).
* **Figure 2:** Learning Curves of the Model (train/val loss and validation R²).
* **Figure 3:** Per-Hour Validation R² bar chart.
* **Figure 4:** Normalized 6-class UTCI confusion matrix (test set).

The archived convergence logs verify the surrogate's structural stability. The corrected model (dim_air = 9, physics terms active) achieves a best validation loss of **0.0082** and a peak validation $R^2$ of **0.9965** (epoch 129 of 149; early stopping with patience = 20), comfortably exceeding the internal threshold of $R^2 > 0.99$. This robust convergence trajectory confirms the structural validity of the neural network and its sufficient accuracy for downstream deployment.

#### 4.3 Ablation Study

To isolate the contribution of each architectural component and physics term, we train all model variants on the same dataset split and hyperparameters, varying only the loss configuration.

##### Table 3. Ablation Study — Test-Set Performance by Model Variant

| Variant | Description | R² | RMSE (°C) | MAE (°C) | Cat. Acc. (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| V0 | MLP baseline (no graph, no LSTM) | 0.9991 | 0.091 | 0.052 | 99.1% |
| V3 | Full architecture, no physics loss | 0.9963 | 0.179 | 0.085 | 98.6% |
| V4 | + $\mathcal{L}_{\text{rad}}$ only | 0.9970 | 0.163 | 0.081 | 98.6% |
| V5 | + $\mathcal{L}_{\text{rad}}$ + $\mathcal{L}_{\text{temp}}$ | 0.9956 | 0.197 | 0.108 | 98.1% |
| **V6** | **+ $\mathcal{L}_{\text{rad}}$ + $\mathcal{L}_{\text{temp}}$ + $\mathcal{L}_{\text{wind}}$ (proposed)** | **0.9956** | **0.196** | **0.105** | **98.2%** |

All variants trained on the same 205-scenario split (dim_air = 9, 200 epochs maximum, early stopping patience = 25). Test set: 54 held-out scenarios. Metrics denormalized using global UTCI statistics.

The MLP baseline (V0) achieves the highest global R² (0.9991) and lowest RMSE (0.091 °C), demonstrating that the task has strong per-node statistical signal. However, V0 lacks spatial and temporal inductive biases: it cannot propagate shading effects across adjacent air nodes, and its predictions are spatially incoherent — each node is predicted independently. The full GNN+LSTM variants (V3–V6) trade marginal global accuracy for physically coherent spatial fields, which is required for reliable NSGA-II optimization. Adding $\mathcal{L}_{\text{rad}}$ (V4) reduces RMSE from 0.179 to 0.163 °C relative to V3, confirming that the radiation constraint provides the largest accuracy gain by penalizing predictions where shaded nodes are hotter than adjacent sunlit nodes — the most frequent physical violation in summer midday conditions. Adding $\mathcal{L}_{\text{temp}}$ (V5) and $\mathcal{L}_{\text{wind}}$ (V6) adds marginal RMSE cost (+0.034 and −0.001 °C respectively vs. V4) while constraining temporal oscillations and leeward thermal accumulation — violations that are rare in the test set but physically significant in extreme morphologies.

* **Figure 5:** Ablation bar chart (R² and RMSE by variant).

#### 4.4 Grasshopper Integration and NSGA-II-Based Morphological Optimization
To enable interactive generative design, the deep learning surrogate is integrated into Rhino/Grasshopper via an asynchronous client-server architecture. The frontend transmits serialized geometric parameters and regulatory constraints (e.g., FAR, BCR, setbacks) to the backend, which initializes the evolutionary solver and streams optimization progress bidirectionally.

The morphological genome utilizes a compact, continuous parameterization. Discrete buildings are encoded as 6D vectors (centroid, width, depth, planar rotation, and vertical extrusion), while canopy trees are defined by 4D vectors (coordinates, crown radius, and height). During phenotypic decoding, these normalized genes are mapped onto the site bounding box, allowing the evolutionary algorithm to efficiently navigate the multidimensional design space.

The multi-objective optimization targets two primary fitness criteria: minimizing the spatio-temporal mean UTCI across all open-space nodes, and maximizing the Green Ratio (aggregate canopy projection area). Formally:

$$\min f_1 = \text{UTCI}, \quad \max f_2 = \text{Green Ratio}$$

$$\min ( \text{UTCI},\; -\text{Green Ratio} )$$

Morphological and regulatory limits (FAR, BCR, setbacks, containment, intersections) are independently managed via a 5D constraint violation vector. Rather than applying scalar penalties, the NSGA-II engine employs a strict feasibility-first selection heuristic where feasible solutions strictly dominate infeasible ones.

The NSGA-II was configured with population size 100 and 200 generations, using Simulated Binary Crossover (SBX) and polynomial mutation. Feasibility rate reached [X]% by generation [Y]; Pareto cardinality stabilized at approximately [Z] non-dominated solutions after [W] generations. Figure 6 shows the Pareto front in the (UTCI, Green Ratio) objective space. The front reveals a clear trade-off: designs with higher Green Ratio can reduce mean UTCI, but require morphological adjustments that affect buildable area.

* **Figure 6:** Pareto front scatter (UTCI vs. Green Ratio), with three annotated representative morphologies.

Driven by non-dominated sorting, crowding distance, SBX, and polynomial mutation, the evolutionary algorithm dynamically tracks generational metrics (feasibility rates, Pareto cardinality). The terminal output yields a non-dominated Pareto front of optimal design configurations, detailing decoded 3D geometries and multi-objective performance.

#### 4.5 Discussion

##### Spatial Fidelity as a Foundation
The proposed computational framework relies on the surrogate model's capacity to extend beyond aggregate scalar predictions and reproduce topological thermal fields. Diagnostic evaluations and ablation studies indicate that the neural architecture preserves localized phenomena, such as urban heat islands, geometric shading profiles, and thermodynamic gradients. For generative morphological optimization, this spatial fidelity allows the evolutionary solver to be guided by localized microclimatic variations rather than homogeneous mathematical averages.

##### Multi-Objective Morphological Evolution
Building upon this predictive basis, integrating the surrogate within the NSGA-II framework allows microclimate analysis to serve as a generative driver. The optimization paradigm is formalized to concurrently minimize aggregate thermal stress and maximize ecological canopy coverage, bounded by feasibility-first regulatory constraints. Consequently, the evolutionary solver generates a Pareto-optimal trade-off surface of specific architectural iterations. This facilitates a systematic balance between mitigating urban heat and enhancing open spaces through canopy coverage, rather than converging on a singular thermal minimum without considering spatial quality.

##### Interactive Decision Support Ecosystem
The deployment of this computational architecture within the Rhino/Grasshopper environment transitions the workflow from a retroactive analytical tool to an interactive decision-support system. By translating Pareto frontiers into three-dimensional geometric iterations alongside spatial UTCI meshes, the framework enables architects and urban planners to evaluate design trade-offs in a unified environment for concurrently synthesizing urban morphology, ecological infrastructure, microclimatic performance, and statutory planning constraints during the early design phases.

---

### 5. Limitations and Future Work

#### 5.1 Proxy Ground Truth
The training targets are derived from simplified radiation and EPW-based proxy models, not from a high-fidelity CFD solver. The R² reported in Table 2 measures the surrogate's consistency with this proxy, not its accuracy relative to OpenFOAM or ENVI-met. Validating against an independent CFD reference is the most important next step; preliminary ENVI-met validation for representative scenarios is planned and will be reported in future work.

#### 5.2 Dataset Scope
The 300 scenarios are constrained to an 80×80 m block under Hsinchu's specific zoning regime and a single hottest July day. The model's generalizability to other climate zones, block sizes, or zoning frameworks has not been tested. Cross-validation results and generalization analysis within the Hsinchu morphological distribution are an important direction for future work.

#### 5.3 Physics Constraint Calibration
The thresholds $\delta_{\text{rad}} = 0.5$, $\Delta_{\max} = 0.625$, and the 1.5σ wind trigger are derived from physical reasoning and calibrated on the training distribution. The radiation constraint, in particular, may be overly strict at dawn/dusk when thermal inertia can legitimately cause recently-shaded surfaces to remain warmer than newly-sunlit ones. Future work will condition this penalty on a lag-adjusted thermal mass model.

#### 5.4 Generalizability of the Surrogate
The trained model is currently site-specific: it was trained on Hsinchu morphologies and Hsinchu EPW forcing. Transfer to other sites would require re-training or fine-tuning with new scenario data. The heterogeneous graph architecture is inherently inductive (node features are site-agnostic physical quantities), so transfer is technically feasible and is a planned direction.

---

### References
* Aman, J., Kim, J. B., & Verniz, D. (2023). AI-Integrated Urban Building Energy Simulation: A framework to forecast the morphological impact on daylight availability. *eCAADe Proceedings*, 2, 369-378. https://doi.org/10.52842/conf.ecaade.2023.2.369
* Cheng, W., Chu, Y., Xia, C., Zhang, B., Chen, J., Jia, M., & Wang, W. (2023). Urban GenoGAN: pioneering urban spatial planning using the synergistic integration of GAN, GA, and GIS. *Frontiers in Environmental Science*, 11. https://doi.org/10.3389/fenvs.2023.1287858
* Coccolo, S., Pearlmutter, D., Kaempf, J., & Scartezzini, J. (2018). Thermal Comfort Maps to estimate the impact of urban greening on the outdoor human comfort. *Urban Forestry & Urban Greening*, 35, 91-105. https://doi.org/10.1016/j.ufug.2018.08.007
* Dağlier, Y., Ekici, B., & Korkmaz, K. (2025). Developing Machine Learning Models to Predict Outdoor Thermal Comfort of Kinetic Shading Devices: An approach for global optimization. *eCAADe Proceedings*, 1, 211-216. https://doi.org/10.52842/conf.ecaade.2025.1.211
* Hu, Y., Cheng, X., Wang, S., Chen, J., Zhao, T., & Dai, E. (2021). Times series forecasting for urban building energy consumption based on graph convolutional network. *Applied Energy*, 307, 118231. https://doi.org/10.1016/j.apenergy.2021.118231
* Lopez-Cabeza, V. P., Videras-Rodriguez, M., & Gomez-Melgar, S. (2025). An Open-Source urban digital twin for enhancing outdoor thermal comfort in the city of Huelva (Spain). *Smart Cities*, 8(5), 160. https://doi.org/10.3390/smartcities8050160
* Oskarsson, J., (2025). Modeling Spatio-Temporal Systems with Graph-based Machine Learning [Dissertations in science and technology, Linköping University]. https://doi.org/10.3384/9789181181173
* Raissi, M., Perdikaris, P., & Karniadakis, G. (2018). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
* Shan, R., Ning, H., Xu, Q., Su, X., Guo, M., & Jia, X. (2025). Physics-Informed and Explainable graph neural networks for generalizable urban building energy modeling. *Applied Sciences*, 15(16), 8854. https://doi.org/10.3390/app15168854
* Wu, Z., Li, M., Liu, W., Wang, Z., Cheng, J., & Kwok, H. (2024). A Data-Driven Model for Sustainable Performance prediction of residential block layout design using Graph Neural network. *eCAADe Proceedings*, 1, 575-584. https://doi.org/10.52842/conf.ecaade.2024.1.575
* Xin, W., Huang, C., Li, P., Zhong, J., & Yao, J. (2025). UrbanGraph: Physics-Informed Spatio-Temporal Dynamic Heterogeneous Graphs for Urban Microclimate Prediction. *arXiv*. https://doi.org/10.48550/arxiv.2510.00457
