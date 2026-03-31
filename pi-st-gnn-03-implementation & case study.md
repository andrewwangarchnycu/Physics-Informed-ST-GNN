# Building Morphology and Urban Open Space Optimization: A Physics-Informed Graph Neural Network Approach

## 3. IMPLEMENTATION & CASE STUDY

### 3.1 Case Study Configuration and Computational Setup

The implementation was configured around a synthetic urban block representing an 80 m x 80 m site in the Hsinchu climatic context, as defined in `site_constraints.yaml` and the associated data-generation scripts. The site uses a minimum setback distance of 3.0 m, a floor-to-floor height of 3.6 m for scenario generation, and planning controls of FAR = 2.5 and BCR = 0.60. Within these bounds, the geometry sampler produces 2 to 5 buildings per scenario, with floor counts ranging from 3 to 12 and optional L-shaped building footprints. The same setup also places 2 to 5 trees in the remaining open space so that both building morphology and green intervention can be represented in downstream prediction and optimization.

The thermal simulations and surrogate inference are both aligned to the hottest typical day in July extracted from the EPW weather file. In the current implementation, the temporal window is fixed to 11 hourly steps from 08:00 to 18:00. This temporal framing is shared across data generation, model training, deployment, and optimization, which ensures that the surrogate model is always queried under the same climatic horizon used during dataset construction. While the codebase leaves room for additional site-specific documentation, such as aerial imagery and surrounding urban context, the implemented case-study logic is centered on a controlled, regulation-aware block prototype rather than a single manually modeled real parcel.

Two spatial resolutions are present in the repository. The earlier v1 dataset uses a coarser 2.0 m sampling grid, while the v2 workflow upgrades the scenario resolution to 1.0 m. The high-resolution v2 dataset is the main implementation target for training and monitoring. According to `dataset_summary_v2.json`, each v2 scenario contains 6,241 air nodes and 11 time steps, and the final dataset includes 300 scenarios split into 205 training, 41 validation, and 54 test cases. This resolution is important for the case study because it allows the open-space thermal field to be evaluated not just at a few representative points, but across a dense network of pedestrian-level sensing nodes.

At runtime, the deployment stack is organized through a FastAPI server (`06_deployment/app.py`) and Grasshopper components (`UTCIPredictor.ghpy` and `UTCIOptimizer.ghpy`). The server loads the EPW climate pickle, normalization statistics, and the trained checkpoint, then automatically detects the air-feature dimension from the checkpoint weights. This is especially relevant because the current codebase supports both the original 8-dimensional air feature set and the newer 9-dimensional version that adds surface temperature. Geometry received from Rhino/Grasshopper is converted into graph-compatible inputs before inference or optimization is executed.

### 3.2 Surrogate Model Evaluation Against Ground Truth

Before deployment, the surrogate model is evaluated against withheld simulation data through an explicit test-split pipeline implemented in `04_training/evaluate.py`, `04_training/retrain_v2.py`, and `05_evaluation_visualization.py`. The evaluation procedure loads the trained graph model, rebuilds the environmental forcing sequence from the same July typical day, runs inference on each test scenario, and then compares predicted UTCI against the stored HDF5 ground truth. The main metrics computed in the code are coefficient of determination ($R^2$), root mean square error (RMSE), mean absolute error (MAE), and UTCI thermal-stress classification accuracy.

The metric calculation follows the denormalization of predicted and target UTCI values:

$$
R^2 = 1-\frac{\sum_i (\hat{y}_i-y_i)^2}{\sum_i (y_i-\bar{y})^2}, \qquad
RMSE = \sqrt{\frac{1}{n}\sum_i (\hat{y}_i-y_i)^2}, \qquad
MAE = \frac{1}{n}\sum_i |\hat{y}_i-y_i|.
$$

In addition to continuous UTCI error, the evaluation module converts predicted and reference UTCI into six thermal-stress classes using threshold bands derived from the UTCI framework. This makes the validation more meaningful for design support, because a model that slightly misses the continuous value but preserves the stress category may still remain decision-relevant in practice.

The current repository snapshot contains the full evaluation logic and the plotting routines for scatter plots, residual histograms, per-hour $R^2$, and confusion matrices, but not a committed final `eval_results_v2.json` file with test-set values. However, the available training record still provides a reliable view of model convergence. `checkpoints_v2/training_history.json` shows that the high-resolution model reaches a best validation loss of approximately 0.00748 and a peak validation $R^2$ of about 0.9966. Earlier progress logs also record that the v2 model exceeded the internal deployment target of validation $R^2 > 0.99`. Thus, while the exact test metrics are generated at runtime rather than stored in the repository snapshot, the implemented validation pipeline is complete and the recorded convergence history indicates that the surrogate is sufficiently accurate for downstream deployment.

This distinction is methodologically important. The codebase separates three levels of evidence: (1) withheld-split evaluation through `evaluate.py`; (2) richer visual validation, including scatter and residual analysis, through `05_evaluation_visualization.py`; and (3) deployment readiness checks through `retrain_v2.py` and progress monitoring scripts. In other words, model validation is not a single scalar check, but a layered procedure combining numerical accuracy, temporal consistency, and classification reliability. Table 2 summarizes the validation outputs currently implemented in the repository.

| Validation item | Implemented in code |
| --- | --- |
| Continuous UTCI accuracy | $R^2$, RMSE, MAE |
| Thermal stress evaluation | Category accuracy and per-class accuracy |
| Visual diagnostics | Scatter plot, residual distribution, per-hour $R^2$, confusion matrix |
| Repository-confirmed convergence | Best validation loss approx. 0.00748; peak validation $R^2$ approx. 0.9966 |

### 3.3 Grasshopper Integration and NSGA-II-Based Morphological Optimization

To support interactive morphological evolution, the surrogate model is connected to Rhino 8 Grasshopper through a WebSocket-based client-server architecture. The FastAPI server exposes a `/ws` endpoint and accepts JSON messages with actions such as `predict`, `optimize`, and `cancel`. On the Grasshopper side, `UTCIOptimizer.ghpy` packages the site boundary, setback, FAR and BCR limits, chromosome settings, and optimization parameters into a request that is sent to the Python backend. The server then instantiates `ChromosomeConfig`, `ConstraintChecker`, `FitnessEvaluator`, and `NSGA2Optimizer`, and streams optimization progress back to Grasshopper asynchronously.

The chromosome encoding in the implemented system is continuous and compact. Each building is represented by six genes corresponding to centroid coordinates, footprint width, footprint depth, rotation, and floor count. Each tree is represented by four genes corresponding to position, crown radius, and height. During decoding, these normalized genes are mapped into the site bounding box and the prescribed design ranges. This allows the optimizer to explore a broad but controlled design space without handcrafting separate parametric scripts for every layout variant.

The optimization objectives in the actual code differ from the simplified statement in the draft text. The first objective is not the maximum UTCI but the mean UTCI over all air nodes and all predicted hours. The second objective is the green ratio, computed from total tree-canopy area relative to site area. Therefore, the optimization problem implemented in `fitness.py` is:

$$
\min f_1=\overline{UTCI}, \qquad \max f_2=Green\ Ratio,
$$

which is internally written as minimization of $[\overline{UTCI}, -Green\ Ratio]$. Constraint handling is implemented separately through a five-term violation vector covering FAR, BCR, setback, site containment, and building overlap. Rather than folding these limits into a simple scalar penalty inside the fitness function, the NSGA-II engine uses a feasibility-first strategy: feasible solutions dominate infeasible ones, and infeasible solutions are ranked by total constraint violation. This is a closer match to planning practice, where regulatory compliance acts as a hard filter rather than a soft preference.

The optimization loop itself uses non-dominated sorting, crowding distance, SBX crossover, and polynomial mutation. Progress messages report generation count, number of feasible individuals, best mean UTCI, best green ratio, and current Pareto-front size. Final results are returned as a sorted list of Pareto designs, each containing the decoded geometry, mean UTCI, green ratio, FAR, BCR, and remaining violation level. As a result, the case-study workflow is not limited to single-design prediction: it becomes a closed design exploration system in which morphology, open-space planting, thermal comfort, and planning constraints are evaluated within the same computational environment.
