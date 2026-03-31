"""
07_optimization/nsga2_engine.py
════════════════════════════════════════════════════════════════
NSGA-II [REMOVED_ZH:8]（[REMOVED_ZH:5]，[REMOVED_ZH:1] numpy [REMOVED_ZH:2]）

[REMOVED_ZH:3]：Deb et al. 2002 "A Fast and Elitist Multiobjective
         Genetic Algorithm: NSGA-II"

[REMOVED_ZH:2]：
  - Feasibility-first [REMOVED_ZH:2]：[REMOVED_ZH:17]
  - Non-dominated Sorting + Crowding Distance
  - SBX Crossover + Polynomial Mutation
  - [REMOVED_ZH:3]Run：[REMOVED_ZH:2] asyncio.Queue [REMOVED_ZH:6] WebSocket
"""
from __future__ import annotations
import asyncio
import time
import threading
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from chromosome import ChromosomeConfig, random_individual, sbx_crossover, polynomial_mutation
from fitness import FitnessEvaluator


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════

def fast_non_dominated_sort(F: np.ndarray,
                             CV: np.ndarray) -> List[List[int]]:
    """
    Feasibility-first non-dominated sort。

    Parameters
    ----------
    F  : (n, n_obj) — [REMOVED_ZH:3]（[REMOVED_ZH:4]）
    CV : (n, n_con) — [REMOVED_ZH:6]（0 = [REMOVED_ZH:2]）

    Returns
    -------
    fronts : list of index lists, fronts[0] = Pareto [REMOVED_ZH:2]
    """
    n = len(F)
    total_cv = CV.sum(axis=1)          # (n,) total constraint violation

    # [REMOVED_ZH:4] vs. [REMOVED_ZH:5]
    feasible   = np.where(total_cv < 1e-6)[0]
    infeasible = np.where(total_cv >= 1e-6)[0]

    def dominates(i, j):
        """[REMOVED_ZH:2] i [REMOVED_ZH:4] j（[REMOVED_ZH:10]）"""
        return (np.all(F[i] <= F[j]) and np.any(F[i] < F[j]))

    fronts: List[List[int]] = []
    remaining = list(feasible)

    while remaining:
        front = []
        dominated_count = {idx: 0 for idx in remaining}
        dominates_set   = {idx: [] for idx in remaining}

        for i in remaining:
            for j in remaining:
                if i == j:
                    continue
                if dominates(i, j):
                    dominates_set[i].append(j)
                elif dominates(j, i):
                    dominated_count[i] += 1

        for idx in remaining:
            if dominated_count[idx] == 0:
                front.append(idx)

        if not front:
            front = list(remaining)  # [REMOVED_ZH:5]（[REMOVED_ZH:8]）

        fronts.append(front)
        remaining = [idx for idx in remaining if idx not in set(front)]

    # [REMOVED_ZH:15]
    if len(infeasible) > 0:
        sorted_inf = infeasible[np.argsort(total_cv[infeasible])].tolist()
        fronts.append(sorted_inf)

    return fronts


def crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    """
    compute front [REMOVED_ZH:5]Crowding Distance。

    Returns
    -------
    dist : (len(front),) — Crowding Distance（[REMOVED_ZH:4]）
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n)
    F_front = F[front]

    for m in range(F.shape[1]):
        order   = np.argsort(F_front[:, m])
        f_min   = F_front[order[0],  m]
        f_max   = F_front[order[-1], m]
        span    = f_max - f_min + 1e-12

        dist[order[0]]  = np.inf
        dist[order[-1]] = np.inf
        for k in range(1, n - 1):
            dist[order[k]] += (F_front[order[k+1], m]
                              - F_front[order[k-1], m]) / span

    return dist


def tournament_selection(rank: np.ndarray,
                          crowd: np.ndarray,
                          n_select: int) -> np.ndarray:
    """
    Binary tournament selection：rank [REMOVED_ZH:5]；rank [REMOVED_ZH:3]
    crowding distance [REMOVED_ZH:5]。

    Returns
    -------
    selected : (n_select,) index into population
    """
    n    = len(rank)
    sel  = np.empty(n_select, dtype=int)
    idxs = np.arange(n)

    for i in range(n_select):
        a, b = np.random.choice(idxs, 2, replace=False)
        if rank[a] < rank[b]:
            sel[i] = a
        elif rank[b] < rank[a]:
            sel[i] = b
        elif crowd[a] >= crowd[b]:
            sel[i] = a
        else:
            sel[i] = b

    return sel


# ════════════════════════════════════════════════════════════════
# NSGA-II [REMOVED_ZH:3]
# ════════════════════════════════════════════════════════════════

class NSGA2Optimizer:
    """
    Parameters
    ----------
    evaluator  : FitnessEvaluator
    cfg        : ChromosomeConfig
    pop_size   : [REMOVED_ZH:4]（[REMOVED_ZH:4]）
    n_gen      : [REMOVED_ZH:5]
    crossover_rate : SBX Crossover[REMOVED_ZH:2]
    mutation_eta   : polynomial mutation eta [REMOVED_ZH:2]
    seed       : [REMOVED_ZH:4]
    """

    def __init__(self,
                 evaluator:       FitnessEvaluator,
                 cfg:             ChromosomeConfig,
                 pop_size:        int   = 40,
                 n_gen:           int   = 50,
                 crossover_rate:  float = 0.9,
                 mutation_eta:    float = 20.0,
                 seed:            int   = 42):
        self.evaluator      = evaluator
        self.cfg            = cfg
        self.pop_size       = pop_size
        self.n_gen          = n_gen
        self.crossover_rate = crossover_rate
        self.mutation_eta   = mutation_eta
        self.seed           = seed

        self._cancel_flag   = threading.Event()

    def cancel(self):
        """[REMOVED_ZH:3]Run[REMOVED_ZH:6]"""
        self._cancel_flag.set()

    # ── [REMOVED_ZH:2]Run（[REMOVED_ZH:1] ThreadPoolExecutor [REMOVED_ZH:3]）──────────────

    def run_sync(self,
                 progress_callback: Callable[[dict], None] | None = None
                 ) -> dict:
        """
        Run NSGA-II [REMOVED_ZH:9]。

        progress_callback(info) [REMOVED_ZH:8]，info [REMOVED_ZH:2]：
          generation, n_feasible, best_utci, best_green,
          pareto_designs (list of dict)
        """
        np.random.seed(self.seed)
        self._cancel_flag.clear()

        n = self.pop_size
        # [REMOVED_ZH:5]
        pop = np.array([random_individual(self.cfg) for _ in range(n)])
        F, CV = self.evaluator.batch_evaluate(pop)

        for gen in range(1, self.n_gen + 1):
            if self._cancel_flag.is_set():
                break

            t0 = time.perf_counter()

            # ── compute rank [REMOVED_ZH:1] crowding distance ──────────────
            fronts = fast_non_dominated_sort(F, CV)
            rank   = np.empty(n, dtype=int)
            crowd  = np.zeros(n)

            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
                cd = crowding_distance(F, front)
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]

            # ── [REMOVED_ZH:2] → Crossover → Mutation → [REMOVED_ZH:2] ────────────────────
            sel = tournament_selection(rank, crowd, n)
            np.random.shuffle(sel)

            offspring = np.empty_like(pop)
            for i in range(0, n - 1, 2):
                p1, p2 = pop[sel[i]], pop[sel[i+1]]
                c1, c2 = sbx_crossover(p1, p2, prob=self.crossover_rate)
                c1 = polynomial_mutation(c1, self.mutation_eta)
                c2 = polynomial_mutation(c2, self.mutation_eta)
                offspring[i]   = c1
                offspring[i+1] = c2
            if n % 2 == 1:
                offspring[-1] = polynomial_mutation(
                    pop[sel[-1]], self.mutation_eta)

            F_off, CV_off = self.evaluator.batch_evaluate(offspring)

            # ── [REMOVED_ZH:4] + [REMOVED_ZH:2]，[REMOVED_ZH:5] N ─────────────────
            combined     = np.vstack([pop, offspring])
            F_combined   = np.vstack([F, F_off])
            CV_combined  = np.vstack([CV, CV_off])

            fronts_c = fast_non_dominated_sort(F_combined, CV_combined)
            rank_c   = np.empty(2*n, dtype=int)
            crowd_c  = np.zeros(2*n)

            for r, front in enumerate(fronts_c):
                for idx in front:
                    rank_c[idx] = r
                cd = crowding_distance(F_combined, front)
                for k, idx in enumerate(front):
                    crowd_c[idx] = cd[k]

            # [REMOVED_ZH:1] (rank, -crowding_distance) [REMOVED_ZH:2]，[REMOVED_ZH:2] N
            order = np.lexsort((-crowd_c, rank_c))[:n]
            pop   = combined[order]
            F     = F_combined[order]
            CV    = CV_combined[order]

            # ── [REMOVED_ZH:4] ──────────────────────────────────────
            if progress_callback is not None:
                feasible_mask = CV.sum(axis=1) < 1e-6
                n_feasible    = int(feasible_mask.sum())
                best_utci     = float(F[feasible_mask, 0].min()) \
                                if n_feasible > 0 else float("nan")
                best_green    = float(-F[feasible_mask, 1].max()) \
                                if n_feasible > 0 else float("nan")

                elapsed = time.perf_counter() - t0
                pareto  = self._extract_pareto(pop, F, CV)
                progress_callback({
                    "generation":   gen,
                    "n_gen":        self.n_gen,
                    "n_feasible":   n_feasible,
                    "best_utci":    round(best_utci, 2),
                    "best_green":   round(best_green, 4),
                    "elapsed_s":    round(elapsed, 2),
                    "pareto_count": len(pareto),
                })

        # ── [REMOVED_ZH:4] ──────────────────────────────────────────
        pareto_designs = self._extract_pareto(pop, F, CV)
        return {
            "status":          "cancelled" if self._cancel_flag.is_set() else "complete",
            "generations_run": gen if self._cancel_flag.is_set() else self.n_gen,
            "pareto_designs":  pareto_designs,
        }

    # ── [REMOVED_ZH:5]（[REMOVED_ZH:1] FastAPI WebSocket handler [REMOVED_ZH:2]）──────

    async def run_async(self,
                        loop:    asyncio.AbstractEventLoop,
                        queue:   asyncio.Queue,
                        executor) -> dict:
        """
        [REMOVED_ZH:1] ThreadPoolExecutor [REMOVED_ZH:1]Run run_sync，
        [REMOVED_ZH:3] asyncio.Queue [REMOVED_ZH:5] WebSocket handler。
        """
        def callback(info: dict):
            asyncio.run_coroutine_threadsafe(queue.put(info), loop)

        result = await loop.run_in_executor(
            executor, self.run_sync, callback)
        return result

    # ── [REMOVED_ZH:2] ──────────────────────────────────────────────────

    def _extract_pareto(self, pop, F, CV) -> List[dict]:
        """
        [REMOVED_ZH:7] Pareto [REMOVED_ZH:2]（[REMOVED_ZH:5]），
        [REMOVED_ZH:13] dict list。
        """
        from chromosome import decode
        feasible_mask = CV.sum(axis=1) < 1e-6
        if not feasible_mask.any():
            return []

        f_idx = np.where(feasible_mask)[0]
        fronts = fast_non_dominated_sort(F[feasible_mask], CV[feasible_mask])
        if not fronts:
            return []

        pareto_local_idx = fronts[0]
        pareto_global    = f_idx[pareto_local_idx]

        results = []
        for gi in pareto_global:
            design = decode(pop[gi], self.cfg)
            cv_sum = float(CV[gi].sum())

            # [REMOVED_ZH:5]
            from constraints import ConstraintChecker
            check = self.evaluator.checker.check_all(design)

            results.append({
                "genes":        pop[gi].tolist(),
                "design":       design.to_dict(),
                "mean_utci":    round(float(F[gi, 0]), 2),
                "green_ratio":  round(float(-F[gi, 1]), 4),
                "far":          round(check["far_actual"],  3),
                "bcr":          round(check["bcr_actual"],  3),
                "violation":    round(cv_sum, 4),
            })

        # [REMOVED_ZH:1] UTCI [REMOVED_ZH:4] Pareto [REMOVED_ZH:2]
        results.sort(key=lambda x: x["mean_utci"])
        return results
