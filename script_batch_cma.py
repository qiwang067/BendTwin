#!/usr/bin/env python3
"""
Batch test script that runs only the optimize stage and compares it with origin_opt.
The best result from each attempt is saved to best_optimize/; the search for a case stops once it outperforms origin.

Usage: python script_batch_cma.py
"""

import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# --- Configuration ---
BASE_PATH = Path("./data/different_types")
ORIGIN_OPT_DIR = Path("origin_opt")
BEST_OUTPUT_DIR = Path("best_optimize")
OPTIMIZE_OUTPUT_DIR = Path("experiments_optimization")
MAX_OPTIMIZE_TRIES = 10
MAX_LOOP_TRIES = 2
HERE = Path(__file__).parent


# --- Utilities ---

def parse_optimal_error(log_path):
    """Parse the last Optimal error from optimize_cma_log.log."""
    if not Path(log_path).exists():
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    matches = re.findall(r"Optimal error:\s*([\d.eE+-]+)", content)
    return float(matches[-1]) if matches else None


def get_optimal_error(case_name, source):
    """
    Get the optimal error from the specified source.
    source: "origin" | "current" | "best"
    """
    dirs = {
        "origin": HERE / ORIGIN_OPT_DIR,
        "current": HERE / OPTIMIZE_OUTPUT_DIR,
        "best": HERE / BEST_OUTPUT_DIR,
    }
    log_path = dirs[source] / case_name / "optimize_cma_log.log"
    return parse_optimal_error(log_path)


def clear_opt_dir(case_name):
    """Clear experiments_optimization/{case} before rerunning."""
    opt_dir = HERE / OPTIMIZE_OUTPUT_DIR / case_name
    if opt_dir.exists():
        shutil.rmtree(opt_dir)


def save_to_best(case_name):
    """Copy results from experiments_optimization to best_optimize."""
    best_dir = HERE / BEST_OUTPUT_DIR / case_name
    opt_src = HERE / OPTIMIZE_OUTPUT_DIR / case_name
    if not opt_src.exists():
        return

    best_dir.mkdir(parents=True, exist_ok=True)
    for f in ["optimal_params.pkl", "optimize_cma_log.log"]:
        src = opt_src / f
        if src.exists():
            shutil.copy2(src, best_dir / f)
    opt_sub = opt_src / "optimizeCMA"
    if opt_sub.exists():
        dst_sub = best_dir / "optimizeCMA"
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.copytree(opt_sub, dst_sub)


def run_optimize(case_name, base_path, train_frame, gpu_id):
    """Run one CMA optimization attempt."""
    cmd = [
        sys.executable, "optimize_cma.py",
        "--base_path", str(base_path),
        "--case_name", case_name,
        "--train_frame", str(train_frame),
        "--max_iter", str(MAX_OPTIMIZE_TRIES),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ret = subprocess.run(cmd, cwd=str(HERE), env=env)
    return ret.returncode == 0


# --- Main logic ---

def should_skip_case(case_name):
    """
    Skip if best_optimize already contains a result that outperforms origin.
    Return (should_skip, reason).
    """
    best_error = get_optimal_error(case_name, "best")
    origin_error = get_optimal_error(case_name, "origin")
    best_log = HERE / BEST_OUTPUT_DIR / case_name / "optimize_cma_log.log"

    if not best_log.exists() or best_error is None or origin_error is None:
        return False, None

    if best_error < origin_error:
        return True, f"Already in {BEST_OUTPUT_DIR} and better than origin (best={best_error:.6e} < origin={origin_error:.6e})"
    return False, None


def run_one_attempt(case_name, base_path, train_frame, gpu_id, attempt, total):
    """
    Run one optimization attempt and return (success, current_error) or (False, None).
    """
    print(f"\nOptimize: {case_name} (attempt {attempt}/{total})")
    clear_opt_dir(case_name)

    if not run_optimize(case_name, base_path, train_frame, gpu_id):
        print("  [Failed] Optimization stage failed")
        return False, None

    current_error = get_optimal_error(case_name, "current")
    if current_error is None:
        print("  [Warning] Optimal error not found")
        return False, None

    return True, current_error


def process_case(case_name, base_path, gpu_id=0):
    """Optimize one case repeatedly, retain the best result, and stop after outperforming origin."""
    split_path = base_path / case_name / "split.json"
    if not split_path.exists():
        print(f"  [Skip] {case_name}: split.json is missing")
        return False

    skip, reason = should_skip_case(case_name)
    if skip:
        print(f"\n[Skip] {case_name}: {reason}")
        return True

    with open(split_path, "r") as f:
        train_frame = json.load(f)["train"][1]

    origin_error = get_optimal_error(case_name, "origin")
    if origin_error is None:
        print(f"\n[Warning] {case_name} is absent from origin_opt; skipping")
        return True

    best_error = float("inf")

    for attempt in range(1, MAX_LOOP_TRIES + 1):
        ok, current_error = run_one_attempt(
            case_name, base_path, train_frame, gpu_id, attempt, MAX_LOOP_TRIES
        )
        if not ok or current_error is None:
            continue

        print(f"  origin: {origin_error:.6e}, current: {current_error:.6e}")

        if current_error < best_error:
            best_error = current_error
            save_to_best(case_name)
            print(f"  → New best result; updated {BEST_OUTPUT_DIR}/{case_name}")

        if current_error < origin_error:
            print("  ✓ Better than origin_opt; stopping search")
            break

        print("  ✗ Did not outperform origin_opt; retrying...")
    else:
        print(f"  [Done] Completed {MAX_LOOP_TRIES} attempts; best result saved to {BEST_OUTPUT_DIR}")

    clear_opt_dir(case_name)
    return True


def get_case_list():
    """Return cases present in both origin_opt and the dataset."""
    origin_path = HERE / ORIGIN_OPT_DIR
    origin_cases = {
        d.name for d in origin_path.iterdir()
        if d.is_dir() and (d / "optimize_cma_log.log").exists()
    } if origin_path.exists() else set()

    data_path = Path(BASE_PATH)
    if not data_path.exists():
        print(f"Error: {BASE_PATH} does not exist")
        sys.exit(1)
    data_cases = {d.name for d in data_path.iterdir() if d.is_dir()}

    return sorted(origin_cases & data_cases)


def get_n_gpus():
    """Return the number of available GPUs."""
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def _worker_run_case(queue, gpu_id):
    """Bind a worker to one GPU and dynamically pull cases from the queue to avoid idle GPUs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        case_name = queue.get()
        if case_name is None:
            break
        try:
            process_case(case_name, BASE_PATH, gpu_id)
        except Exception as e:
            print(f"  [Exception] {case_name}: {e}")


def main():
    cases = get_case_list()
    if not cases:
        print("No cases to process")
        return

    n_gpus = get_n_gpus()
    print(f"Processing {len(cases)} cases on {n_gpus} GPUs with dynamic scheduling: {cases}")

    queue = multiprocessing.Queue()
    for c in cases:
        queue.put(c)
    for _ in range(n_gpus):
        queue.put(None)

    workers = [
        multiprocessing.Process(target=_worker_run_case, args=(queue, i))
        for i in range(n_gpus)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    best_dir = HERE / BEST_OUTPUT_DIR
    n = len(list(best_dir.iterdir())) if best_dir.exists() else 0
    print(f"\nDone. Best results for {n} cases were saved to {BEST_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
