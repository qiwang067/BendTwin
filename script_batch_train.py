#!/usr/bin/env python3
"""
Batch training script that runs only the train stage and compares it with origin_train.
The best result from each attempt is saved to best_train/; the search for a case stops once it outperforms origin.

Usage: python script_batch_train.py
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
ORIGIN_TRAIN_DIR = Path("origin_train")
BEST_OUTPUT_DIR = Path("best_train")
TRAIN_OUTPUT_DIR = Path("experiments")
OPTIMIZE_SOURCE = Path("best_optimize")  # Prefer optimal_params from here; otherwise use experiments_optimization.
MAX_TRAIN_TRIES = 5
HERE = Path(__file__).parent


# --- Utilities ---

def parse_best_loss(log_path):
    """Parse the last 'Latest best model saved' loss from inv_phy_log.log."""
    if not Path(log_path).exists():
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    matches = re.findall(r"Latest best model saved: epoch \d+ with loss ([\d.eE+-]+)", content)
    return float(matches[-1]) if matches else None


def get_best_loss(case_name, source):
    """
    Get the best loss from the specified source.
    source: "origin" | "current" | "best"
    """
    dirs = {
        "origin": HERE / ORIGIN_TRAIN_DIR,
        "current": HERE / TRAIN_OUTPUT_DIR,
        "best": HERE / BEST_OUTPUT_DIR,
    }
    log_path = dirs[source] / case_name / "inv_phy_log.log"
    return parse_best_loss(log_path)


def ensure_optimal_params(case_name):
    """
    Ensure experiments_optimization/{case} contains optimal_params.pkl.
    Copy it from best_optimize when available for use by train_warp.
    """
    best_src = HERE / OPTIMIZE_SOURCE / case_name / "optimal_params.pkl"
    opt_dst_dir = HERE / "experiments_optimization" / case_name
    opt_dst = opt_dst_dir / "optimal_params.pkl"
    if best_src.exists() and (not opt_dst.exists() or best_src.stat().st_mtime > opt_dst.stat().st_mtime):
        opt_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, opt_dst)


def clear_train_dir(case_name):
    """Clear experiments/{case} before rerunning."""
    train_dir = HERE / TRAIN_OUTPUT_DIR / case_name
    if train_dir.exists():
        shutil.rmtree(train_dir)


def save_to_best(case_name):
    """Copy results from experiments to best_train."""
    best_dir = HERE / BEST_OUTPUT_DIR / case_name
    train_src = HERE / TRAIN_OUTPUT_DIR / case_name
    if not train_src.exists():
        return

    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(train_src, best_dir)


def run_train(case_name, base_path, train_frame, gpu_id, seed=None):
    """Run one training attempt."""
    cmd = [
        sys.executable, "train_warp.py",
        "--base_path", str(base_path),
        "--case_name", case_name,
        "--train_frame", str(train_frame),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ret = subprocess.run(cmd, cwd=str(HERE), env=env)
    return ret.returncode == 0


# --- Main logic ---

def should_skip_case(case_name):
    """
    Skip if best_train already contains a result that outperforms origin.
    Return (should_skip, reason).
    """
    best_loss = get_best_loss(case_name, "best")
    origin_loss = get_best_loss(case_name, "origin")
    best_log = HERE / BEST_OUTPUT_DIR / case_name / "inv_phy_log.log"

    if not best_log.exists() or best_loss is None or origin_loss is None:
        return False, None

    if best_loss < origin_loss:
        return True, f"Already in {BEST_OUTPUT_DIR} and better than origin (best={best_loss:.6e} < origin={origin_loss:.6e})"
    return False, None


def run_one_attempt(case_name, base_path, train_frame, gpu_id, attempt, total, seed):
    """
    Run one training attempt and return (success, current_loss) or (False, None).
    """
    print(f"\nTrain: {case_name} (attempt {attempt}/{total}, seed={seed})")
    clear_train_dir(case_name)
    ensure_optimal_params(case_name)

    if not run_train(case_name, base_path, train_frame, gpu_id, seed):
        print("  [Failed] Training stage failed")
        return False, None

    current_loss = get_best_loss(case_name, "current")
    if current_loss is None:
        print("  [Warning] Best loss not found")
        return False, None

    return True, current_loss


def process_case(case_name, base_path, gpu_id=0):
    """Train one case repeatedly, retain the best result, and stop after outperforming origin."""
    split_path = base_path / case_name / "split.json"
    if not split_path.exists():
        print(f"  [Skip] {case_name}: split.json is missing")
        return False

    ensure_optimal_params(case_name)
    opt_path = HERE / "experiments_optimization" / case_name / "optimal_params.pkl"
    if not opt_path.exists():
        print(f"  [Skip] {case_name}: optimal_params.pkl is missing; run script_batch_cma or optimize first")
        return False

    skip, reason = should_skip_case(case_name)
    if skip:
        print(f"\n[Skip] {case_name}: {reason}")
        return True

    with open(split_path, "r") as f:
        train_frame = json.load(f)["train"][1]

    origin_loss = get_best_loss(case_name, "origin")
    if origin_loss is None:
        print(f"\n[Warning] {case_name} is absent from origin_train; skipping")
        return True

    best_loss = float("inf")
    base_seed = 42

    for attempt in range(1, MAX_TRAIN_TRIES + 1):
        seed = base_seed + attempt - 1
        ok, current_loss = run_one_attempt(
            case_name, base_path, train_frame, gpu_id, attempt, MAX_TRAIN_TRIES, seed
        )
        if not ok or current_loss is None:
            continue

        print(f"  origin: {origin_loss:.6e}, current: {current_loss:.6e}")

        if current_loss < best_loss:
            best_loss = current_loss
            save_to_best(case_name)
            print(f"  → New best result; updated {BEST_OUTPUT_DIR}/{case_name}")

        if current_loss < origin_loss:
            print("  ✓ Better than origin_train; stopping search")
            break

        print("  ✗ Did not outperform origin_train; retrying...")
    else:
        print(f"  [Done] Completed {MAX_TRAIN_TRIES} attempts; best result saved to {BEST_OUTPUT_DIR}")

    return True


def get_case_list():
    """Return cases present in both origin_train and the dataset."""
    origin_path = HERE / ORIGIN_TRAIN_DIR
    origin_cases = {
        d.name for d in origin_path.iterdir()
        if d.is_dir() and (d / "inv_phy_log.log").exists()
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
    n = len([d for d in best_dir.iterdir() if d.is_dir()]) if best_dir.exists() else 0
    print(f"\nDone. Best results for {n} cases were saved to {BEST_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
