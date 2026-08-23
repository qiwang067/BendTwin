#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


BASE_PATH = "./data/different_types"


@dataclass
class SceneTiming:
    scene: str
    spring_mass_infer_elapsed_s: float | None
    spring_mass_infer_wall_s: float | None
    gaussian_wall_s: float | None
    total_wall_s: float | None
    gaussian_end_source: str
    infer_source: str
    error: str
    status: str


class ProgressBar:
    def __init__(self, total: int, width: int = 32) -> None:
        self.total = max(total, 1)
        self.width = width
        self.done = 0
        self.started_at = time.perf_counter()
        self.is_tty = sys.stderr.isatty()

    def _format_elapsed(self) -> str:
        elapsed = time.perf_counter() - self.started_at
        minutes, seconds = divmod(int(elapsed), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _render(self, label: str) -> None:
        fraction = min(self.done / self.total, 1.0)
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"[{bar}] {self.done}/{self.total} "
            f"{fraction * 100:5.1f}% elapsed={self._format_elapsed()} {label}"
        )
        if self.is_tty:
            print(f"\r{line}", end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)

    def set_current(self, label: str) -> None:
        self._render(label)

    def advance(self, label: str) -> None:
        self.done += 1
        self._render(label)

    def finish(self) -> None:
        if self.is_tty:
            print(file=sys.stderr, flush=True)


def append_material_dir(base_dir: Path, material_hash: str) -> Path:
    if not material_hash:
        return base_dir
    material_tag = material_hash if material_hash.startswith("mat") else f"mat{material_hash}"
    return base_dir / material_tag


def run_live_inference(
    project_root: Path,
    scene: str,
    surface_sample_ratio: str,
    obj_sample_ratio: str,
    material_hash: str,
    gpu: int,
) -> float:
    cmd = [
        sys.executable,
        str(project_root / "inference_warp.py"),
        "--base_path",
        BASE_PATH,
        "--case_name",
        scene,
        "--surface_sample_ratio",
        str(surface_sample_ratio),
        "--obj_sample_ratio",
        str(obj_sample_ratio),
    ]
    if material_hash:
        cmd.extend(["--material_hash", material_hash])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return run_and_time(cmd, project_root, env)


def run_live_gaussian(
    project_root: Path,
    scene: str,
    surface_sample_ratio: str,
    obj_sample_ratio: str,
    material_hash: str,
    gpu: int,
) -> float:
    cmd = [
        "bash",
        str(project_root / "gs_run.sh"),
        str(surface_sample_ratio),
        str(obj_sample_ratio),
        str(gpu),
        scene,
    ]
    env = os.environ.copy()
    if material_hash:
        env["MATERIAL_HASH"] = material_hash
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return run_and_time(cmd, project_root, env)


def trim_error_message(text: str, max_lines: int = 3) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-max_lines:])


def discover_scenes(exp_root: Path, gs_root: Path) -> list[str]:
    exp_scenes = {path.name for path in exp_root.iterdir() if path.is_dir()} if exp_root.exists() else set()
    gs_scenes = {path.name for path in gs_root.iterdir() if path.is_dir()} if gs_root.exists() else set()
    return sorted(exp_scenes | gs_scenes)


def list_child_dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(child.name for child in path.iterdir() if child.is_dir())


def format_missing_scenes_error(
    exp_root: Path,
    gs_root: Path,
    exp_base: Path,
    gs_base: Path,
) -> str:
    exp_candidates = list_child_dirs(exp_base)
    gs_candidates = list_child_dirs(gs_base)
    return (
        "No scenes found for the requested timing roots.\n"
        f"  experiments path: {exp_root}\n"
        f"  gaussian path: {gs_root}\n"
        f"  available experiment dirs under {exp_base}: {exp_candidates or 'none'}\n"
        f"  available gaussian dirs under {gs_base}: {gs_candidates or 'none'}\n"
        "Pass --scenes explicitly or use an existing --material-hash."
    )


def run_and_time(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
) -> float:
    t0 = time.perf_counter()
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = trim_error_message(exc.stdout or str(exc), max_lines=8)
        raise RuntimeError(f"command failed: {' '.join(cmd)} | {detail}") from exc
    return time.perf_counter() - t0


def build_scene_timing(
    scene: str,
    project_root: Path,
    surface_sample_ratio: str,
    obj_sample_ratio: str,
    material_hash: str,
    gpu: int,
    progress: ProgressBar | None = None,
) -> SceneTiming:
    infer_source = "live_inference"
    error = ""
    if progress:
        progress.set_current(f"{scene}: inference")
    try:
        infer_wall_s = run_live_inference(
            project_root,
            scene,
            surface_sample_ratio,
            obj_sample_ratio,
            material_hash,
            gpu,
        )
        infer_elapsed_s = infer_wall_s
    except Exception as exc:
        infer_elapsed_s = None
        infer_wall_s = None
        infer_source = "live_inference_failed"
        error = trim_error_message(str(exc))
    finally:
        if progress:
            progress.advance(f"{scene}: inference done")

    if progress:
        progress.set_current(f"{scene}: gaussian")
    try:
        gaussian_wall_s = run_live_gaussian(
            project_root,
            scene,
            surface_sample_ratio,
            obj_sample_ratio,
            material_hash,
            gpu,
        )
        gaussian_end_source = "live_gaussian"
    except Exception as exc:
        gaussian_wall_s = None
        gaussian_end_source = "live_gaussian_failed"
        error = error or trim_error_message(str(exc))
    finally:
        if progress:
            progress.advance(f"{scene}: gaussian done")

    status_parts = []
    if infer_elapsed_s is None:
        status_parts.append("missing_inference_elapsed")
    if infer_wall_s is None:
        status_parts.append("missing_inference_wall")
    if gaussian_wall_s is None:
        status_parts.append(gaussian_end_source)

    total_wall_s = None
    if infer_wall_s is not None and gaussian_wall_s is not None:
        total_wall_s = infer_wall_s + gaussian_wall_s

    return SceneTiming(
        scene=scene,
        spring_mass_infer_elapsed_s=infer_elapsed_s,
        spring_mass_infer_wall_s=infer_wall_s,
        gaussian_wall_s=gaussian_wall_s,
        total_wall_s=total_wall_s,
        gaussian_end_source=gaussian_end_source,
        infer_source=infer_source,
        error=error,
        status="ok" if not status_parts else ";".join(status_parts),
    )


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_csv(output_path: Path, rows: list[SceneTiming]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
                "spring_mass_infer_elapsed_s",
                "spring_mass_infer_wall_s",
                "gaussian_wall_s",
                "spring_mass_plus_gaussian_wall_s",
                "infer_source",
                "gaussian_end_source",
                "error",
                "status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.scene,
                    format_float(row.spring_mass_infer_elapsed_s),
                    format_float(row.spring_mass_infer_wall_s),
                    format_float(row.gaussian_wall_s),
                    format_float(row.total_wall_s),
                    row.infer_source,
                    row.gaussian_end_source,
                    row.error,
                    row.status,
                ]
            )


def print_summary(rows: list[SceneTiming]) -> None:
    header = (
        f"{'scene':<24} | {'infer_elapsed_s':>14} | {'infer_wall_s':>12} | "
        f"{'gaussian_wall_s':>15} | {'total_wall_s':>12} | {'status':<32}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.scene:<24} | "
            f"{format_float(row.spring_mass_infer_elapsed_s):>14} | "
            f"{format_float(row.spring_mass_infer_wall_s):>12} | "
            f"{format_float(row.gaussian_wall_s):>15} | "
            f"{format_float(row.total_wall_s):>12} | "
            f"{row.status:<32}"
        )
    failed_rows = [row for row in rows if row.error]
    if failed_rows:
        print("\nErrors:")
        for row in failed_rows:
            print(f"- {row.scene}: {row.error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report spring-mass inference time and Gaussian wall time per scene."
    )
    parser.add_argument("--surface-sample-ratio", type=str, required=True)
    parser.add_argument("--obj-sample-ratio", type=str, required=True)
    parser.add_argument("--material-hash", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    exp_base = project_root / f"experiments_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}"
    gs_base = project_root / f"gaussian_output_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}"
    exp_root = append_material_dir(exp_base, args.material_hash)
    gs_root = append_material_dir(gs_base, args.material_hash)

    scenes = args.scenes or discover_scenes(exp_root, gs_root)
    if not scenes:
        raise ValueError(format_missing_scenes_error(exp_root, gs_root, exp_base, gs_base))
    progress = ProgressBar(total=len(scenes) * 2)
    rows = []
    for scene in scenes:
        rows.append(build_scene_timing(
            scene,
            project_root,
            args.surface_sample_ratio,
            args.obj_sample_ratio,
            args.material_hash,
            args.gpu,
            progress,
        ))
    progress.finish()

    output_path = args.output
    if output_path is None:
        suffix = f"ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}"
        if args.material_hash:
            material_tag = args.material_hash if args.material_hash.startswith("mat") else f"mat{args.material_hash}"
            suffix = f"{suffix}_{material_tag}"
        output_path = project_root / f"timing_report_{suffix}.csv"

    write_csv(output_path, rows)
    print_summary(rows)
    print(f"\nTiming CSV written to: {output_path}")

    live_failures = [
        row for row in rows
        if row.infer_source == "live_inference_failed" or row.gaussian_end_source == "live_gaussian_failed"
    ]
    if live_failures:
        raise SystemExit(f"Live timing failed for {len(live_failures)} scene(s).")


if __name__ == "__main__":
    main()
