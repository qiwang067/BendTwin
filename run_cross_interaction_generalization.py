#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from material_utils import append_material_dir, append_run_tag, ball_query_run_tag


BASE_PATH = Path("./data/different_types")
GS_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"


def read_data_config(path):
    groups = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            scene = row[0].strip()
            object_name = row[1].strip()
            enabled = len(row) < 3 or row[2].strip().lower() == "true"
            if scene and object_name and enabled:
                groups.setdefault(object_name, []).append(scene)
    return groups


def parse_pairs(args, groups):
    if args.pairs:
        pairs = []
        for item in args.pairs:
            if ":" not in item:
                raise ValueError(f"pair must be source:target, got {item}")
            source, target = item.split(":", 1)
            pairs.append((source, target))
        return pairs

    if args.object:
        if args.object not in groups:
            raise ValueError(f"Unknown object {args.object}. Available: {', '.join(sorted(groups))}")
        selected_groups = {args.object: groups[args.object]}
    else:
        selected_groups = groups

    pairs = []
    for scenes in selected_groups.values():
        for source in scenes:
            for target in scenes:
                if source != target:
                    pairs.append((source, target))
    return pairs


def compact_pair_tag(source, target):
    return f"src-{source}__tgt-{target}"


def run(cmd, dry_run=False):
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def copy_target_training_artifacts(target_exp_dir, output_scene_dir, force=False):
    target_train_dir = target_exp_dir / "train"
    require_file(target_train_dir, "target train directory")
    output_train_dir = output_scene_dir / "train"
    if output_train_dir.exists():
        if force:
            shutil.rmtree(output_train_dir)
        else:
            return
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target_train_dir, output_train_dir)


def copy_json_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_scene_tag(pairs):
    if len(pairs) == 1:
        return compact_pair_tag(*pairs[0])
    return f"cross_interaction_{len(pairs)}pairs"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether optimized physical parameters generalize across unseen interactions of the same object."
    )
    parser.add_argument("--ssr", type=float, default=1.0)
    parser.add_argument("--objssr", type=float, default=1.0)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--material_hash", required=True)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--object_max_neighbours", type=int, default=None)
    parser.add_argument("--object_radius", type=float, default=None)
    parser.add_argument("--object", default="", help="Only generate pairs within this object category from data_config.csv")
    parser.add_argument("--pairs", nargs="*", default=None, help="Explicit source:target pairs")
    parser.add_argument("--data_config", default="data_config.csv")
    parser.add_argument("--output_root", default="", help="Default: generalization_ssr*_objssr*[_run_tag]")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    run_tag = args.run_tag or ball_query_run_tag(args.object_max_neighbours, args.object_radius)
    source_experiment_root = Path(append_run_tag(f"experiments_ssr{args.ssr}_objssr{args.objssr}", run_tag))
    source_optimization_root = Path(append_run_tag(f"experiments_optimization_ssr{args.ssr}_objssr{args.objssr}", run_tag))
    output_root = Path(args.output_root or append_run_tag(f"generalization_ssr{args.ssr}_objssr{args.objssr}", run_tag))

    source_exp_base = Path(append_material_dir(str(source_experiment_root), args.material_hash))
    source_opt_base = Path(append_material_dir(str(source_optimization_root), args.material_hash))
    output_base = Path(append_material_dir(str(output_root), args.material_hash))

    groups = read_data_config(args.data_config)
    pairs = parse_pairs(args, groups)
    if not pairs:
        raise RuntimeError("No source->target pairs selected")

    output_base.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ssr": args.ssr,
        "objssr": args.objssr,
        "material_hash": args.material_hash,
        "run_tag": run_tag,
        "pairs": [{"source": s, "target": t} for s, t in pairs],
        "source_experiment_root": str(source_experiment_root),
        "source_optimization_root": str(source_optimization_root),
        "output_root": str(output_root),
    }
    (output_base / "generalization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    eval_dirs = []
    for source, target in pairs:
        pair_tag = compact_pair_tag(source, target)
        output_scene_dir = output_base / pair_tag
        target_exp_dir = source_exp_base / target
        source_optimal_path = source_opt_base / source / "optimal_params.pkl"
        require_file(source_optimal_path, "source optimal_params.pkl")
        require_file(target_exp_dir / "train", "target trained checkpoint directory")

        copy_target_training_artifacts(target_exp_dir, output_scene_dir, force=args.force)
        copy_json_if_exists(source_opt_base / "material_config.json", output_base / "material_config.json")

        if not args.skip_inference:
            cmd = [
                sys.executable,
                "inference_warp.py",
                "--base_path",
                str(BASE_PATH),
                "--case_name",
                target,
                "--surface_sample_ratio",
                str(args.ssr),
                "--obj_sample_ratio",
                str(args.objssr),
                "--material_hash",
                args.material_hash,
                "--experiment_root_override",
                str(output_root),
                "--optimal_params_path",
                str(source_optimal_path),
                "--checkpoint_load_mode",
                "sampling_indices",
                "--output_case_name",
                pair_tag,
            ]
            if run_tag:
                cmd.extend(["--run_tag", run_tag])
            if args.object_radius is not None:
                cmd.extend(["--object_radius", str(args.object_radius)])
            if args.object_max_neighbours is not None:
                cmd.extend(["--object_max_neighbours", str(args.object_max_neighbours)])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
            print("+ " + " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True, env=env)

        eval_dirs.append((pair_tag, output_scene_dir))

    if not args.skip_eval:
        rows = []
        scene_tag = build_scene_tag(pairs)
        for pair_tag, prediction_dir in eval_dirs:
            source, target = pair_tag.removeprefix("src-").split("__tgt-", 1)
            chamfer_cmd = [
                sys.executable,
                "evaluate_chamfer.py",
                "--surface_sample_ratio",
                str(args.ssr),
                "--obj_sample_ratio",
                str(args.objssr),
                "--material_hash",
                args.material_hash,
                "--prediction_dir",
                str(prediction_dir.parent),
                "--scenes",
                pair_tag,
                "--scene_tag",
                pair_tag,
                "--scene_alias",
                f"{pair_tag}={target}",
            ]
            track_cmd = [
                sys.executable,
                "evaluate_track.py",
                "--surface_sample_ratio",
                str(args.ssr),
                "--obj_sample_ratio",
                str(args.objssr),
                "--material_hash",
                args.material_hash,
                "--prediction_dir",
                str(prediction_dir.parent),
                "--scenes",
                pair_tag,
                "--scene_tag",
                pair_tag,
                "--scene_alias",
                f"{pair_tag}={target}",
            ]
            if run_tag:
                chamfer_cmd.extend(["--run_tag", run_tag])
                track_cmd.extend(["--run_tag", run_tag])
            if args.object_radius is not None:
                chamfer_cmd.extend(["--object_radius", str(args.object_radius)])
                track_cmd.extend(["--object_radius", str(args.object_radius)])
            if args.object_max_neighbours is not None:
                chamfer_cmd.extend(["--object_max_neighbours", str(args.object_max_neighbours)])
                track_cmd.extend(["--object_max_neighbours", str(args.object_max_neighbours)])
            run(chamfer_cmd, dry_run=args.dry_run)
            run(track_cmd, dry_run=args.dry_run)
            rows.append([source, target, pair_tag, str(prediction_dir / "inference.pkl")])

        summary_csv = output_base / f"cross_interaction_pairs_{scene_tag}.csv"
        if not args.dry_run:
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["source_scene", "target_scene", "pair_tag", "inference_path"])
                writer.writerows(rows)
            print(f"Wrote pair summary: {summary_csv}")


if __name__ == "__main__":
    main()
