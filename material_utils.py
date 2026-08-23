#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path


MATERIAL_KEYS = (
    "init_spring_Y",
    "spring_Y_min",
    "spring_Y_max",
    "bend_stiffness",
    "bend_stiffness_min",
    "bend_stiffness_max",
)


def normalize_number(value):
    return format(float(value), ".12g")


def build_material_config(
    init_spring_Y,
    spring_Y_min,
    spring_Y_max,
    bend_stiffness,
    bend_stiffness_min,
    bend_stiffness_max,
):
    return {
        "init_spring_Y": float(init_spring_Y),
        "spring_Y_min": float(spring_Y_min),
        "spring_Y_max": float(spring_Y_max),
        "bend_stiffness": float(bend_stiffness),
        "bend_stiffness_min": float(bend_stiffness_min),
        "bend_stiffness_max": float(bend_stiffness_max),
    }


def material_payload(config):
    return "|".join(f"{key}={normalize_number(config[key])}" for key in MATERIAL_KEYS)


def compute_material_hash(config, length=10):
    return hashlib.sha1(material_payload(config).encode("utf-8")).hexdigest()[:length]



def ball_query_run_tag(object_max_neighbours=None, object_radius=None):
    parts = []
    if object_max_neighbours is not None:
        parts.append(f"bqk{int(object_max_neighbours)}")
    if object_radius is not None:
        parts.append(f"r{normalize_number(object_radius)}")
    return "_".join(parts)


def parse_optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def parse_optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def resolve_ball_query_params(object_max_neighbours=None, object_radius=None):
    if object_max_neighbours is None:
        object_max_neighbours = parse_optional_int(os.environ.get("OBJECT_MAX_NEIGHBOURS"))
    if object_radius is None:
        object_radius = parse_optional_float(os.environ.get("OBJECT_RADIUS"))
    return object_max_neighbours, object_radius


def build_result_run_tag(run_tag="", object_max_neighbours=None, object_radius=None):
    object_max_neighbours, object_radius = resolve_ball_query_params(
        object_max_neighbours,
        object_radius,
    )
    explicit_bq_tag = ball_query_run_tag(object_max_neighbours, object_radius)
    parts = []
    if explicit_bq_tag:
        parts.append(explicit_bq_tag)
    if run_tag and run_tag not in parts:
        parts.append(run_tag)
    return "_".join(parts)


def append_run_tag(base_dir, run_tag):
    if not run_tag:
        return base_dir
    return f"{base_dir}_{run_tag}"

def material_dirname(material_hash):
    if not material_hash:
        return ""
    return material_hash if material_hash.startswith("mat") else f"mat{material_hash}"


def append_material_dir(base_dir, material_hash):
    if not material_hash:
        return base_dir
    return str(Path(base_dir) / material_dirname(material_hash))


def write_material_config(directory, config, material_hash=None):
    material_hash = material_hash or compute_material_hash(config)
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    output_path = directory_path / "material_config.json"
    payload = {"material_hash": material_hash}
    payload.update(config)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Material hash/config helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("hash", "write"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--init_spring_Y", type=float, required=True)
        subparser.add_argument("--spring_Y_min", type=float, required=True)
        subparser.add_argument("--spring_Y_max", type=float, required=True)
        subparser.add_argument("--bend_stiffness", type=float, required=True)
        subparser.add_argument("--bend_stiffness_min", type=float, required=True)
        subparser.add_argument("--bend_stiffness_max", type=float, required=True)
        if command == "write":
            subparser.add_argument("--dir", required=True)
            subparser.add_argument("--material_hash", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    config = build_material_config(
        args.init_spring_Y,
        args.spring_Y_min,
        args.spring_Y_max,
        args.bend_stiffness,
        args.bend_stiffness_min,
        args.bend_stiffness_max,
    )
    if args.command == "hash":
        print(compute_material_hash(config))
        return

    path = write_material_config(args.dir, config, material_hash=args.material_hash)
    print(path)


if __name__ == "__main__":
    main()
