#!/usr/bin/env python3

import argparse
import csv
import pickletools
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


INIT_RE = re.compile(
    r"Initialize the Spring-Mass System with "
    r"(?P<vertices>\d+) vertices, "
    r"(?P<springs>\d+) springs, "
    r"(?P<object_points>\d+) object points, "
    r"(?P<control_points>\d+) control points"
)

DEFAULT_LOG_NAMES = ("inference_log.log", "inv_phy_log.log")
BENCHMARK14_SCENES = (
    "rope_double_hand",
    "single_push_rope_4",
    "double_stretch_zebra",
    "single_lift_dinosor",
    "single_lift_sloth",
    "weird_package",
    "single_push_sloth",
    "double_stretch_sloth",
    "single_push_rope_1",
    "double_lift_sloth",
    "single_lift_rope",
    "single_lift_zebra",
    "double_lift_zebra",
    "single_push_rope",
)


@dataclass(frozen=True)
class SpringMassRecord:
    scene: str
    experiment: str
    log_path: Path
    vertices: int
    springs: int
    object_points: int
    control_points: int


def load_known_scenes(root: Path) -> set[str]:
    config_path = root / "data_config.csv"
    if not config_path.exists():
        return set()

    scenes: set[str] = set()
    with config_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip():
                scenes.add(row[0].strip())
    return scenes


def infer_scene_name(log_path: Path, known_scenes: set[str]) -> str:
    for parent in log_path.parents:
        if parent.name in known_scenes:
            return parent.name
    return log_path.parent.name


def find_experiment_root(log_path: Path, root: Path) -> str:
    try:
        rel_parts = log_path.relative_to(root).parts
    except ValueError:
        return ""

    for part in rel_parts:
        if part.startswith("experiments"):
            return part
    return ""


def parse_log(log_path: Path, occurrence: str) -> dict[str, int] | None:
    matches: list[dict[str, int]] = []
    with log_path.open(errors="replace") as handle:
        for line in handle:
            match = INIT_RE.search(line)
            if not match:
                continue
            values = {key: int(value) for key, value in match.groupdict().items()}
            if occurrence == "first":
                return values
            matches.append(values)

    if not matches:
        return None
    return matches[-1]


def parse_numpy_shapes_by_key(pickle_path: Path, keys: set[str]) -> dict[str, tuple[int, ...]]:
    # Read pickle opcodes only; do not unpickle, because pickle can execute code.
    if not pickle_path.exists():
        return {}

    shapes: dict[str, tuple[int, ...]] = {}
    current_key: str | None = None
    waiting_for_shape = False
    int_stack: list[int] = []

    try:
        for op, arg, _pos in pickletools.genops(pickle_path.read_bytes()):
            if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"} and arg in keys:
                current_key = str(arg)
                waiting_for_shape = False
                int_stack = []
            elif current_key and op.name == "REDUCE":
                waiting_for_shape = True
                int_stack = []
            elif current_key and waiting_for_shape:
                if op.name == "MARK":
                    int_stack = []
                elif op.name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT"}:
                    int_stack.append(int(arg))
                elif op.name.startswith("TUPLE") and int_stack:
                    tuple_len = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}.get(op.name)
                    tuple_values = tuple(int_stack[-tuple_len:]) if tuple_len else tuple(int_stack)
                    if all(value > 0 for value in tuple_values):
                        shapes[current_key] = tuple_values
                        current_key = None
                        waiting_for_shape = False
                        int_stack = []
    except ValueError:
        return shapes

    return shapes


def parse_ratio_from_experiment(experiment: str, key: str, default: float = 1.0) -> float:
    match = re.search(rf"(?:^|_){key}([0-9.]+)", experiment)
    if not match:
        return default
    return float(match.group(1).rstrip("."))


def find_data_path(root: Path, log_path: Path, scene: str) -> Path:
    with log_path.open(errors="replace") as handle:
        for line in handle:
            marker = "[DATA]: loading data from "
            if marker in line:
                raw_path = line.split(marker, 1)[1].strip()
                path = Path(raw_path)
                return path if path.is_absolute() else root / path
    return root / "data" / "different_types" / scene / "final_data.pkl"


def parse_surface_points_from_final_data(root: Path, log_path: Path, scene: str) -> dict[str, int] | None:
    data_path = find_data_path(root, log_path, scene)
    shapes = parse_numpy_shapes_by_key(data_path, {"object_points", "surface_points"})
    if "object_points" not in shapes or "surface_points" not in shapes:
        return None

    experiment = find_experiment_root(log_path, root)
    obj_ratio = parse_ratio_from_experiment(experiment, "objssr")
    surface_ratio = parse_ratio_from_experiment(experiment, "ssr")

    original_points = shapes["object_points"][1]
    surface_points = shapes["surface_points"][0]
    original_count = max(1, int(original_points * obj_ratio)) if obj_ratio < 1.0 else original_points
    surface_count = int(surface_points * surface_ratio) if surface_ratio < 1.0 else surface_points
    object_points = original_count + surface_count

    log_values = parse_log(log_path, "first") or {}
    control_points = int(log_values.get("control_points", 0))
    return {
        "vertices": object_points + control_points,
        "springs": int(log_values.get("springs", 0)),
        "object_points": object_points,
        "control_points": control_points,
    }


def resolve_experiment_roots(
    root: Path,
    experiments_glob: str,
    experiment_dirs: list[Path] | None,
) -> list[Path]:
    if experiment_dirs:
        resolved_dirs = []
        for exp_dir in experiment_dirs:
            resolved = exp_dir if exp_dir.is_absolute() else root / exp_dir
            if not resolved.is_dir():
                raise SystemExit(f"Experiment directory does not exist: {resolved}")
            resolved_dirs.append(resolved)
        return sorted(resolved_dirs)

    return sorted(path for path in root.glob(experiments_glob) if path.is_dir())


def select_logs(
    root: Path,
    experiments_glob: str,
    experiment_dirs: list[Path] | None,
    log_names: tuple[str, ...],
) -> list[Path]:
    log_paths: list[Path] = []
    for exp_root in resolve_experiment_roots(root, experiments_glob, experiment_dirs):
        scene_dirs = [path for path in exp_root.rglob("*") if path.is_dir()]
        for scene_dir in scene_dirs:
            for log_name in log_names:
                candidate = scene_dir / log_name
                if candidate.exists():
                    log_paths.append(candidate)
                    break
    return log_paths


def collect_records(
    root: Path,
    experiments_glob: str,
    experiment_dirs: list[Path] | None,
    log_names: tuple[str, ...],
    occurrence: str,
) -> list[SpringMassRecord]:
    known_scenes = load_known_scenes(root)
    records: list[SpringMassRecord] = []

    for log_path in select_logs(root, experiments_glob, experiment_dirs, log_names):
        scene = infer_scene_name(log_path, known_scenes)
        values = parse_surface_points_from_final_data(root, log_path, scene)
        if values is None:
            values = parse_log(log_path, occurrence)
        if values is None:
            continue

        records.append(
            SpringMassRecord(
                scene=scene,
                experiment=find_experiment_root(log_path, root),
                log_path=log_path,
                vertices=values["vertices"],
                springs=values["springs"],
                object_points=values["object_points"],
                control_points=values["control_points"],
            )
        )

    return records


def metric_value(record: SpringMassRecord, metric: str) -> int:
    return getattr(record, metric)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def print_summary(label: str, values: list[float]) -> None:
    stats = summarize(values)
    print(
        f"{label}: count={int(stats['count'])}, "
        f"mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
        f"min={stats['min']:.0f}, max={stats['max']:.0f}"
    )


def write_csv(path: Path, records: list[SpringMassRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scene",
                "experiment",
                "vertices",
                "springs",
                "object_points",
                "control_points",
                "log_path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.scene,
                    record.experiment,
                    record.vertices,
                    record.springs,
                    record.object_points,
                    record.control_points,
                    record.log_path,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute spring-mass point counts from PhysTwin explicit simulation logs. "
            "By default the reported point count is surface non-control points for the explicit baseline."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="PhysTwin explicit repo root.",
    )
    parser.add_argument(
        "--experiments-glob",
        default="experiments*",
        help="Top-level experiment directory glob under --root.",
    )
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        type=Path,
        help="Specific experiment directories to scan. Paths may be absolute or relative to --root.",
    )
    parser.add_argument(
        "--metric",
        choices=("vertices", "object_points", "control_points", "springs"),
        default="object_points",
        help="Metric to average. Use object_points for surface non-control points in the explicit baseline.",
    )
    parser.add_argument(
        "--occurrence",
        choices=("first", "last"),
        default="first",
        help="Which initialization line to use when a log contains multiple runs.",
    )
    parser.add_argument(
        "--all-log-files",
        action="store_true",
        help="Count both inference_log.log and inv_phy_log.log when both exist.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write per-record extracted values.",
    )
    parser.add_argument(
        "--benchmark14",
        action="store_true",
        help="Only include the 14 main benchmark scenes, excluding cloth variants and extra scenes.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Only include these scene names.",
    )
    parser.add_argument(
        "--quiet-scenes",
        action="store_true",
        help="Only print aggregate summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    log_names = DEFAULT_LOG_NAMES if not args.all_log_files else tuple(reversed(DEFAULT_LOG_NAMES))

    records = collect_records(
        root=root,
        experiments_glob=args.experiments_glob,
        experiment_dirs=args.experiment_dirs,
        log_names=log_names,
        occurrence=args.occurrence,
    )
    if not records:
        raise SystemExit(f"No spring-mass initialization logs found under {root}/{args.experiments_glob}")

    scene_filter = set(args.scenes or [])
    if args.benchmark14:
        scene_filter.update(BENCHMARK14_SCENES)
    if scene_filter:
        records = [record for record in records if record.scene in scene_filter]
        if not records:
            raise SystemExit(f"No records matched scene filter: {sorted(scene_filter)}")

    per_scene: dict[str, list[SpringMassRecord]] = defaultdict(list)
    for record in records:
        per_scene[record.scene].append(record)

    per_record_values = [float(metric_value(record, args.metric)) for record in records]
    per_scene_means = [
        mean(metric_value(record, args.metric) for record in scene_records)
        for scene_records in per_scene.values()
    ]

    print(f"Root: {root}")
    print(f"Metric: {args.metric}")
    if args.experiment_dirs:
        print("Experiment dirs: " + ", ".join(str(path) for path in args.experiment_dirs))
    else:
        print(f"Experiment glob: {args.experiments_glob}")
    if scene_filter:
        print(f"Scene filter: {len(scene_filter)} scenes")
    print_summary("Per-record average", per_record_values)
    print_summary("Per-scene average", per_scene_means)

    if not args.quiet_scenes:
        print("")
        print("Scene means:")
        for scene in sorted(per_scene):
            scene_records = per_scene[scene]
            values = [metric_value(record, args.metric) for record in scene_records]
            print(
                f"  {scene}: mean={mean(values):.2f}, "
                f"records={len(values)}, min={min(values)}, max={max(values)}"
            )

    if args.csv:
        write_csv(args.csv, records)
        print("")
        print(f"Wrote per-record CSV: {args.csv}")


if __name__ == "__main__":
    main()
