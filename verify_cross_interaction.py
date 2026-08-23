#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

from material_utils import build_result_run_tag


def fmt_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pair_from_tag(tag):
    if not tag.startswith("src-") or "__tgt-" not in tag:
        return None, None
    source, target = tag.removeprefix("src-").split("__tgt-", 1)
    return source, target


def build_prefix(kind, ssr, objssr, run_tag, material_hash):
    run_suffix = f"_{run_tag}" if run_tag else ""
    material_suffix = f"_mat{material_hash}" if material_hash else ""
    return f"{kind}_ssr{ssr}_objssr{objssr}{run_suffix}{material_suffix}"


def collect_cross_chamfer(results_dir, prefix):
    out = {}
    for path in sorted(results_dir.glob(f"{prefix}_src-*.csv")):
        for row in read_csv_rows(path):
            pair_tag = row.get("Case Name", "")
            source, target = pair_from_tag(pair_tag)
            if not source:
                continue
            out[pair_tag] = {
                "source": source,
                "target": target,
                "cross_chamfer_train": parse_float(row.get("Train Chamfer Error")),
                "cross_chamfer_test": parse_float(row.get("Test Chamfer Error")),
                "cross_chamfer_file": str(path),
            }
    return out


def collect_cross_track(results_dir, prefix):
    out = {}
    for path in sorted(results_dir.glob(f"{prefix}_src-*.csv")):
        for row in read_csv_rows(path):
            pair_tag = row.get("Case Name", "")
            source, target = pair_from_tag(pair_tag)
            if not source:
                continue
            out[pair_tag] = {
                "cross_track_train": parse_float(row.get("Train Track Error")),
                "cross_track_test": parse_float(row.get("Test Track Error")),
                "cross_track_file": str(path),
            }
    return out


def collect_self_chamfer(results_dir, prefix):
    by_scene = {}
    for path in sorted(results_dir.glob(f"{prefix}_*.csv")):
        if "_src-" in path.name:
            continue
        for row in read_csv_rows(path):
            scene = row.get("Case Name", "")
            if scene.startswith("src-"):
                continue
            by_scene[scene] = {
                "self_chamfer_train": parse_float(row.get("Train Chamfer Error")),
                "self_chamfer_test": parse_float(row.get("Test Chamfer Error")),
                "self_chamfer_file": str(path),
            }
    return by_scene


def collect_self_track(results_dir, prefix):
    by_scene = {}
    for path in sorted(results_dir.glob(f"{prefix}_*.csv")):
        if "_src-" in path.name:
            continue
        for row in read_csv_rows(path):
            scene = row.get("Case Name", "")
            if scene.startswith("src-"):
                continue
            by_scene[scene] = {
                "self_track_train": parse_float(row.get("Train Track Error")),
                "self_track_test": parse_float(row.get("Test Track Error")),
                "self_track_file": str(path),
            }
    return by_scene


def ratio(cross_value, self_value):
    if self_value is None or math.isnan(self_value) or self_value == 0:
        return float("nan")
    if cross_value is None or math.isnan(cross_value):
        return float("nan")
    return cross_value / self_value


def mean(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Summarize cross-interaction generalization vs self-fit.")
    parser.add_argument("--ssr", default="1.0")
    parser.add_argument("--objssr", default="1.0")
    parser.add_argument("--material_hash", required=True)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--object_max_neighbours", type=int, default=None)
    parser.add_argument("--object_radius", type=float, default=None)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--output_txt", default="")
    args = parser.parse_args()

    run_tag = args.run_tag or build_result_run_tag(
        "",
        args.object_max_neighbours,
        args.object_radius,
    )
    results_dir = Path(args.results_dir)
    chamfer_prefix = build_prefix("final_results", args.ssr, args.objssr, run_tag, args.material_hash)
    track_prefix = build_prefix("final_track", args.ssr, args.objssr, run_tag, args.material_hash)

    cross = collect_cross_chamfer(results_dir, chamfer_prefix)
    cross_track = collect_cross_track(results_dir, track_prefix)
    self_chamfer = collect_self_chamfer(results_dir, chamfer_prefix)
    self_track = collect_self_track(results_dir, track_prefix)

    if not cross:
        raise RuntimeError(f"No cross chamfer files found with prefix: {results_dir}/{chamfer_prefix}_src-*.csv")

    rows = []
    for pair_tag, item in sorted(cross.items()):
        target = item["target"]
        item.update(cross_track.get(pair_tag, {}))
        item.update(self_chamfer.get(target, {}))
        item.update(self_track.get(target, {}))
        item["chamfer_test_ratio_cross_over_self"] = ratio(
            item.get("cross_chamfer_test"), item.get("self_chamfer_test")
        )
        item["track_test_ratio_cross_over_self"] = ratio(
            item.get("cross_track_test"), item.get("self_track_test")
        )
        rows.append(item)

    output_csv = Path(args.output_csv or results_dir / f"cross_interaction_verify_ssr{args.ssr}_objssr{args.objssr}{('_' + run_tag) if run_tag else ''}_mat{args.material_hash}.csv")
    output_txt = Path(args.output_txt or results_dir / f"cross_interaction_verify_ssr{args.ssr}_objssr{args.objssr}{('_' + run_tag) if run_tag else ''}_mat{args.material_hash}.txt")

    fieldnames = [
        "source",
        "target",
        "cross_chamfer_test",
        "self_chamfer_test",
        "chamfer_test_ratio_cross_over_self",
        "cross_track_test",
        "self_track_test",
        "track_test_ratio_cross_over_self",
        "cross_chamfer_train",
        "self_chamfer_train",
        "cross_track_train",
        "self_track_train",
        "cross_chamfer_file",
        "self_chamfer_file",
        "cross_track_file",
        "self_track_file",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    chamfer_ratios = [r["chamfer_test_ratio_cross_over_self"] for r in rows]
    track_ratios = [r["track_test_ratio_cross_over_self"] for r in rows]
    cross_chamfers = [r.get("cross_chamfer_test", float("nan")) for r in rows]
    self_chamfers = [r.get("self_chamfer_test", float("nan")) for r in rows]
    cross_tracks = [r.get("cross_track_test", float("nan")) for r in rows]
    self_tracks = [r.get("self_track_test", float("nan")) for r in rows]

    lines = []
    lines.append("Cross-Interaction Verification")
    lines.append("================================")
    lines.append(f"ssr={args.ssr}, objssr={args.objssr}, run_tag={run_tag or '(none)'}, material=mat{args.material_hash}")
    lines.append(f"pairs={len(rows)}")
    lines.append("")
    lines.append("Overall test metrics (lower is better):")
    lines.append(f"  Cross Chamfer mean: {fmt_float(mean(cross_chamfers))}")
    lines.append(f"  Self  Chamfer mean: {fmt_float(mean(self_chamfers))}")
    lines.append(f"  Cross/Self Chamfer ratio mean: {fmt_float(mean(chamfer_ratios))}")
    lines.append(f"  Cross Track mean:   {fmt_float(mean(cross_tracks))}")
    lines.append(f"  Self  Track mean:   {fmt_float(mean(self_tracks))}")
    lines.append(f"  Cross/Self Track ratio mean:   {fmt_float(mean(track_ratios))}")
    lines.append("")
    lines.append("Per pair:")
    lines.append("source,target,cross_chamfer,self_chamfer,chamfer_ratio,cross_track,self_track,track_ratio")
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["source"],
                    row["target"],
                    fmt_float(row.get("cross_chamfer_test")),
                    fmt_float(row.get("self_chamfer_test")),
                    fmt_float(row.get("chamfer_test_ratio_cross_over_self")),
                    fmt_float(row.get("cross_track_test")),
                    fmt_float(row.get("self_track_test")),
                    fmt_float(row.get("track_test_ratio_cross_over_self")),
                ]
            )
        )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote TXT: {output_txt}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
