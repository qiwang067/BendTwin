import os
import pickle
import glob
import csv
import json
import numpy as np
import torch
import hashlib
from argparse import ArgumentParser
from scipy.spatial import KDTree
from material_utils import (
    append_material_dir,
    append_run_tag,
    build_result_run_tag,
)

parser = ArgumentParser()
parser.add_argument("--surface_sample_ratio", type=float, default=0.3)
parser.add_argument("--obj_sample_ratio", type=float, default=1.0)
parser.add_argument("--material_hash", type=str, default="")
parser.add_argument("--run_tag", type=str, default="")
parser.add_argument("--object_max_neighbours", type=int, default=None)
parser.add_argument("--object_radius", type=float, default=None)
parser.add_argument("--scenes", nargs="*", default=None, help="Scenes to process; process all scenes if omitted")
parser.add_argument("--scene_tag", type=str, default="")
parser.add_argument("--prediction_dir", type=str, default="", help="Override prediction directory")
parser.add_argument("--scene_alias", action="append", default=[], help="Map prediction scene name to data scene name: pred=data")
args = parser.parse_args()
scene_alias = {}
for item in args.scene_alias:
    if "=" not in item:
        raise ValueError(f"--scene_alias must be pred=data, got {item}")
    pred, data = item.split("=", 1)
    scene_alias[pred] = data

base_path = "./data/different_types"
prediction_path = args.prediction_dir or append_material_dir(
    append_run_tag(f"experiments_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}", args.run_tag),
    args.material_hash or None,
)
def compact_scene_tag(scene_tag, scenes, max_len=80):
    if not scene_tag:
        return ""
    if len(scene_tag) <= max_len:
        return scene_tag

    digest_source = "__".join(sorted(scenes)) if scenes else scene_tag
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    if scenes:
        prefix = scenes[0] if len(scenes) == 1 else f"{scenes[0]}__plus{len(scenes) - 1}"
    else:
        prefix = scene_tag
    prefix = prefix[: max_len - len(digest) - 1].rstrip("_")
    return f"{prefix}_{digest}"

scene_tag = compact_scene_tag(args.scene_tag, args.scenes)
scene_suffix = f"_{scene_tag}" if scene_tag else ""
material_suffix = f"_mat{args.material_hash}" if args.material_hash else ""
result_run_tag = build_result_run_tag(
    args.run_tag,
    args.object_max_neighbours,
    args.object_radius,
)
run_suffix = f"_{result_run_tag}" if result_run_tag else ""
output_file = f"results/final_track_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}{run_suffix}{material_suffix}{scene_suffix}.csv"
per_frame_output_file = f"results/per_frame_track_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}{run_suffix}{material_suffix}{scene_suffix}.csv"


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        # idx contains the nearest-neighbor indices of GT track points among the predicted surface vertices.
        # new_mask filters which matched GT track points remain valid for evaluation in the current frame.
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return {
        "track_error": np.mean(track_errors),
        "per_frame": [float(error) for error in track_errors],
    }


file = open(output_file, mode="w", newline="", encoding="utf-8")
per_frame_file = open(per_frame_output_file, mode="w", newline="", encoding="utf-8")
writer = csv.writer(file)
per_frame_writer = csv.writer(per_frame_file)
writer.writerow(
    [
        "Case Name",
        "Train Track Error",
        "Test Track Error",
    ]
)
per_frame_writer.writerow(["scene", "split", "frame", "track"])

dir_names = glob.glob(f"{prediction_path}/*") if args.prediction_dir else glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    if args.scenes is not None and case_name not in args.scenes:
        continue
    data_case_name = scene_alias.get(case_name, case_name)
    # if data_case_name != "single_lift_dinosor":
    #     continue
    print(f"Processing {case_name} -> {data_case_name}!!!!!!!!!!!!!!!")
    
    if not os.path.exists(f"{prediction_path}/{case_name}/inference.pkl"):
        continue
    
    with open(f"{base_path}/{data_case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    with open(f"{prediction_path}/{case_name}/inference.pkl", "rb") as f:
        vertices = pickle.load(f)

    with open(f"{base_path}/{data_case_name}/gt_track_3d.pkl", "rb") as f:
        gt_track_3d = pickle.load(f)

    # Determine num_surface_points from checkpoint to limit KDTree candidates
    with open(f"{base_path}/{data_case_name}/final_data.pkl", "rb") as f:
        data = pickle.load(f)
    num_original_points = data["object_points"].shape[1]
    num_surface_points = num_original_points + data["surface_points"].shape[0]
    best_model_paths = glob.glob(f"{prediction_path}/{case_name}/train/best_*.pth")
    if best_model_paths:
        ckpt = torch.load(best_model_paths[0], map_location="cpu")
        sampled_obj = ckpt.get("obj_sample_indices", None)
        sampled_surf = ckpt.get("surface_sample_indices", None)
        if sampled_obj is not None:
            num_original_points = len(sampled_obj)
        if sampled_surf is not None:
            num_surface_points = num_original_points + len(sampled_surf)
        else:
            num_surface_points = num_original_points + data["surface_points"].shape[0]

    # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

    surface_vertices = vertices[:, :num_surface_points]
    kdtree = KDTree(surface_vertices[0])
    dis, idx = kdtree.query(gt_track_3d[0][mask])

    results_train = evaluate_prediction(
        1, train_frame, surface_vertices, gt_track_3d, idx, mask
    )
    results_test = evaluate_prediction(
        train_frame, test_frame, surface_vertices, gt_track_3d, idx, mask
    )
    writer.writerow([case_name, results_train["track_error"], results_test["track_error"]])
    for offset, track_error in enumerate(results_train["per_frame"]):
        per_frame_writer.writerow([case_name, "Reconstruction", 1 + offset, track_error])
    for offset, track_error in enumerate(results_test["per_frame"]):
        per_frame_writer.writerow([case_name, "Prediction", train_frame + offset, track_error])
file.close()
per_frame_file.close()
