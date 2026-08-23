import glob
import pickle
import json
import torch
import csv
import numpy as np
import os
import hashlib
from argparse import ArgumentParser
from pytorch3d.loss import chamfer_distance
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

prediction_dir = args.prediction_dir or append_material_dir(
    append_run_tag(f"./experiments_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}", args.run_tag),
    args.material_hash or None,
)
base_path = "./data/different_types"
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
output_file = f"results/final_results_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}{run_suffix}{material_suffix}{scene_suffix}.csv"
per_frame_output_file = f"results/per_frame_chamfer_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}{run_suffix}{material_suffix}{scene_suffix}.csv"

if not os.path.exists("results"):
    os.makedirs("results")

def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    object_motions_valid,
    num_original_points,
    num_surface_points,
):
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        # The motion valid indicates if the tracking is valid from prev_frame
        current_object_motions_valid = object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        chamfer_errors.append(chamfer_error.item())

    chamfer_errors = np.array(chamfer_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
        "per_frame": chamfer_errors.tolist(),
    }

    return results


if __name__ == "__main__":
    file = open(output_file, mode="w", newline="", encoding="utf-8")
    per_frame_file = open(per_frame_output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    per_frame_writer = csv.writer(per_frame_file)

    writer.writerow(
        [
            "Case Name",
            "Train Frame Num",
            "Train Chamfer Error",
            "Test Frame Num",
            "Test Chamfer Error",
        ]
    )
    per_frame_writer.writerow(["scene", "split", "frame", "chamfer"])

    dir_names = glob.glob(f"{prediction_dir}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        if args.scenes is not None and case_name not in args.scenes:
            continue
        data_case_name = scene_alias.get(case_name, case_name)
        print(f"Processing {case_name} -> {data_case_name}")

        if not os.path.exists(f"{dir_name}/inference.pkl"):
            continue
        
        # Read the trajectory data
        with open(f"{dir_name}/inference.pkl", "rb") as f:
            vertices = pickle.load(f)

        # Read the GT object points and masks
        with open(f"{base_path}/{data_case_name}/final_data.pkl", "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        other_surface_points = data["surface_points"]

        # Load sampling indices from checkpoint to match training/inference
        best_model_paths = glob.glob(f"{dir_name}/train/best_*.pth")
        if best_model_paths:
            ckpt = torch.load(best_model_paths[0], map_location="cpu")
            if "surface_sample_indices" in ckpt and ckpt["surface_sample_indices"] is not None:
                other_surface_points = other_surface_points[ckpt["surface_sample_indices"]]
            if "obj_sample_indices" in ckpt and ckpt["obj_sample_indices"] is not None:
                obj_indices = ckpt["obj_sample_indices"]
                object_points = object_points[:, obj_indices, :]
                object_visibilities = object_visibilities[:, obj_indices]
                object_motions_valid = object_motions_valid[:, obj_indices]

        num_original_points = object_points.shape[1]
        num_surface_points = num_original_points + other_surface_points.shape[0]

        # read the train/test split
        with open(f"{base_path}/{data_case_name}/split.json", "r") as f:
            split = json.load(f)
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        assert (
            test_frame == vertices.shape[0]
        ), f"Test frame {test_frame} != {vertices.shape[0]}"

        # Do the statistics on train split, only evalaute from the 2nd frame
        results_train = evaluate_prediction(
            1,
            train_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )
        results_test = evaluate_prediction(
            train_frame,
            test_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )

        writer.writerow(
            [
                case_name,
                results_train["frame_len"],
                results_train["chamfer_error"],
                results_test["frame_len"],
                results_test["chamfer_error"],
            ]
        )
        for offset, chamfer_error in enumerate(results_train["per_frame"]):
            per_frame_writer.writerow([case_name, "Reconstruction", 1 + offset, chamfer_error])
        for offset, chamfer_error in enumerate(results_test["per_frame"]):
            per_frame_writer.writerow([case_name, "Prediction", train_frame + offset, chamfer_error])
    file.close()
    per_frame_file.close()
