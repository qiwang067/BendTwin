import os
import csv
import json
import pickle
import numpy as np
import open3d as o3d

base_path = "./data/different_types"
output_path = "./data/gaussian_data"
CONTROLLER_NAME = "hand"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


existDir(output_path)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        # Create the directory for the case
        existDir(f"{output_path}/{case_name}")
        for i in range(3):
            # Copy the original RGB image
            os.system(
                f"cp {base_path}/{case_name}/color/{i}/0.png {output_path}/{case_name}/{i}.png"
            )
            # Copy the original mask image
            # Get the mask path for the image
            with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            mask_path = f"{base_path}/{case_name}/mask/{i}/{obj_idx}/0.png"
            os.system(f"cp {mask_path} {output_path}/{case_name}/mask_{i}.png")
            # Prepare the high-resolution image
            os.system(
                f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/{i}/0.png --output_path {output_path}/{case_name}/{i}_high.png --category {category}"
            )
            # Prepare the segmentation mask of the high-resolution image
            os.system(
                f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}_high.png --TEXT_PROMPT {category} --output_path {output_path}/{case_name}/mask_{i}_high.png"
            )

            # Copy the original depth image
            os.system(
                f"cp {base_path}/{case_name}/depth/{i}/0.npy {output_path}/{case_name}/{i}_depth.npy"
            )

            # Prepare the human mask for the low-resolution image and high-resolution image
            os.system(
                f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}.png --TEXT_PROMPT 'human' --output_path {output_path}/{case_name}/mask_human_{i}.png"
            )
            os.system(
                f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}_high.png --TEXT_PROMPT 'human' --output_path {output_path}/{case_name}/mask_human_{i}_high.png"
            )

        # Prepare the intrinsic and extrinsic parameters
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            intrinsics = json.load(f)["intrinsics"]
        data = {}
        data["c2ws"] = c2ws
        data["intrinsics"] = intrinsics
        with open(f"{output_path}/{case_name}/camera_meta.pkl", "wb") as f:
            pickle.dump(data, f)

        # Prepare the shape initialization data
        # If with shape prior, then copy the shape prior data
        if shape_prior.lower() == "true":
            os.system(
                f"cp {base_path}/{case_name}/shape/matching/final_mesh.glb {output_path}/{case_name}/shape_prior.glb"
            )
        # Save the original pcd data into the world coordinate system
        obs_points = []
        obs_colors = []
        pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
        processed_mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
        data = np.load(pcd_path)
        with open(processed_mask_path, "rb") as f:
            processed_masks = pickle.load(f)
        for i in range(3):
            points = data["points"][i]
            colors = data["colors"][i]
            mask = processed_masks[0][i]["object"]
            obs_points.append(points[mask])
            obs_colors.append(colors[mask])

        obs_points = np.vstack(obs_points)
        obs_colors = np.vstack(obs_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([pcd, coordinate])
        o3d.io.write_point_cloud(f"{output_path}/{case_name}/observation.ply", pcd)
