import glob
import os
import json
import numpy as np
import cv2
from argparse import ArgumentParser
from material_utils import append_material_dir, append_run_tag

parser = ArgumentParser()
parser.add_argument("--surface_sample_ratio", type=float, default=0.3)
parser.add_argument("--obj_sample_ratio", type=float, default=1.0)
parser.add_argument("--material_hash", type=str, default="")
parser.add_argument("--run_tag", type=str, default="")
parser.add_argument("--scenes", nargs="*", default=None, help="Scenes to process; process all scenes if omitted")
parser.add_argument("--prediction_dir", type=str, default="", help="Override dynamic render directory")
args = parser.parse_args()

base_path = "./data/different_types"
prediction_dir = args.prediction_dir or append_material_dir(
    append_run_tag(
        f"./gaussian_output_dynamic_white_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}",
        args.run_tag,
    ),
    args.material_hash or None,
)
human_mask_path = "./data/different_types_human_mask"
object_mask_path = "./data/render_eval_data"

height, width = 480, 848
FPS = 30
alpha = 0.7

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    if args.scenes is not None and case_name not in args.scenes:
        continue
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]

    # Need to prepare the video
    for i in range(3):
        # Process each camera
        overlay_frame_dir = f"{prediction_dir}/{case_name}/{i}_overlay"
        os.makedirs(overlay_frame_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(
            f"{prediction_dir}/{case_name}/{i}_integrate.mp4",
            fourcc,
            FPS,
            (width, height),
        )

        for frame_idx in range(frame_len):
            render_path = f"{prediction_dir}/{case_name}/{i}/{frame_idx:05d}.png"
            origin_image_path = f"{base_path}/{case_name}/color/{i}/{frame_idx}.png"
            human_mask_image_path = (
                f"{human_mask_path}/{case_name}/mask/{i}/0/{frame_idx}.png"
            )
            object_image_path = (
                f"{object_mask_path}/{case_name}/mask/{i}/{frame_idx}.png"
            )

            render_img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
            origin_img = cv2.imread(origin_image_path)
            human_mask = cv2.imread(human_mask_image_path)
            human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)
            human_mask = human_mask > 0
            object_mask = cv2.imread(object_image_path)
            object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
            object_mask = object_mask > 0

            final_image = origin_img.copy()
            render_mask = np.logical_and(
                (render_img != 0).any(axis=2), render_img[:, :, 3] > 100
            )
            render_img[~render_mask, 3] = 0

            final_image[:, :, :] = alpha * final_image + (1 - alpha) * np.array(
                [255, 255, 255], dtype=np.uint8
            )

            test_alpha = render_img[:, :, 3] / 255
            final_image[:, :, :] = render_img[:, :, :3] * test_alpha[
                :, :, None
            ] + final_image * (1 - test_alpha[:, :, None])

            final_image[human_mask] = alpha * origin_img[human_mask] + (
                1 - alpha
            ) * np.array([255, 255, 255], dtype=np.uint8)

            cv2.imwrite(f"{overlay_frame_dir}/{frame_idx:05d}.png", final_image)
            video_writer.write(final_image)

        video_writer.release()
