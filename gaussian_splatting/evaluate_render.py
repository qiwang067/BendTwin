import os
import sys
from pathlib import Path
from PIL import Image
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json
import csv
from tqdm import tqdm
import torch
# import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import numpy as np
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from material_utils import append_material_dir, append_run_tag, build_result_run_tag

SECTION_DIVIDER = "=" * 80
TABLE_DIVIDER = "-" * 42
METRIC_KEYS = ("psnr", "ssim", "lpips", "iou")
METRIC_LABELS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "iou": "IoU",
}


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


def write_section(log_file, title):
    log_file.write("\n" + SECTION_DIVIDER + "\n")
    log_file.write(f"{title}\n")
    log_file.write(SECTION_DIVIDER + "\n\n")


def write_metric_summary(log_file, title, train_metrics, test_metrics):
    write_section(log_file, title)
    log_file.write("Note: Higher is better for PSNR / SSIM / IoU; lower is better for LPIPS.\n\n")
    log_file.write(f"{'Metric':<8} | {'Train':>12} | {'Test':>12}\n")
    log_file.write(TABLE_DIVIDER + "\n")
    for metric_key in METRIC_KEYS:
        log_file.write(
            f"{METRIC_LABELS[metric_key]:<8} | "
            f"{train_metrics[metric_key]:>12.6f} | "
            f"{test_metrics[metric_key]:>12.6f}\n"
        )


def write_scene_block(log_file, scene, metrics):
    log_file.write(f"Scene: {scene}\n")
    log_file.write(
        "  Train | "
        f"PSNR {metrics['psnr_train']:.6f} | "
        f"SSIM {metrics['ssim_train']:.6f} | "
        f"LPIPS {metrics['lpips_train']:.6f} | "
        f"IoU {metrics['iou_train']:.6f}\n"
    )
    log_file.write(
        "  Test  | "
        f"PSNR {metrics['psnr_test']:.6f} | "
        f"SSIM {metrics['ssim_test']:.6f} | "
        f"LPIPS {metrics['lpips_test']:.6f} | "
        f"IoU {metrics['iou_test']:.6f}\n\n"
    )


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--surface_sample_ratio", type=float, default=0.3)
    parser.add_argument("--obj_sample_ratio", type=float, default=1.0)
    parser.add_argument("--material_hash", type=str, default="")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--object_max_neighbours", type=int, default=None)
    parser.add_argument("--object_radius", type=float, default=None)
    parser.add_argument("--scenes", nargs="*", default=None, help="Scenes to process; process all scenes if omitted")
    parser.add_argument("--scene_tag", type=str, default="")
    args = parser.parse_args()

    render_path = './data/render_eval_data'
    human_mask_path = "./data/different_types_human_mask"
    root_data_dir = './data/gaussian_data'
    output_dir = append_material_dir(
        append_run_tag(f'./gaussian_output_dynamic_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}', args.run_tag),
        args.material_hash or None,
    )

    log_dir = './results'
    os.makedirs(log_dir, exist_ok=True)
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
    log_file_path = os.path.join(
        log_dir,
        f"output_dynamic_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}{run_suffix}{material_suffix}{scene_suffix}.txt",
    )
    per_frame_file_path = os.path.join(
        log_dir,
        f"per_frame_render_ssr{args.surface_sample_ratio}_objssr{args.obj_sample_ratio}{run_suffix}{material_suffix}{scene_suffix}.csv",
    )

    with open(log_file_path, 'w') as log_file, open(
        per_frame_file_path, "w", newline="", encoding="utf-8"
    ) as per_frame_file:
        per_frame_writer = csv.writer(per_frame_file)
        per_frame_writer.writerow(
            ["scene", "split", "frame", "view", "psnr", "ssim", "lpips", "iou"]
        )

        scene_name = sorted(os.listdir(render_path))
        scene_name = [scene for scene in scene_name if os.path.exists(os.path.join(output_dir, scene))]
        if args.scenes is not None:
            scene_name = [scene for scene in scene_name if scene in args.scenes]
        all_psnrs_train, all_ssims_train, all_lpipss_train, all_ious_train = [], [], [], []
        all_psnrs_test, all_ssims_test, all_lpipss_test, all_ious_test = [], [], [], []

        scene_metrics = {}

        for scene in scene_name:

            scene_dir = os.path.join(root_data_dir, scene)
            output_scene_dir = os.path.join(output_dir, scene)
            render_path_dir = os.path.join(render_path, scene)
            human_mask_dir = os.path.join(human_mask_path, scene)

            # Load frame split info
            with open(f"{render_path_dir}/split.json", 'r') as f:
                info = json.load(f)
            frame_len = info['frame_len']
            train_f_idx_range = list(range(info["train"][0] + 1, info["train"][1]))   # +1 if ignoring the first frame
            test_f_idx_range = list(range(info["test"][0], info["test"][1]))

            print("train indices range from", train_f_idx_range[0], "to", train_f_idx_range[-1])
            print("test indices range from", test_f_idx_range[0], "to", test_f_idx_range[-1])

            psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
            psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

            # for view_idx in range(3):
            for view_idx in range(1):   # only consider the first view

                for frame_idx in train_f_idx_range:
                    gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = gt_mask.astype(np.float32) / 255.

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
                    inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    frame_psnr = psnr(render_tensor, gt_tensor).item()
                    frame_ssim = ssim(render_tensor, gt_tensor).item()
                    frame_lpips = lpips(render_tensor, gt_tensor).item()
                    frame_iou = compute_iou(gt_mask > 0, render_mask > 0)
                    psnrs_train.append(frame_psnr)
                    ssims_train.append(frame_ssim)
                    lpipss_train.append(frame_lpips)
                    ious_train.append(frame_iou)
                    per_frame_writer.writerow(
                        [
                            scene,
                            "Reconstruction",
                            frame_idx,
                            view_idx,
                            frame_psnr,
                            frame_ssim,
                            frame_lpips,
                            frame_iou,
                        ]
                    )

                for frame_idx in test_f_idx_range:
                        
                    gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = gt_mask.astype(np.float32) / 255.

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
                    inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    frame_psnr = psnr(render_tensor, gt_tensor).item()
                    frame_ssim = ssim(render_tensor, gt_tensor).item()
                    frame_lpips = lpips(render_tensor, gt_tensor).item()
                    frame_iou = compute_iou(gt_mask > 0, render_mask > 0)
                    psnrs_test.append(frame_psnr)
                    ssims_test.append(frame_ssim)
                    lpipss_test.append(frame_lpips)
                    ious_test.append(frame_iou)
                    per_frame_writer.writerow(
                        [
                            scene,
                            "Prediction",
                            frame_idx,
                            view_idx,
                            frame_psnr,
                            frame_ssim,
                            frame_lpips,
                            frame_iou,
                        ]
                    )

            scene_metrics[scene] = {
                'psnr_train': np.mean(psnrs_train),
                'ssim_train': np.mean(ssims_train),
                'lpips_train': np.mean(lpipss_train),
                'iou_train': np.mean(ious_train),
                'psnr_test': np.mean(psnrs_test),
                'ssim_test': np.mean(ssims_test),
                'lpips_test': np.mean(lpipss_test),
                'iou_test': np.mean(ious_test)
            }

            all_psnrs_train.extend(psnrs_train)
            all_ssims_train.extend(ssims_train)
            all_lpipss_train.extend(lpipss_train)
            all_ious_train.extend(ious_train)

            all_psnrs_test.extend(psnrs_test)
            all_ssims_test.extend(ssims_test)
            all_lpipss_test.extend(lpipss_test)
            all_ious_test.extend(ious_test)

            print(f'===== Scene: {scene} =====')
            print(f'\t PSNR (train): {np.mean(psnrs_train):.4f}')
            print(f'\t SSIM (train): {np.mean(ssims_train):.4f}')
            print(f'\t LPIPS (train): {np.mean(lpipss_train):.4f}')
            print(f'\t IoU (train): {np.mean(ious_train):.4f}')

            print(f'\t PSNR (test): {np.mean(psnrs_test):.4f}')
            print(f'\t SSIM (test): {np.mean(ssims_test):.4f}')
            print(f'\t LPIPS (test): {np.mean(lpipss_test):.4f}')
            print(f'\t IoU (test): {np.mean(ious_test):.4f}')

        print('===== Overall Results Across All Scenes =====')
        print(f'\t Overall PSNR (train): {np.mean(all_psnrs_train):.4f}')
        print(f'\t Overall SSIM (train): {np.mean(all_ssims_train):.4f}')
        print(f'\t Overall LPIPS (train): {np.mean(all_lpipss_train):.4f}')
        print(f'\t Overall IoU (train): {np.mean(all_ious_train):.4f}')

        print(f'\t Overall PSNR (test): {np.mean(all_psnrs_test):.4f}')
        print(f'\t Overall SSIM (test): {np.mean(all_ssims_test):.4f}')
        print(f'\t Overall LPIPS (test): {np.mean(all_lpipss_test):.4f}')
        print(f'\t Overall IoU (test): {np.mean(all_ious_test):.4f}')

        overall_psnr_train = np.mean(all_psnrs_train)
        overall_ssim_train = np.mean(all_ssims_train)
        overall_lpips_train = np.mean(all_lpipss_train)
        overall_iou_train = np.mean(all_ious_train)
        
        overall_psnr_test = np.mean(all_psnrs_test)
        overall_ssim_test = np.mean(all_ssims_test)
        overall_lpips_test = np.mean(all_lpipss_test)
        overall_iou_test = np.mean(all_ious_test)

        write_metric_summary(
            log_file,
            "OVERALL RESULTS ACROSS ALL SCENES",
            {
                "psnr": overall_psnr_train,
                "ssim": overall_ssim_train,
                "lpips": overall_lpips_train,
                "iou": overall_iou_train,
            },
            {
                "psnr": overall_psnr_test,
                "ssim": overall_ssim_test,
                "lpips": overall_lpips_test,
                "iou": overall_iou_test,
            },
        )

        write_section(log_file, "PER-SCENE METRICS")

        for scene in scene_name:
            write_scene_block(log_file, scene, scene_metrics[scene])

        write_scene_block(
            log_file,
            "OVERALL",
            {
                "psnr_train": overall_psnr_train,
                "ssim_train": overall_ssim_train,
                "lpips_train": overall_lpips_train,
                "iou_train": overall_iou_train,
                "psnr_test": overall_psnr_test,
                "ssim_test": overall_ssim_test,
                "lpips_test": overall_lpips_test,
                "iou_test": overall_iou_test,
            },
        )
        
        print(f"\nMetrics have been saved to: {log_file_path}")
        print(f"Per-frame render metrics have been saved to: {per_frame_file_path}")
