import argparse
import os

import cv2
import numpy as np


DEFAULT_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"


def list_images(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        name
        for name in os.listdir(path)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def infer_background(render_img):
    h, w = render_img.shape[:2]
    patch = max(4, min(h, w) // 32)
    corners = np.concatenate(
        [
            render_img[:patch, :patch].reshape(-1, 3),
            render_img[:patch, w - patch :].reshape(-1, 3),
            render_img[h - patch :, :patch].reshape(-1, 3),
            render_img[h - patch :, w - patch :].reshape(-1, 3),
        ],
        axis=0,
    )
    return corners.mean(axis=0)


def foreground_mask(render_img, mode, threshold):
    if mode == "black":
        return np.linalg.norm(render_img.astype(np.float32), axis=2) > threshold
    if mode == "white":
        return np.linalg.norm(255.0 - render_img.astype(np.float32), axis=2) > threshold

    bg = infer_background(render_img).astype(np.float32)
    return np.linalg.norm(render_img.astype(np.float32) - bg[None, None, :], axis=2) > threshold


def write_overlay_video(render_dir, gt_dir, output_path, fps, alpha, render_alpha, background, threshold):
    render_names = list_images(render_dir)
    gt_names = set(list_images(gt_dir))
    names = [name for name in render_names if name in gt_names]
    if not names:
        raise FileNotFoundError(f"No matching render/gt frames: {render_dir} vs {gt_dir}")

    first_gt = cv2.imread(os.path.join(gt_dir, names[0]), cv2.IMREAD_COLOR)
    if first_gt is None:
        raise FileNotFoundError(os.path.join(gt_dir, names[0]))
    height, width = first_gt.shape[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    white = np.array([255, 255, 255], dtype=np.float32)
    try:
        for name in names:
            render_img = cv2.imread(os.path.join(render_dir, name), cv2.IMREAD_COLOR)
            gt_img = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_COLOR)
            if render_img is None or gt_img is None:
                continue
            if render_img.shape[:2] != (height, width):
                render_img = cv2.resize(render_img, (width, height), interpolation=cv2.INTER_LINEAR)
            if gt_img.shape[:2] != (height, width):
                gt_img = cv2.resize(gt_img, (width, height), interpolation=cv2.INTER_LINEAR)

            final_img = alpha * gt_img.astype(np.float32) + (1.0 - alpha) * white
            mask = foreground_mask(render_img, background, threshold)
            blended = render_alpha * render_img.astype(np.float32) + (1.0 - render_alpha) * final_img
            final_img[mask] = blended[mask]
            writer.write(np.clip(final_img, 0, 255).astype(np.uint8))
    finally:
        writer.release()

    return len(names)


def main():
    parser = argparse.ArgumentParser(description="Overlay reconstruction renders on their GT frames.")
    parser.add_argument("--model_root", required=True, help="Gaussian reconstruction output root")
    parser.add_argument("--video_root", required=True, help="Directory for overlay mp4 outputs")
    parser.add_argument("--scenes", nargs="+", required=True, help="Scene names to process")
    parser.add_argument("--exp_name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--iteration", default="10000")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.7, help="Original frame fade alpha")
    parser.add_argument("--render_alpha", type=float, default=1.0, help="Render opacity on foreground pixels")
    parser.add_argument("--background", default="auto", choices=("auto", "black", "white"))
    parser.add_argument("--threshold", type=float, default=20.0)
    args = parser.parse_args()

    for scene_name in args.scenes:
        frame_root = os.path.join(
            args.model_root,
            scene_name,
            args.exp_name,
            args.split,
            f"ours_{args.iteration}",
        )
        render_dir = os.path.join(frame_root, "renders")
        gt_dir = os.path.join(frame_root, "gt")
        output_path = os.path.join(args.video_root, scene_name, f"{args.exp_name}_integrate.mp4")
        frame_count = write_overlay_video(
            render_dir,
            gt_dir,
            output_path,
            args.fps,
            args.alpha,
            args.render_alpha,
            args.background,
            args.threshold,
        )
        print(f"Wrote {output_path} ({frame_count} frames)")


if __name__ == "__main__":
    main()
