from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json
from material_utils import append_material_dir, append_run_tag


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)


def apply_object_ball_query_overrides(cli_object_radius=None, cli_object_max_neighbours=None):
    object_radius = cli_object_radius if cli_object_radius is not None else os.environ.get("OBJECT_RADIUS")
    object_max_neighbours = (
        cli_object_max_neighbours
        if cli_object_max_neighbours is not None
        else os.environ.get("OBJECT_MAX_NEIGHBOURS")
    )

    if object_radius not in (None, ""):
        cfg.object_radius = float(object_radius)
    if object_max_neighbours not in (None, ""):
        cfg.object_max_neighbours = int(object_max_neighbours)


def apply_material_overrides(
    cli_init_spring_Y=None,
    cli_spring_Y_min=None,
    cli_spring_Y_max=None,
    cli_bend_stiffness=None,
    cli_bend_stiffness_min=None,
    cli_bend_stiffness_max=None,
):
    init_spring_Y = cli_init_spring_Y if cli_init_spring_Y is not None else os.environ.get("INIT_SPRING_Y")
    spring_Y_min = cli_spring_Y_min if cli_spring_Y_min is not None else os.environ.get("SPRING_Y_MIN")
    spring_Y_max = cli_spring_Y_max if cli_spring_Y_max is not None else os.environ.get("SPRING_Y_MAX")
    bend_stiffness = cli_bend_stiffness if cli_bend_stiffness is not None else os.environ.get("BEND_STIFFNESS")
    bend_stiffness_min = cli_bend_stiffness_min if cli_bend_stiffness_min is not None else os.environ.get("BEND_STIFFNESS_MIN")
    bend_stiffness_max = cli_bend_stiffness_max if cli_bend_stiffness_max is not None else os.environ.get("BEND_STIFFNESS_MAX")

    if init_spring_Y not in (None, ""):
        cfg.init_spring_Y = float(init_spring_Y)
    if spring_Y_min not in (None, ""):
        cfg.spring_Y_min = float(spring_Y_min)
    if spring_Y_max not in (None, ""):
        cfg.spring_Y_max = float(spring_Y_max)
    if bend_stiffness not in (None, ""):
        cfg.bend_stiffness = float(bend_stiffness)
    if bend_stiffness_min not in (None, ""):
        cfg.bend_stiffness_min = float(bend_stiffness_min)
    if bend_stiffness_max not in (None, ""):
        cfg.bend_stiffness_max = float(bend_stiffness_max)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--surface_sample_ratio", type=float, default=0.3)
    parser.add_argument("--obj_sample_ratio", type=float, default=1.0)
    parser.add_argument("--init_spring_Y", type=float, default=None)
    parser.add_argument("--spring_Y_min", type=float, default=None)
    parser.add_argument("--spring_Y_max", type=float, default=None)
    parser.add_argument("--bend_stiffness", type=float, default=None)
    parser.add_argument("--bend_stiffness_min", type=float, default=None)
    parser.add_argument("--bend_stiffness_max", type=float, default=None)
    parser.add_argument("--material_hash", type=str, default=None)
    parser.add_argument("--object_radius", type=float, default=None)
    parser.add_argument("--object_max_neighbours", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument(
        "--experiment_root_override",
        type=str,
        default="",
        help="Optional output experiment root. Used for cross-interaction inference.",
    )
    parser.add_argument(
        "--optimal_params_path",
        type=str,
        default="",
        help="Optional optimal_params.pkl path. Overrides the default per-scene path.",
    )
    parser.add_argument(
        "--checkpoint_load_mode",
        choices=("full", "sampling_indices"),
        default="full",
        help="Use full checkpoint physics, or only its sampling indices.",
    )
    parser.add_argument("--output_case_name", type=str, default="", help="Optional output directory name for this case")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    ssr = args.surface_sample_ratio
    objssr = args.obj_sample_ratio

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    cfg.surface_sample_ratio = ssr
    cfg.obj_sample_ratio = objssr
    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    experiment_root = args.experiment_root_override or append_run_tag(
        f"experiments_ssr{ssr}_objssr{objssr}",
        args.run_tag,
    )
    base_root = append_material_dir(experiment_root, args.material_hash)
    output_case_name = args.output_case_name or case_name
    base_dir = f"{base_root}/{output_case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimization_root = append_run_tag(f"experiments_optimization_ssr{ssr}_objssr{objssr}", args.run_tag)
    optimal_root = append_material_dir(optimization_root, args.material_hash)
    optimal_path = args.optimal_params_path or f"{optimal_root}/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)
    apply_material_overrides(
        args.init_spring_Y,
        args.spring_Y_min,
        args.spring_Y_max,
        args.bend_stiffness,
        args.bend_stiffness_min,
        args.bend_stiffness_max,
    )
    apply_object_ball_query_overrides(args.object_radius, args.object_max_neighbours)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inference_log")

    # Load sampling indices from the checkpoint before initializing the trainer, which loads the data.
    assert len(glob.glob(f"{base_dir}/train/best_*.pth")) > 0
    best_model_path = glob.glob(f"{base_dir}/train/best_*.pth")[0]
    checkpoint = torch.load(best_model_path, map_location="cpu")
    if "surface_sample_indices" in checkpoint:
        cfg.surface_sample_indices = checkpoint["surface_sample_indices"]
        logger.info("Pre-loaded surface_sample_indices from checkpoint")
    if "obj_sample_indices" in checkpoint:
        cfg.obj_sample_indices = checkpoint["obj_sample_indices"]
        logger.info("Pre-loaded obj_sample_indices from checkpoint")

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    if args.checkpoint_load_mode == "full":
        trainer.test(best_model_path)
    else:
        trainer.test(None)
