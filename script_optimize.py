import glob
import os
import sys
import json
import multiprocessing
from pathlib import Path
import torch
from argparse import ArgumentParser
from material_utils import append_material_dir, append_run_tag, ball_query_run_tag

BASE_PATH = "./data/different_types"


def build_optional_arg(name, value):
    if value in (None, ""):
        return ""
    return f" --{name} {value}"


def worker_run_case(
    queue,
    gpu_id,
    surface_sample_ratio,
    obj_sample_ratio,
    init_spring_Y,
    spring_Y_min,
    spring_Y_max,
    bend_stiffness,
    bend_stiffness_min,
    bend_stiffness_max,
    material_hash,
    object_radius,
    object_max_neighbours,
    run_tag,
):
    """Bind a worker to one GPU and dynamically pull tasks from the queue."""
    here_path = Path(__file__).parent
    while True:
        case_name = queue.get()
        if case_name is None:
            break
        try:
            with open(f"{BASE_PATH}/{case_name}/split.json", "r") as f:
                split = json.load(f)
            train_frame = split["train"][1]
        except Exception:
            continue
        exp_dir = case_name
        experiment_root = append_run_tag(
            f"experiments_optimization_ssr{surface_sample_ratio}_objssr{obj_sample_ratio}",
            run_tag,
        )
        exp_base = append_material_dir(experiment_root, material_hash)
        if (here_path / exp_base / exp_dir / "optimal_params.pkl").exists():
            continue
        print(
            f"[GPU {gpu_id}] Start: {case_name}, train_frame={train_frame}, "
            f"surface_sample_ratio={surface_sample_ratio}, obj_sample_ratio={obj_sample_ratio}, "
            f"init_spring_Y={init_spring_Y}, spring_Y_min={spring_Y_min}, spring_Y_max={spring_Y_max}, "
            f"bend_stiffness={bend_stiffness}, bend_stiffness_min={bend_stiffness_min}, bend_stiffness_max={bend_stiffness_max}, "
            f"object_max_neighbours={object_max_neighbours}, object_radius={object_radius}, run_tag={run_tag}"
        )
        optional_args = (
            build_optional_arg("init_spring_Y", init_spring_Y)
            + build_optional_arg("spring_Y_min", spring_Y_min)
            + build_optional_arg("spring_Y_max", spring_Y_max)
            + build_optional_arg("bend_stiffness", bend_stiffness)
            + build_optional_arg("bend_stiffness_min", bend_stiffness_min)
            + build_optional_arg("bend_stiffness_max", bend_stiffness_max)
            + build_optional_arg("material_hash", material_hash)
            + build_optional_arg("object_radius", object_radius)
            + build_optional_arg("object_max_neighbours", object_max_neighbours)
            + build_optional_arg("run_tag", run_tag)
        )
        exit_code = os.system(
            f"CUDA_VISIBLE_DEVICES={gpu_id} python optimize_cma.py "
            f"--base_path {BASE_PATH} --case_name {case_name} --train_frame {train_frame} "
            f"--surface_sample_ratio {surface_sample_ratio} --obj_sample_ratio {obj_sample_ratio} "
            f"{optional_args}"
        )
        if exit_code != 0:
            raise RuntimeError(f"optimize_cma.py failed for {case_name} with exit code {exit_code}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--surface_sample_ratio", type=float, default=0.3,
                        help="Ratio of randomly sampled exterior surface points (0.0-1.0); default: 0.3")
    parser.add_argument("--obj_sample_ratio", type=float, default=1.0,
                        help="Ratio of randomly sampled object points (0.0-1.0); default: 1.0")
    parser.add_argument("--init_spring_Y", type=float, default=None)
    parser.add_argument("--spring_Y_min", type=float, default=None)
    parser.add_argument("--spring_Y_max", type=float, default=None)
    parser.add_argument("--bend_stiffness", type=float, default=None)
    parser.add_argument("--bend_stiffness_min", type=float, default=None)
    parser.add_argument("--bend_stiffness_max", type=float, default=None)
    parser.add_argument("--material_hash", type=str, default=None)
    parser.add_argument("--object_radius", type=float, default=None)
    parser.add_argument("--object_max_neighbours", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--scenes", nargs="*", default=None, help="Scenes to process; process all scenes if omitted")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use; default: 0")
    parser.add_argument("--parallel", action="store_true", help="Enable multi-GPU parallelism; default: serial execution on one GPU")
    args = parser.parse_args()
    args.run_tag = args.run_tag or ball_query_run_tag(
        args.object_max_neighbours,
        args.object_radius,
    )

    dir_names = glob.glob(f"{BASE_PATH}/*")
    case_names = sorted([d.split("/")[-1] for d in dir_names])
    if args.scenes is not None:
        case_names = [c for c in case_names if c in args.scenes]

    if args.parallel:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPU detected; using the CPU.")
            num_gpus = 1
        print(
            f"[Parallel mode] Detected {num_gpus} GPUs for {len(case_names)} cases; tasks will be dynamically assigned to available GPUs."
            f"surface_sample_ratio={args.surface_sample_ratio}, obj_sample_ratio={args.obj_sample_ratio}, "
            f"init_spring_Y={args.init_spring_Y}, spring_Y_min={args.spring_Y_min}, spring_Y_max={args.spring_Y_max}, "
            f"bend_stiffness={args.bend_stiffness}, bend_stiffness_min={args.bend_stiffness_min}, bend_stiffness_max={args.bend_stiffness_max}, "
            f"object_max_neighbours={args.object_max_neighbours}, object_radius={args.object_radius}, run_tag={args.run_tag}"
        )

        queue = multiprocessing.Queue()
        for name in case_names:
            queue.put(name)
        for _ in range(num_gpus):
            queue.put(None)

        workers = [
            multiprocessing.Process(
                target=worker_run_case,
                args=(
                    queue,
                    i,
                    args.surface_sample_ratio,
                    args.obj_sample_ratio,
                    args.init_spring_Y,
                    args.spring_Y_min,
                    args.spring_Y_max,
                    args.bend_stiffness,
                    args.bend_stiffness_min,
                    args.bend_stiffness_max,
                    args.material_hash,
                    args.object_radius,
                    args.object_max_neighbours,
                    args.run_tag,
                ),
            )
            for i in range(num_gpus)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        failed_workers = [w.exitcode for w in workers if w.exitcode not in (0, None)]
        if failed_workers:
            print(f"Worker failed with exit codes: {failed_workers}", file=sys.stderr)
            sys.exit(1)
    else:
        gpu_id = args.gpu
        print(
            f"[Serial mode] Using GPU {gpu_id} for {len(case_names)} cases."
            f"surface_sample_ratio={args.surface_sample_ratio}, obj_sample_ratio={args.obj_sample_ratio}, "
            f"init_spring_Y={args.init_spring_Y}, spring_Y_min={args.spring_Y_min}, spring_Y_max={args.spring_Y_max}, "
            f"bend_stiffness={args.bend_stiffness}, bend_stiffness_min={args.bend_stiffness_min}, bend_stiffness_max={args.bend_stiffness_max}, "
            f"object_max_neighbours={args.object_max_neighbours}, object_radius={args.object_radius}, run_tag={args.run_tag}"
        )
        queue = multiprocessing.Queue()
        for name in case_names:
            queue.put(name)
        queue.put(None)
        worker_run_case(
            queue,
            gpu_id,
            args.surface_sample_ratio,
            args.obj_sample_ratio,
            args.init_spring_Y,
            args.spring_Y_min,
            args.spring_Y_max,
            args.bend_stiffness,
            args.bend_stiffness_min,
            args.bend_stiffness_max,
            args.material_hash,
            args.object_radius,
            args.object_max_neighbours,
            args.run_tag,
        )
