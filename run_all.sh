#!/usr/bin/env bash

set -e

ssr=${1:-1.0}
objssr=${2:-1.0}
gpu=${3:-0}
shift 3 || true

mode="all"
object_radius=""
object_max_neighbours=""
run_tag=""

# ===== Optional: configure material parameters here; leave empty to use configs/real.yaml defaults =====
init_spring_Y=1e4
spring_Y_min=0
spring_Y_max=5e4
bend_stiffness=75
bend_stiffness_min=50
bend_stiffness_max=100
# ==========================================================================

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_file="${script_dir}/configs/real.yaml"
gs_exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

yaml_get() {
    local key=$1
    awk -F': ' -v k="${key}" '$1 == k {print $2; exit}' "${config_file}"
}

: "${init_spring_Y:=$(yaml_get init_spring_Y)}"
: "${spring_Y_min:=$(yaml_get spring_Y_min)}"
: "${spring_Y_max:=$(yaml_get spring_Y_max)}"
: "${bend_stiffness:=$(yaml_get bend_stiffness)}"
: "${bend_stiffness_min:=$(yaml_get bend_stiffness_min)}"
: "${bend_stiffness_max:=$(yaml_get bend_stiffness_max)}"

default_scenes=(
    "double_lift_sloth" "double_lift_zebra"
    "double_stretch_sloth" "double_stretch_zebra"
    "rope_double_hand"
    "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
    "single_push_rope" "single_push_rope_1" "single_push_rope_4"
    "single_push_sloth"
    "weird_package"
)

print_help() {
    cat <<'EOF'
Usage:
  bash run_all.sh [ssr] [objssr] [gpu] [--mode all|train|eval] [material options] [scene ...]

Modes:
  --mode all    Run the complete pipeline from optimization through evaluation
  --mode train  Run optimization, training, inference, and GS only
  --mode eval   Run evaluation only; requires material-specific artifacts

Compatibility aliases:
  --eval-only   Equivalent to --mode eval
  --train-only  Equivalent to --mode train

Material options:
  --init_spring_Y v
  --spring_Y_min v
  --spring_Y_max v
  --bend_stiffness v
  --bend_stiffness_min v
  --bend_stiffness_max v

Ball-query test options:
  --object_max_neighbours k   Ball-query k for object points
  --object_radius r           Ball-query size/radius for object points
  --run_tag tag               Output-directory suffix; defaults to bqk<k>_r<r> generated from k/r
EOF
}

scenes=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)
            mode=$2
            shift 2
            ;;
        --eval-only)
            mode="eval"
            shift
            ;;
        --train-only)
            mode="train"
            shift
            ;;
        --init_spring_Y)
            init_spring_Y=$2
            shift 2
            ;;
        --spring_Y_min)
            spring_Y_min=$2
            shift 2
            ;;
        --spring_Y_max)
            spring_Y_max=$2
            shift 2
            ;;
        --bend_stiffness)
            bend_stiffness=$2
            shift 2
            ;;
        --bend_stiffness_min)
            bend_stiffness_min=$2
            shift 2
            ;;
        --bend_stiffness_max)
            bend_stiffness_max=$2
            shift 2
            ;;
        --object_max_neighbours)
            object_max_neighbours=$2
            shift 2
            ;;
        --object_radius)
            object_radius=$2
            shift 2
            ;;
        --run_tag)
            run_tag=$2
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            scenes+=("$1")
            shift
            ;;
    esac
done

case "${mode}" in
    all|train|eval)
        ;;
    *)
        echo "❌ Unsupported mode: ${mode}"
        print_help
        exit 1
        ;;
esac

if [ -z "${run_tag}" ]; then
    if [ -n "${object_max_neighbours}" ] && [ -n "${object_radius}" ]; then
        run_tag="bqk${object_max_neighbours}_r${object_radius}"
    elif [ -n "${object_max_neighbours}" ]; then
        run_tag="bqk${object_max_neighbours}"
    elif [ -n "${object_radius}" ]; then
        run_tag="r${object_radius}"
    fi
fi

run_suffix=""
if [ -n "${run_tag}" ]; then
    run_suffix="_${run_tag}"
fi

experiment_root="./experiments_ssr${ssr}_objssr${objssr}${run_suffix}"
optimization_root="./experiments_optimization_ssr${ssr}_objssr${objssr}${run_suffix}"
gaussian_root="./gaussian_output_ssr${ssr}_objssr${objssr}${run_suffix}"
gaussian_video_root="./gaussian_output_video_ssr${ssr}_objssr${objssr}${run_suffix}"
gaussian_dynamic_root="./gaussian_output_dynamic_ssr${ssr}_objssr${objssr}${run_suffix}"
gaussian_dynamic_white_root="./gaussian_output_dynamic_white_ssr${ssr}_objssr${objssr}${run_suffix}"

ball_query_args=()
if [ -n "${object_radius}" ]; then
    ball_query_args+=(--object_radius "${object_radius}")
fi
if [ -n "${object_max_neighbours}" ]; then
    ball_query_args+=(--object_max_neighbours "${object_max_neighbours}")
fi
if [ -n "${run_tag}" ]; then
    ball_query_args+=(--run_tag "${run_tag}")
fi

material_hash=$(
    python3 material_utils.py hash \
        --init_spring_Y "${init_spring_Y}" \
        --spring_Y_min "${spring_Y_min}" \
        --spring_Y_max "${spring_Y_max}" \
        --bend_stiffness "${bend_stiffness}" \
        --bend_stiffness_min "${bend_stiffness_min}" \
        --bend_stiffness_max "${bend_stiffness_max}"
)
material_tag="mat${material_hash}"

format_time() {
    local seconds=$1
    printf "%dh %dm %ds" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

write_material_config_dir() {
    python3 material_utils.py write \
        --dir "$1" \
        --material_hash "${material_hash}" \
        --init_spring_Y "${init_spring_Y}" \
        --spring_Y_min "${spring_Y_min}" \
        --spring_Y_max "${spring_Y_max}" \
        --bend_stiffness "${bend_stiffness}" \
        --bend_stiffness_min "${bend_stiffness_min}" \
        --bend_stiffness_max "${bend_stiffness_max}" >/dev/null
}

discover_eval_scenes() {
    local discovered=()
    local base_dir="${experiment_root}/${material_tag}"
    [ -d "${base_dir}" ] || return 0

    while IFS= read -r scene_dir; do
        local scene_name
        scene_name=$(basename "${scene_dir}")
        local inference_path="${base_dir}/${scene_name}/inference.pkl"
        local model_cfg="${gaussian_root}/${material_tag}/${scene_name}/${gs_exp_name}/cfg_args"

        if [ -f "${inference_path}" ] && [ -f "${model_cfg}" ]; then
            discovered+=("${scene_name}")
        fi
    done < <(find "${base_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

    if [ ${#discovered[@]} -gt 0 ]; then
        scenes=("${discovered[@]}")
    fi
}

validate_material_scene() {
    local scene_name=$1
    local exp_dir="${experiment_root}/${material_tag}/${scene_name}"
    local inference_path="${exp_dir}/inference.pkl"
    local model_cfg="${gaussian_root}/${material_tag}/${scene_name}/${gs_exp_name}/cfg_args"

    if [ ! -d "${exp_dir}" ]; then
        echo "Missing material experiment directory: ${exp_dir}"
        return 1
    fi

    if [ ! -f "${inference_path}" ]; then
        echo "Missing material inference output: ${inference_path}"
        return 1
    fi

    if [ ! -f "${model_cfg}" ]; then
        echo "Missing material Gaussian output: ${model_cfg}"
        return 1
    fi

    return 0
}

validate_eval_scenes_or_exit() {
    local invalid=0

    if [ ${#scenes[@]} -eq 0 ]; then
        echo "❌ No material-specific scenes available for evaluation were found."
        echo "Both of the following are required:"
        echo "  1. ${experiment_root}/${material_tag}/<scene>/inference.pkl"
        echo "  2. ${gaussian_root}/${material_tag}/<scene>/${gs_exp_name}/cfg_args"
        echo "Run training, inference, and GS first."
        exit 1
    fi

    for scene_name in "${scenes[@]}"; do
        if ! validate_material_scene "${scene_name}"; then
            invalid=1
        fi
    done

    if [ ${invalid} -ne 0 ]; then
        echo "❌ The scenes above are missing material-specific artifacts; evaluation aborted."
        exit 1
    fi
}

validate_train_outputs_or_exit() {
    local invalid=0
    for scene_name in "${scenes[@]}"; do
        local best_count
        best_count=$(find "${experiment_root}/${material_tag}/${scene_name}/train" -maxdepth 1 -type f -name 'best_*.pth' 2>/dev/null | wc -l)
        if [ "${best_count}" -eq 0 ]; then
            echo "Missing best training checkpoint: ${experiment_root}/${material_tag}/${scene_name}/train/best_*.pth" >&2
            invalid=1
        fi
    done
    if [ ${invalid} -ne 0 ]; then
        echo "❌ Physics-training artifacts are incomplete; subsequent stages aborted." >&2
        exit 1
    fi
}

validate_inference_outputs_or_exit() {
    local invalid=0
    for scene_name in "${scenes[@]}"; do
        if [ ! -f "${experiment_root}/${material_tag}/${scene_name}/inference.pkl" ]; then
            echo "Missing inference output: ${experiment_root}/${material_tag}/${scene_name}/inference.pkl" >&2
            invalid=1
        fi
    done
    if [ ${invalid} -ne 0 ]; then
        echo "❌ Inference artifacts are incomplete; subsequent stages aborted." >&2
        exit 1
    fi
}

run_step() {
    local name=$1
    shift
    echo "=== ${name} (mode=${mode}, ssr=${ssr}, objssr=${objssr}, material_hash=${material_hash}, init_spring_Y=${init_spring_Y}, spring_Y_min=${spring_Y_min}, spring_Y_max=${spring_Y_max}, bend_stiffness=${bend_stiffness}, bend_stiffness_min=${bend_stiffness_min}, bend_stiffness_max=${bend_stiffness_max}, object_max_neighbours=${object_max_neighbours}, object_radius=${object_radius}, run_tag=${run_tag}) ==="
    local t0=$SECONDS
    "$@"
    local elapsed=$((SECONDS - t0))
    echo "⏱ ${name} elapsed time: $(format_time ${elapsed})"
    echo ""
}

run_train_pipeline() {
    step="[1/4] Zero-order Optimization"
    run_step "${step}" python script_optimize.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --gpu ${gpu} --material_hash ${material_hash} --init_spring_Y ${init_spring_Y} --spring_Y_min ${spring_Y_min} --spring_Y_max ${spring_Y_max} --bend_stiffness ${bend_stiffness} --bend_stiffness_min ${bend_stiffness_min} --bend_stiffness_max ${bend_stiffness_max} "${ball_query_args[@]}" --scenes "${scenes[@]}"

    step="[2/4] First-order Optimization"
    run_step "${step}" python script_train.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --gpu ${gpu} --material_hash ${material_hash} --init_spring_Y ${init_spring_Y} --spring_Y_min ${spring_Y_min} --spring_Y_max ${spring_Y_max} --bend_stiffness ${bend_stiffness} --bend_stiffness_min ${bend_stiffness_min} --bend_stiffness_max ${bend_stiffness_max} "${ball_query_args[@]}" --scenes "${scenes[@]}"
    validate_train_outputs_or_exit

    step="[3/4] Inference"
    run_step "${step}" python script_inference.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --gpu ${gpu} --material_hash ${material_hash} --init_spring_Y ${init_spring_Y} --spring_Y_min ${spring_Y_min} --spring_Y_max ${spring_Y_max} --bend_stiffness ${bend_stiffness} --bend_stiffness_min ${bend_stiffness_min} --bend_stiffness_max ${bend_stiffness_max} "${ball_query_args[@]}" --scenes "${scenes[@]}"
    validate_inference_outputs_or_exit

    step="[4/4] Gaussian Splatting"
    run_step "${step}" bash gs_run.sh ${ssr} ${objssr} ${gpu} "${scenes[@]}"
}

run_eval_pipeline() {
    validate_inference_outputs_or_exit

    step="[1/5] Render Dynamic Videos"
    run_step "${step}" bash gs_run_simulate.sh ${ssr} ${objssr} "${scenes[@]}"

    step="[2/5] Export Render Eval Data"
    run_step "${step}" python export_render_eval_data.py --scenes "${scenes[@]}"

    step="[3/5] Quantitative Evaluation"
    run_step "${step}" bash evaluate.sh ${ssr} ${objssr} "${scenes[@]}"

    step="[4/5] Render White Background Videos"
    run_step "${step}" bash gs_run_simulate_white.sh ${ssr} ${objssr} "${scenes[@]}"

    step="[5/5] Visualize Render Results"
    run_step "${step}" python visualize_render_results.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash ${material_hash} --run_tag "${run_tag}" --scenes "${scenes[@]}"
}

trap 'echo ""; echo "❌ Run failed at step ${step}; exit code: $?"' ERR

if [ ${#scenes[@]} -eq 0 ]; then
    if [ "${mode}" = "eval" ]; then
        discover_eval_scenes
    else
        scenes=("${default_scenes[@]}")
    fi
fi

if [ "${mode}" = "eval" ]; then
    validate_eval_scenes_or_exit
fi

SCENES="${scenes[*]}"
export SCENES
export CUDA_VISIBLE_DEVICES=${gpu}
export MATERIAL_HASH=${material_hash}
export BQ_RUN_TAG=${run_tag}
export OBJECT_RADIUS=${object_radius}
export OBJECT_MAX_NEIGHBOURS=${object_max_neighbours}

write_material_config_dir "${optimization_root}/${material_tag}"
write_material_config_dir "${experiment_root}/${material_tag}"
write_material_config_dir "${gaussian_root}/${material_tag}"
write_material_config_dir "${gaussian_video_root}/${material_tag}"
write_material_config_dir "${gaussian_dynamic_root}/${material_tag}"
write_material_config_dir "${gaussian_dynamic_white_root}/${material_tag}"

total_start=$SECONDS

case "${mode}" in
    all)
        run_train_pipeline
        run_eval_pipeline
        ;;
    train)
        run_train_pipeline
        ;;
    eval)
        run_eval_pipeline
        ;;
esac

total_elapsed=$((SECONDS - total_start))
echo "=== ✅ All done! Total elapsed time: $(format_time ${total_elapsed}) ==="
