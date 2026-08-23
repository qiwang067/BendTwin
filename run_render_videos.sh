#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_render_videos.sh [ssr] [objssr] [gpu] [scene...]

Examples:
  bash run_render_videos.sh 1.0 1.0 1 single_push_rope_1
  bash run_render_videos.sh 0.05 0.05 1 single_push_rope_1
  bash run_render_videos.sh 1.0 1.0 0 double_lift_sloth single_push_sloth
  MATERIAL_HASH=eaf28e8120 SCENES="double_lift_sloth single_push_sloth" bash run_render_videos.sh 1.0 1.0 0
  BQ_RUN_TAG=bqk30_r0.02 MATERIAL_HASH=05e72ba389 bash run_render_videos.sh 0.05 0.05 0

Environment overrides:
  RUN_RECONSTRUCTION=false     Skip physics reconstruction videos driven by control-point actions.
  RUN_RESIMULATION=false       Deprecated alias for RUN_RECONSTRUCTION=false.
  RUN_STATIC_RECONSTRUCTION=true Render raw Gaussian test-view reconstruction videos.
  RUN_WHITE=false              Skip white-background physics reconstruction renders.
  RUN_INTEGRATED=false         Skip physics reconstruction integrated overlay videos.
  RUN_RECON_INTEGRATED=false   Skip raw Gaussian integrated overlay videos.
  FORCE_RENDER=true            Re-render reconstruction frames even if they exist.
  GS_ITERATIONS=10000          Gaussian iteration to render/convert.
  AUTO_BEST_HASH=false         Disable automatic best-of report material selection.
  BEST_PTPP_REPORT=...         Override best-of report used for automatic material selection.
  BQ_RUN_TAG=...               Append run tag to experiment/output directories.
  MATERIAL_HASH=...            Use explicit material subdirectory mat<hash>.
  GS_OUTPUT_DIR=...            Override reconstruction model directory.
  GS_VIDEO_DIR=...             Override reconstruction video directory.
  DYNAMIC_OUTPUT_DIR=...       Override reconstruction-resimulation output directory.
  DYNAMIC_OUTPUT_WHITE_DIR=... Override white-background reconstruction-resimulation output directory.

Default outputs are written to:
  ./gaussian_output_ssr...
  ./gaussian_output_video_ssr...
  ./gaussian_output_dynamic_ssr...
  ./gaussian_output_reconstruction_resimulation_white_ssr...

When MATERIAL_HASH is set, /mat<hash> is appended to each root, matching explicit outputs.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

ssr=${1:-1.0}
objssr=${2:-1.0}
gpu=${3:-0}
if [ $# -ge 3 ]; then
    shift 3
else
    set --
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

run_static_reconstruction="${RUN_STATIC_RECONSTRUCTION:-false}"
run_reconstruction="${RUN_RECONSTRUCTION:-${RUN_RESIMULATION:-true}}"
run_white="${RUN_WHITE:-true}"
run_integrated="${RUN_INTEGRATED:-true}"
run_recon_integrated="${RUN_RECON_INTEGRATED:-true}"
force_render="${FORCE_RENDER:-false}"
gs_iterations="${GS_ITERATIONS:-10000}"
auto_best_hash="${AUTO_BEST_HASH:-true}"

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

set_output_paths() {
    run_suffix=""
    if [ -n "${BQ_RUN_TAG:-}" ]; then
        run_suffix="_${BQ_RUN_TAG}"
    fi

    material_root_suffix=""
    if [ -n "${MATERIAL_HASH:-}" ]; then
        material_root_suffix="/mat${MATERIAL_HASH#mat}"
    fi

    gs_output_dir="${GS_OUTPUT_DIR:-./gaussian_output_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}}"
    gs_video_dir="${GS_VIDEO_DIR:-./gaussian_output_video_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}}"
    exp_dir="./experiments_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}"
    dynamic_output_dir="${DYNAMIC_OUTPUT_DIR:-./gaussian_output_dynamic_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}}"
    dynamic_output_white_dir="${DYNAMIC_OUTPUT_WHITE_DIR:-./gaussian_output_reconstruction_resimulation_white_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}}"
}

set_output_paths

if [ $# -gt 0 ]; then
    scenes=("$@")
elif [ -n "${SCENES:-}" ]; then
    read -r -a scenes <<< "${SCENES}"
elif [ -d "${gs_output_dir}" ]; then
    scenes=()
    for scene_dir in "${gs_output_dir}"/*; do
        [ -d "${scene_dir}" ] || continue
        scenes+=("$(basename "${scene_dir}")")
    done
else
    scenes=("double_stretch_sloth")
fi

find_best_ptpp_report() {
    if [ -n "${BEST_PTPP_REPORT:-}" ]; then
        if [ -f "${BEST_PTPP_REPORT}" ]; then
            printf '%s\n' "${BEST_PTPP_REPORT}"
            return 0
        fi
        echo "ERROR: BEST_PTPP_REPORT does not exist: ${BEST_PTPP_REPORT}" >&2
        return 1
    fi

    local report
    if [ -n "${BQ_RUN_TAG:-}" ]; then
        report="./results/best_vs_pt_ssr${ssr}_objssr${objssr}_${BQ_RUN_TAG}_all_scenes_ptpp_merged.txt"
    else
        report="./results/best_vs_pt_ssr${ssr}_objssr${objssr}_all_scenes_ptpp_merged.txt"
    fi
    if [ -f "${report}" ]; then
        printf '%s\n' "${report}"
        return 0
    fi

    return 1
}

source_candidate_for_scene() {
    local report=$1
    local scene_name=$2

    awk -v target="${scene_name}" '
        $0 == "Scene: " target { in_scene = 1; next }
        in_scene && /^Scene: / { exit }
        in_scene && /Source Candidate:/ {
            sub(/^  Source Candidate: /, "")
            print
            exit
        }
    ' "${report}"
}

apply_source_candidate() {
    local candidate=$1
    local prefix="ssr${ssr}_objssr${objssr}_"
    local rest
    local material_part

    if [[ "${candidate}" != ${prefix}* ]]; then
        echo "ERROR: best candidate does not match requested ratios (${ssr}/${objssr}): ${candidate}" >&2
        return 1
    fi

    rest="${candidate#${prefix}}"
    if [[ "${rest}" == mat* ]]; then
        material_part="${rest%%_*}"
        MATERIAL_HASH="${material_part#mat}"
        BQ_RUN_TAG="${BQ_RUN_TAG:-}"
    elif [[ "${rest}" =~ ^(bqk[^_]+_r[^_]+)_(mat[^_]+)_ ]]; then
        BQ_RUN_TAG="${BASH_REMATCH[1]}"
        material_part="${BASH_REMATCH[2]}"
        MATERIAL_HASH="${material_part#mat}"
    else
        echo "ERROR: cannot parse best candidate: ${candidate}" >&2
        return 1
    fi

    export MATERIAL_HASH
    export BQ_RUN_TAG
}

select_best_material_if_needed() {
    [ "${auto_best_hash}" = "true" ] || return 0
    [ -z "${MATERIAL_HASH:-}" ] || return 0
    [ ${#scenes[@]} -gt 0 ] || return 0

    local report
    local scene_name
    local candidate
    local first_candidate=""

    report="$(find_best_ptpp_report || true)"
    [ -n "${report}" ] || return 0

    for scene_name in "${scenes[@]}"; do
        candidate="$(source_candidate_for_scene "${report}" "${scene_name}")"
        if [ -z "${candidate}" ]; then
            echo "ERROR: no Source Candidate for scene '${scene_name}' in ${report}" >&2
            exit 1
        fi
        if [ -z "${first_candidate}" ]; then
            first_candidate="${candidate}"
        elif [ "${candidate}" != "${first_candidate}" ]; then
            echo "ERROR: scenes use different best candidates in ${report}; pass MATERIAL_HASH/BQ_RUN_TAG explicitly or render one scene at a time." >&2
            echo "  first: ${first_candidate}" >&2
            echo "  ${scene_name}: ${candidate}" >&2
            exit 1
        fi
    done

    apply_source_candidate "${first_candidate}"
    echo "Auto-selected best PT++ candidate from ${report}: ${first_candidate}"
    echo "Auto-selected MATERIAL_HASH=${MATERIAL_HASH}, BQ_RUN_TAG=${BQ_RUN_TAG:-<plain>}"
}

select_best_material_if_needed
set_output_paths

format_time() {
    local seconds=$1
    printf "%dh %dm %ds" $((seconds / 3600)) $((seconds % 3600 / 60)) $((seconds % 60))
}

run_step() {
    local name=$1
    shift
    echo "=== ${name} ==="
    local t0=$SECONDS
    "$@"
    local elapsed=$((SECONDS - t0))
    echo "Elapsed: $(format_time "${elapsed}")"
    echo ""
}

has_frames() {
    local frame_dir=$1
    [ -d "${frame_dir}" ] || return 1
    find "${frame_dir}" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q .
}

needs_dynamic_render() {
    local output_dir=$1
    local scene_name=$2
    [ "${force_render}" = "true" ] || ! has_frames "${output_dir}/${scene_name}/0"
}

check_inputs() {
    local missing=0
    for scene_name in "${scenes[@]}"; do
        if [ "${run_static_reconstruction}" = "true" ] && [ ! -f "${gs_output_dir}/${scene_name}/${exp_name}/cfg_args" ]; then
            echo "ERROR: missing raw Gaussian reconstruction model: ${gs_output_dir}/${scene_name}/${exp_name}/cfg_args" >&2
            missing=1
        fi
        if [ "${run_reconstruction}" = "true" ] && needs_dynamic_render "${dynamic_output_dir}" "${scene_name}"; then
            if [ ! -f "${exp_dir}/${scene_name}/inference.pkl" ]; then
                echo "ERROR: missing reconstruction-resimulation trajectory: ${exp_dir}/${scene_name}/inference.pkl" >&2
                missing=1
            fi
            if [ ! -f "${gs_output_dir}/${scene_name}/${exp_name}/cfg_args" ]; then
                echo "ERROR: missing Gaussian config for reconstruction-resimulation: ${gs_output_dir}/${scene_name}/${exp_name}/cfg_args" >&2
                missing=1
            fi
        fi
    done
    if [ "${missing}" -ne 0 ]; then
        echo "Inputs are incomplete; stop before rendering." >&2
        exit 1
    fi
}

render_reconstruction_videos() {
    for scene_name in "${scenes[@]}"; do
        local model_dir="${gs_output_dir}/${scene_name}/${exp_name}"
        local render_dir="${model_dir}/test/ours_${gs_iterations}/renders"
        local video_path="${gs_video_dir}/${scene_name}/${exp_name}.mp4"

        if [ "${force_render}" = "true" ] || ! has_frames "${render_dir}"; then
            python gs_render.py \
                -s "./data/gaussian_data/${scene_name}" \
                -m "${model_dir}" \
                --iteration "${gs_iterations}"
        fi

        if ! has_frames "${render_dir}"; then
            echo "ERROR: reconstruction frames not found after render: ${render_dir}" >&2
            exit 1
        fi

        python gaussian_splatting/img2video.py \
            --image_folder "${render_dir}" \
            --video_path "${video_path}"
    done

    if [ "${run_recon_integrated}" = "true" ]; then
        python visualize_reconstruction_results.py \
            --model_root "${gs_output_dir}" \
            --video_root "${gs_video_dir}" \
            --scenes "${scenes[@]}" \
            --exp_name "${exp_name}" \
            --iteration "${gs_iterations}"
    fi
}

convert_dynamic_videos() {
    local output_dir=$1
    for scene_name in "${scenes[@]}"; do
        [ -d "${output_dir}/${scene_name}" ] || continue
        for view_dir in "${output_dir}/${scene_name}"/*; do
            [ -d "${view_dir}" ] || continue
            local view_name
            view_name="$(basename "${view_dir}")"
            case "${view_name}" in
                ''|*[!0-9]*) continue ;;
            esac
            if has_frames "${view_dir}"; then
                python gaussian_splatting/img2video.py \
                    --image_folder "${view_dir}" \
                    --video_path "${output_dir}/${scene_name}/${view_name}.mp4"
            fi
        done
    done
}

render_dynamic_frames_if_needed() {
    local output_dir=$1
    local label=$2
    shift 2
    local script_cmd=("$@")
    local needed_scenes=()

    for scene_name in "${scenes[@]}"; do
        if needs_dynamic_render "${output_dir}" "${scene_name}"; then
            needed_scenes+=("${scene_name}")
        fi
    done

    if [ "${#needed_scenes[@]}" -gt 0 ]; then
        run_step "Render ${label} frames" "${script_cmd[@]}" "${ssr}" "${objssr}" "${needed_scenes[@]}"
    else
        echo "${label}: existing frames found; only converting videos."
        echo ""
    fi
}

render_resimulation_videos() {
    export GS_OUTPUT_DIR="${gs_output_dir}"
    export DYNAMIC_OUTPUT_DIR="${dynamic_output_dir}"
    export DYNAMIC_OUTPUT_WHITE_DIR="${dynamic_output_white_dir}"

    render_dynamic_frames_if_needed "${dynamic_output_dir}" "physics reconstruction" bash gs_run_simulate.sh
    run_step "Convert physics reconstruction videos" convert_dynamic_videos "${dynamic_output_dir}"

    if [ "${run_white}" = "true" ] || [ "${run_integrated}" = "true" ]; then
        render_dynamic_frames_if_needed "${dynamic_output_white_dir}" "white-background physics reconstruction" bash gs_run_simulate_white.sh
    fi

    if [ "${run_white}" = "true" ]; then
        run_step "Convert white-background physics reconstruction videos" convert_dynamic_videos "${dynamic_output_white_dir}"
    fi

    if [ "${run_integrated}" = "true" ]; then
        run_step "Create integrated overlay videos" python visualize_render_results.py \
            --surface_sample_ratio "${ssr}" \
            --obj_sample_ratio "${objssr}" \
            --material_hash "${MATERIAL_HASH:-}" \
            --run_tag "${BQ_RUN_TAG:-}" \
            --prediction_dir "${dynamic_output_white_dir}" \
            --scenes "${scenes[@]}"
    fi
}

trap 'echo ""; echo "Failed at line ${LINENO}, exit code: $?" >&2' ERR

export CUDA_VISIBLE_DEVICES="${gpu}"

echo "Scenes: ${scenes[*]}"
echo "Reconstruction models: ${gs_output_dir}"
echo "Raw Gaussian reconstruction videos: ${gs_video_dir}"
echo "Training-action trajectories: ${exp_dir}"
echo "Physics reconstruction videos: ${dynamic_output_dir}"
echo "White/integrated physics reconstruction videos: ${dynamic_output_white_dir}"
echo ""

check_inputs

total_start=$SECONDS

if [ "${run_static_reconstruction}" = "true" ]; then
    run_step "Render raw Gaussian reconstruction videos" render_reconstruction_videos
fi

if [ "${run_reconstruction}" = "true" ]; then
    render_resimulation_videos
fi

total_elapsed=$((SECONDS - total_start))
echo "All done. Total elapsed: $(format_time "${total_elapsed}")"
