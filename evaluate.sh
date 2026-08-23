#!/usr/bin/env bash

ssr=${1:-0.3}
objssr=${2:-1.0}
shift 2 || true

scene_tag=""
extra_result_args=()
run_tag_arg=()
if [ -n "${BQ_RUN_TAG:-}" ]; then
    run_tag_arg=(--run_tag "${BQ_RUN_TAG}")
fi

build_scene_tag() {
    local scene_count=$#
    if [ ${scene_count} -eq 0 ]; then
        printf ''
        return
    fi

    if [ ${scene_count} -eq 1 ]; then
        printf '%s' "$1"
        return
    fi

    local first_scene=$1
    local digest
    digest=$(printf '%s\n' "$@" | sort | sha1sum | cut -c1-12)
    printf '%s__plus%d_%s' "${first_scene}" "$((scene_count - 1))" "${digest}"
}

# Prefer explicitly supplied scene arguments; otherwise fall back to the SCENES environment variable.
if [ $# -gt 0 ]; then
    scenes=("$@")
elif [ -n "${SCENES}" ]; then
    read -r -a scenes <<< "${SCENES}"
else
    scenes=()
fi

if [ ${#scenes[@]} -gt 0 ]; then
    scene_tag=$(build_scene_tag "${scenes[@]}")
    extra_result_args+=(--scene_tag "${scene_tag}")
fi

if [ ${#scenes[@]} -gt 0 ]; then
    python evaluate_chamfer.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash "${MATERIAL_HASH:-}" "${run_tag_arg[@]}" --scenes "${scenes[@]}" "${extra_result_args[@]}"
    python evaluate_track.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash "${MATERIAL_HASH:-}" "${run_tag_arg[@]}" --scenes "${scenes[@]}" "${extra_result_args[@]}"
    python gaussian_splatting/evaluate_render.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash "${MATERIAL_HASH:-}" "${run_tag_arg[@]}" --scenes "${scenes[@]}" "${extra_result_args[@]}"
else
    python evaluate_chamfer.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash "${MATERIAL_HASH:-}" "${run_tag_arg[@]}"
    python evaluate_track.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash "${MATERIAL_HASH:-}" "${run_tag_arg[@]}"
    python gaussian_splatting/evaluate_render.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash "${MATERIAL_HASH:-}" "${run_tag_arg[@]}"
fi
merge_run_tag_arg=()
if [ -n "${BQ_RUN_TAG:-}" ]; then
    merge_run_tag_arg=(--run_tag "${BQ_RUN_TAG}")
fi

python results/merge_per_frame_results.py --surface_sample_ratio ${ssr} --obj_sample_ratio ${objssr} --material_hash "${MATERIAL_HASH:-}" "${merge_run_tag_arg[@]}" "${extra_result_args[@]}"
