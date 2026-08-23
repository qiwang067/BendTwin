ssr=${1:-0.3}
objssr=${2:-1.0}
shift 2
if [ $# -gt 0 ]; then
    scenes=("$@")
elif [ -n "${SCENES}" ]; then
    read -r -a scenes <<< "${SCENES}"
else
    scenes=("double_stretch_sloth")
fi

run_suffix=""
run_tag_arg=()
if [ -n "${BQ_RUN_TAG:-}" ]; then
    run_suffix="_${BQ_RUN_TAG}"
    run_tag_arg=(--run_tag "${BQ_RUN_TAG}")
fi

material_root_suffix=""
if [ -n "${MATERIAL_HASH:-}" ]; then
    material_root_suffix="/mat${MATERIAL_HASH#mat}"
fi

output_dir="${DYNAMIC_OUTPUT_DIR:-./gaussian_output_dynamic_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}}"

# views=("0" "1" "2")
views=("0")

# scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
#         "double_stretch_sloth" "double_stretch_zebra"
#         "rope_double_hand"
#         "single_clift_cloth_1" "single_clift_cloth_3"
#         "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
#         "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
#         "single_push_rope" "single_push_rope_1" "single_push_rope_4"
#         "single_push_sloth"
#         "weird_package")

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

resolve_material_paths() {
    local scene_name=$1
    MATERIAL_EXP_DIR="./experiments_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}/${scene_name}"
    MATERIAL_INFERENCE_PATH="${MATERIAL_EXP_DIR}/inference.pkl"
    MATERIAL_MODEL_DIR="./gaussian_output_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}/${scene_name}/${exp_name}"
    MATERIAL_MODEL_CFG="${MATERIAL_MODEL_DIR}/cfg_args"
}

for scene_name in "${scenes[@]}"; do
    resolve_material_paths "${scene_name}"

    if [ ! -d "${MATERIAL_EXP_DIR}" ]; then
        echo "Skip: ${MATERIAL_EXP_DIR} does not exist"
        continue
    fi

    if [ ! -f "${MATERIAL_INFERENCE_PATH}" ]; then
        echo "Skip: ${MATERIAL_INFERENCE_PATH} does not exist"
        continue
    fi

    if [ ! -f "${MATERIAL_MODEL_CFG}" ]; then
        echo "Skip: ${MATERIAL_MODEL_CFG} does not exist"
        continue
    fi

    python gs_render_dynamics.py \
        -s ./data/gaussian_data/${scene_name} \
        -m "${MATERIAL_MODEL_DIR}" \
        --name ${scene_name} \
        --output_dir "${output_dir}" \
        --surface_sample_ratio ${ssr} \
        --obj_sample_ratio ${objssr} \
        "${run_tag_arg[@]}"

    if [ $? -ne 0 ]; then
        echo "Skip: dynamic rendering failed for ${scene_name}"
        continue
    fi

    for view_name in "${views[@]}"; do
        if [ ! -d "${output_dir}/${scene_name}/${view_name}" ]; then
            echo "Skip: ${output_dir}/${scene_name}/${view_name} does not exist"
            continue
        fi
        # Convert images to video
        python gaussian_splatting/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

done
