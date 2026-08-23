ssr=${1:-0.3}
objssr=${2:-1.0}
gpu=${3:-0}
shift 3 || true
if [ $# -gt 0 ]; then
    scenes=("$@")
elif [ -n "${SCENES}" ]; then
    read -r -a scenes <<< "${SCENES}"
else
    scenes=("double_stretch_sloth")
fi

run_suffix=""
if [ -n "${BQ_RUN_TAG:-}" ]; then
    run_suffix="_${BQ_RUN_TAG}"
fi

material_root_suffix=""
if [ -n "${MATERIAL_HASH:-}" ]; then
    material_root_suffix="/mat${MATERIAL_HASH}"
fi

output_dir="./gaussian_output_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}"
output_video_dir="./gaussian_output_video_ssr${ssr}_objssr${objssr}${run_suffix}${material_root_suffix}"

export CUDA_VISIBLE_DEVICES=${gpu}

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

python ./gaussian_splatting/generate_interp_poses.py

# Iterate over each folder
for scene_name in "${scenes[@]}"; do
    echo "Processing: $scene_name"

    # Training
    python gs_train.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} \
        --iterations 10000 \
        --lambda_depth 0.001 \
        --lambda_normal 0.0 \
        --lambda_anisotropic 0.0 \
        --lambda_seg 1.0 \
        --use_masks \
        --isotropic \
        --gs_init_opt 'hybrid' \
        --disable_viewer \
        --disable_viewer

    # Rendering
    python gs_render.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} \

    # Convert images to video
    python gaussian_splatting/img2video.py \
        --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
        --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
done
