import os
import glob

base_path = "./data/different_types"
output_path = "./data/different_types_human_mask"

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    existDir(f"{output_path}/{case_name}")
    # Process to get the whole human mask for the video

    TEXT_PROMPT = "human"
    camera_num = 3
    assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num

    for camera_idx in range(camera_num):
        print(f"Processing {case_name} camera {camera_idx}")
        os.system(
            f"python ./data_process/segment_util_video.py --output_path {output_path}/{case_name} --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}"
        )
        os.system(f"rm -rf {base_path}/{case_name}/tmp_data")