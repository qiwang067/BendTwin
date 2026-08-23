import os
import csv
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--scenes", nargs="*", default=None, help="Scenes to process; process all scenes if omitted")
args = parser.parse_args()

base_path = "./data/different_types"
output_path = "./data/render_eval_data"
CONTROLLER_NAME = "hand"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


existDir(output_path)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if args.scenes is not None and case_name not in args.scenes:
            continue
        if not os.path.exists(f"{base_path}/{case_name}"):
            continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    
        # Create the directory for the case
        existDir(f"{output_path}/{case_name}")
        existDir(f"{output_path}/{case_name}/mask")
        for i in range(3):
            # Copy the original RGB image
            os.system(
                f"cp -r {base_path}/{case_name}/color {output_path}/{case_name}/"
            )
            # Copy only the object mask image
            # Get the mask path for the image
            with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            existDir(f"{output_path}/{case_name}/mask/{i}")
            os.system(f"cp -r {base_path}/{case_name}/mask/{i}/{obj_idx}/* {output_path}/{case_name}/mask/{i}/")
        
        # Copy the split.json
        os.system(f"cp {base_path}/{case_name}/split.json {output_path}/{case_name}/")
