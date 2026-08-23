import os
import csv

base_path = "./data/different_types"

os.system("rm -f timer.log")

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        if shape_prior.lower() == "true":
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category} --shape_prior"
            )
        else:
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category}"
            )
