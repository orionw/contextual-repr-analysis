import os
import json
import glob
import argparse
import typing

import pandas as pd


def gather_paths(args: argparse.Namespace):
    """ Used to create the data folders needed for the task """
    list_of_json = []
    for file_name in os.listdir(args.dir_with_config):
        print(file_name)
        with open(os.path.join(args.dir_with_config, file_name), "r") as fin:
            list_of_json.append((file_name, json.load(fin)))

    name_and_path_list = []
    for (file_name, file_json) in list_of_json:
        root_folder = "/".join(file_json["train_data_path"].split("/")[:-1])
        name_and_path_list.append({
            "train": file_json["train_data_path"],
            "validation": file_json["validation_data_path"],
            "test": file_json["test_data_path"],
            "dataset": file_name,
            "root_folder": root_folder
        })
        if not os.path.isdir(os.path.join("..", root_folder)):
            os.makedirs(os.path.join("..", root_folder))

    results = pd.DataFrame(name_and_path_list)

    results.to_csv("name_and_paths.csv")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_with_config", help="a string containing the path to the config needed to gather the paths")
    args = parser.parse_args()
    gather_paths(args)