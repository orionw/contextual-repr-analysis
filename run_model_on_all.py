import glob
import os
import json
import typing
import argparse
import subprocess
import sys
import shutil
import copy

import pandas as pd
import numpy as np

from scripts.create_hdf5_file_from_huggingface_model import load_model_and_write_hdf5


BLACKLIST = [
    "coreference", # skip for now
]


def remove(path: str):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def file_contains_blacklist(string: str):
    """ A helper function to skip tasks we don't want to to for now """
    for skip_task in BLACKLIST:
        if skip_task in string:
            return 
            

def clean_up_dir(run_args: argparse.Namespace, dir_base_path: str, results_file_name: str = "metrics.json"):
    """ A helper function to remove everything except the results file to save space """
    base_path_list = []
    if run_args.n_layers > 1:
        for layer_num in range(run_args.n_layers - 1):
            base_path_list.append(dir_base_path + "_{}".format(layer_num))
    else:
        base_path_list.append(dir_base_path)

    for base_path in base_path_list:
        for file_path in glob.glob(os.path.join(base_path, "*")):
            if results_file_name in file_path:
                continue
            else:
                print("Removing {}".format(file_path))
                remove(file_path)


def get_config_paths() -> typing.Dict[str, typing.Dict[str, str]]:
    """
    As each task uses a different setup, this function gathers the file that contains all the paths needed

    Returns
    -------
    A dictionary containing a string to dictionary mapping, containing the hdf5 file path, 
        the sentences file path and the config file path
    """
    data_paths = {}
    with open(os.path.join("scripts", "task_info.txt"), "r") as fin:
        for line in fin:
            config_reader, hd5_path, sentence_path = line.strip().split(" ")
            if file_contains_blacklist(config_reader):
                continue
            data_paths[config_reader] = {
                "json_file": config_reader,
                "hdf5_path": hd5_path,
                "sentence_path": sentence_path
            }
    return data_paths


def get_run_scripts(model_name: str, location_path: str = "") -> typing.Dict[str, typing.Dict[str, str]]:
    """
    As each task uses a different setup, this function gathers the base generic task configs and fills them
    in with the correct model name, used for running an `allennlp train` command

    Args:
    -----
    model_name: a string that defines the model name used in the experiments

    Returns
    -------
    A dictionary containing a string to dictionary mapping, where that dictionary is the allennlp config dict
    """
    run_configs = {}
    for file_name in glob.glob(os.path.join("generic_config", "*.json")):
        if file_contains_blacklist(file_name):
            continue
        with open(file_name, "r") as fin:
            data = fin.read().replace('\n', '')
            run_configs[file_name.split("/")[-1].replace(".json", "")] = data

    model_name_dict = {"model_name": model_name if location_path == "" else os.path.join(model_name, location_path), "cuda_device": 0} # zero to always use the $CUDA_VISIBLE_DEVICE
    for name, config in run_configs.items():
        run_configs[name] = json.loads(config % model_name_dict)

    return run_configs
        

def run_probes(run_args: argparse.Namespace, config: typing.Dict[str, typing.Dict],
                  all_run_scripts: typing.Dict[str, typing.Dict]):
    """ 
    A wrapper script to train all the models on all the configs
    Creates and saves HDF5 file and deletes it after it's done
    By the end of this script, all probes should be trained and in the folder:
         `run_args.temp_path/{model_name}/{task_name}`

    Args
    ----
    run_args: the arguments of the main wrapping process
    config: a dictionary mapping string names of config files (aka ccg) to the config information with paths
    all_run_scripts: a dictionary mapping string names of config files to a json file to execute with allennlp
    """
    base_output_hdf5 = os.path.join("contextualizers", run_args.model_name, run_args.temp_path)
    if not os.path.isdir(base_output_hdf5):
        os.makedirs(base_output_hdf5)

    config_base_dir = os.path.join("experiment_configs/{model_name}".format(model_name=run_args.model_name))
    if not os.path.isdir(config_base_dir):
        os.makedirs(config_base_dir)

    for name, (config_task) in config.items():
        # create HDF5 file
        config_reader, hd5_path, sentence_path = config_task["json_file"], config_task["hdf5_path"], config_task["sentence_path"]
        output_file_path_hdf5 = os.path.join(base_output_hdf5, hd5_path)
        namespace_dict = {
            "model_class": run_args.model_name,
            "data_path": sentence_path,
            "output_path": output_file_path_hdf5,
            "model_name_or_path": run_args.model_weights,
            "cuda": run_args.cuda,
            "all_layers": run_args.n_layers > 1,
        }
        args = argparse.Namespace(**namespace_dict)
        print("Processing arguments: \n", run_args, args)
        if not os.path.isfile(output_file_path_hdf5):
            load_model_and_write_hdf5(args)
        assert os.path.isfile(output_file_path_hdf5), "did not create hdf5 file for {}".format(name)

        # write experimental config 
        config_json_path = "{base_dir}/{task_name}.json".format(base_dir=config_base_dir, task_name=name)
        with open(config_json_path, "w") as fout:
            json.dump(all_run_scripts[name], fout)

        # create executable command and run probe
        executable_str_list = []
        if run_args.n_layers > 1:
            for layer_num in range(run_args.n_layers - 1):
                execute_str = "allennlp train {config_path} -s {save_path}/{model_name}/{task_name}_{layer_num} --include-package contexteval" \
                                .format(config_path=config_json_path, task_name=name, save_path=run_args.temp_path,
                                         layer_num=layer_num, model_name=run_args.model_name)
                execute_str +=""" --overrides '{"dataset_reader": {"contextualizer": {"layer_num": %s}}, "validation_dataset_reader": {"contextualizer": {"layer_num": %s}}}'""" % (layer_num, layer_num)
                executable_str_list.append(execute_str)
        else:
            execute_str = "allennlp train {config_path} -s {save_path}/{model_name}/{task_name} --include-package contexteval"\
                            .format(config_path=config_json_path, task_name=name, save_path=run_args.temp_path, model_name=run_args.model_name)
            executable_str_list.append(execute_str)

        base_save_path = "{save_path}/{model_name}/{task_name}".format(task_name=name, save_path=run_args.temp_path, model_name=run_args.model_name)
        if run_args.n_layers > 1:
            check_dir = base_save_path + "_{}".format(str(run_args.n_layers - 1))
        else:
            check_dir = base_save_path
        if not os.path.isdir(check_dir):
            for execute_str in executable_str_list:
                process = subprocess.Popen([execute_str], stdout=sys.stdout, stderr=sys.stderr, shell=True).wait()
                if process != 0:
                    raise Exception("Failed to execute {}, return code of {}".format(execute_str, process.returncode))

        # cleanup
        clean_up_dir(run_args, base_save_path)
        os.remove(output_file_path_hdf5)

def gather_results(run_args: argparse.Namespace):
    for file_path in glob.glob(os.path.join(run_args.temp_path, args.model_name, "**", "metrics.json")):
        with open(file_path, "r") as fin:
            metric_results = json.load(fin)
        final_dict = {}
        for key, value in metric_results.items():
            if "test" in key and key[-1] != "3":
                final_dict[key + file_path.split("/")[-2]] = value
    
    with open("final_results.json", "w") as fout:
        json.dump(final_dict, fout)



def get_probes_for_model_all_tasks(run_args: argparse.Namespace):
    """ A wrapper function for the wrapper functions: the all in one function """
    if not os.path.isdir(run_args.temp_path):
        os.makedirs(run_args.temp_path)

    config = get_config_paths()
    all_run_scripts = get_run_scripts(run_args.model_name, run_args.temp_path)
    run_probes(run_args, config, all_run_scripts)
    gather_results(run_args)


def run_all_models_on_all_probes(run_args: argparse.Namespace):
    for model_weights_path in glob.glob(os.path.join(run_args.saved_model_dir, "**", "**", "pytorch_model.bin")):
        print("Running probes for model at {}".format(model_weights_path))
        cur_run_args = copy.deepcopy(run_args)
        cur_run_args.model_weights = "/".join(model_weights_path.split("/")[:-1])
        cur_run_args.temp_path = os.path.join(run_args.temp_path, "-".join(model_weights_path.split("/")[-4:])) # keep other info
        get_probes_for_model_all_tasks(cur_run_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="the model you want to use to generate the embedding")
    parser.add_argument("--temp_path", help="the path to the data you want to embed", default="results")
    parser.add_argument("--saved_model_dir", help="the path to directories of model weights you want to use, loaded from file", default=None)
    parser.add_argument("--cuda", action="store_true", help="use cuda to speed up the embedding process", default=False)
    parser.add_argument("--n_layers", type=int, help="use n layers in probing", default=1)
    parser.add_argument("--model_weights", type=str, help="ignore this, used later", default=None)
    run_args = parser.parse_args()
    run_all_models_on_all_probes(run_args)
