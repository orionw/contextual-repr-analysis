import glob
import os
import json
import typing
import argparse
import subprocess
import sys

import pandas as pd
import numpy as np

from scripts.create_hdf5_file_from_huggingface_model import load_model_and_write_hdf5


BLACKLIST = [
    "coreference", # skip for now
]


def file_contains_blacklist(string: str):
    """ A helper function to skip tasks we don't want to to for now """
    for skip_task in BLACKLIST:
        if skip_task in string:
            return True
    return False

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


def get_run_scripts(model_name: str) -> typing.Dict[str, typing.Dict[str, str]]:
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

    model_name_dict = {"model_name": model_name}
    for name, config in run_configs.items():
        run_configs[name] = json.loads(config % model_name_dict)

    return run_configs
        

def generate_embeddings(run_args: argparse.Namespace, config: typing.Dict[str, typing.Dict]):
    """ 
    A wrapper script to generate the hdf5 files for all the tasks we're doing.  Calls `load_model_and_write_hdf5`
    By the end of this script, all hdf5 files should be written the the `run_args.temp_path`

    Args
    ----
    run_args: the arguments of the main wrapping process
    config: a dictionary mapping string names of config files (aka ccg) to the config information with paths
    """
    base_output_hdf5 = os.path.join("contextualizers", run_args.model_name)
    if not os.path.isdir(base_output_hdf5):
        os.makedirs(base_output_hdf5)

    for name, (config_task) in config.items():
        config_reader, hd5_path, sentence_path = config_task["json_file"], config_task["hdf5_path"], config_task["sentence_path"]
        output_file_path = os.path.join(base_output_hdf5, hd5_path)
        if os.path.isfile(output_file_path) and not run_args.overwrite:
            continue
        namespace_dict = {
            "model_name": run_args.model_name,
            "data_path": sentence_path,
            "output_path": output_file_path,
            "model_weights": run_args.model_weights,
            "cuda": run_args.cuda,
            "all_layers": run_args.n_layers > 1
        }
        args = argparse.Namespace(**namespace_dict)
        print("Processing arguments: \n", run_args, args)
        load_model_and_write_hdf5(args)
        assert os.path.isfile(output_file_path), "did not create hdf5 file for {}".format(name)


def train_models(run_args: argparse.Namespace, config: typing.Dict[str, typing.Dict],
                  all_run_scripts: typing.Dict[str, typing.Dict]):
    """ 
    A wrapper script to train all the models on all the configs
    By the end of this script, all probes should be trained and in the folder `run_args.temp_path/{task_name}`

    Args
    ----
    run_args: the arguments of the main wrapping process
    config: a dictionary mapping string names of config files (aka ccg) to the config information with paths
    all_run_scripts: a dictionary mapping string names of config files to a json file to execute with allennlp
    """
    config_base_dir = os.path.join("experiment_configs/{model_name}".format(model_name=run_args.model_name))
    if not os.path.isdir(config_base_dir):
        os.makedirs(config_base_dir)

    for name, (config_task) in config.items():
        # write experimental config 
        config_json_path = "{base_dir}/{task_name}.json".format(base_dir=config_base_dir, task_name=name)
        with open(config_json_path, "w") as fout:
            json.dump(all_run_scripts[name], fout)
        # create executable command
        executable_str_list = []
        if run_args.n_layers > 1:
            for layer_num in range(run_args.n_layers):
                execute_str = "allennlp train {config_path} -s {save_path}/{model_name}/{task_name}_{layer_num} --include-package contexteval" \
                                .format(config_path=config_json_path, task_name=name, save_path=run_args.temp_path,
                                         layer_num=layer_num, model_name=run_args.model_name)
                execute_str +=""" --overrides '{"dataset_reader": {"contextualizer": {"layer_num": %s}}, "validation_dataset_reader": {"contextualizer": {"layer_num": %s}}}'""" % (layer_num, layer_num)
                executable_str_list.append(execute_str)
        else:
            execute_str = "allennlp train {config_path} -s {save_path}/{model_name}/{task_name} --include-package contexteval"\
                            .format(config_path=config_json_path, task_name=name, save_path=run_args.temp_path, model_name=run_args.model_name)
            executable_str_list.append(execute_str)
        for execute_str in executable_str_list:
            process = subprocess.Popen([execute_str], stdout=sys.stdout, stderr=sys.stderr, shell=True).wait()
            if process != 0:
                raise Exception("Failed to execute {}, return code of {}".format(execute_str, process.returncode))


def evaluate_models(run_args: argparse.Namespace, config: typing.Dict[str, typing.Dict],
                  all_run_scripts: typing.Dict[str, typing.Dict]):
    """ 
    A wrapper script to evaluate all the trained models on all the configs
    By the end of this script, all probes should be evaluated and in the folder `run_args.temp_path/{task_name}`
    NOTE: Don't need to evaluate since it does this in the regular train function, leaving the function here just in case I need it

    Args
    ----
    run_args: the arguments of the main wrapping process
    config: a dictionary mapping string names of config files (aka ccg) to the config information with paths
    all_run_scripts: a dictionary mapping string names of config files to a json file to execute with allennlp
    """
    for name, (config_task) in config.items():
        # create executable command
        executable_str_list = []
        eval_file = all_run_scripts[name]["test_data_path"]
        if run_args.n_layers > 1:
            for layer_num in range(run_args.n_layers):
                execute_str = """allennlp evaluate {save_path}/{model_name}/{task_name}_{layer_num}/model.tar.gz  {eval_file} 
                --cuda-device 0 --include-package contexteval 2>&1 | tee {save_path}/{task_name}_{layer_num}/evaluation.log"""\
                .format(task_name=name, save_path=run_args.temp_path, layer_num=layer_num, eval_file=eval_file, model_name=run_args.model_name) 
                execute_str = execute_str.replace("\n", "")              
                executable_str_list.append(execute_str)
        else:
            execute_str = """allennlp evaluate {save_path}/{model_name}/{task_name}/model.tar.gz {eval_file} 
                --cuda-device 0 --include-package contexteval 2>&1 | tee {save_path}/{task_name}/evaluation.log"""\
                .format(task_name=name, save_path=run_args.temp_path, eval_file=eval_file, model_name=run_args.model_name) 
            execute_str = execute_str.replace("\n", "")              
            executable_str_list.append(execute_str)
            
        for execute_str in executable_str_list:
            process = subprocess.Popen([execute_str], stdout=sys.stdout, stderr=sys.stderr, shell=True).wait()
            if process != 0:
                raise Exception("Failed to execute {}, return code of {}".format(execute_str, process.returncode))


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
    all_run_scripts = get_run_scripts(run_args.model_name)
    # TODO: make this one by one, since we can't batch them (too much space reqs)
    # generate_embeddings(run_args, config)
    # train_models(run_args, config, all_run_scripts)
    gather_results(run_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="the model you want to use to generate the embedding")
    parser.add_argument("--temp_path", help="the path to the data you want to embed", default="results")
    parser.add_argument("--model_weights", help="the model weights you want to use, loaded from file", default=None)
    parser.add_argument("--cuda", action="store_true", help="use cuda to speed up the embedding process", default=False)
    parser.add_argument("--n_layers", type=int, help="use n layers in probing", default=1)
    parser.add_argument("--overwrite", action="store_true", help="re-make each embedding file, even if present", default=False)
    run_args = parser.parse_args()
    get_probes_for_model_all_tasks(run_args)
