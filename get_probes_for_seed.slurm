#! /usr/bin/env bash
python3 run_model_on_all.py roberta --cuda --n_layers=6 --saved_model_dir "/net/nfs.corp/allennlp/orionw/output/$1"