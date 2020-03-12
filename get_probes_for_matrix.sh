#! /usr/bin/env bash
declare -a random_num=(42 544460 801760 428404 257803) # also includes 42
for ((i=0; i<5; i++)); # random seeds
do
    sbatch -J "${random_num[i]}-probes" --gpus=1 -w allennlp-server1 -p allennlp_hipri get_probes_for_seed.slurm ${random_num[i]}
done