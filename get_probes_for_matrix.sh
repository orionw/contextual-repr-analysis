#! /usr/bin/env bash
declare -a paths=(801760/CoLA/checkpoint-670 42/MNLI/checkpoint-24544 257803/MRPC/checkpoint-174 428404/QNLI/checkpoint-4914 257803/QQP/checkpoint-22744 428404/RTE/checkpoint-160 801760/SST-2/checkpoint-4216 257803/STS-B/checkpoint-360 42/WNLI/checkpoint-10)
for ((i=0; i<${#paths[@]}; i++)); # random seeds
do
    sbatch -J "${paths[i]}-probes" --gpus=1 -w allennlp-server1 -p allennlp_hipri get_probes_for_seed.slurm ${paths[i]}
done