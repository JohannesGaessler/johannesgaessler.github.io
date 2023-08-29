#! /bin/sh
#SBATCH --cpus-per-gpu=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID

srun --gpus=1 --cpus-per-gpu=1 python run_with_preset.py ppl_in.yml --model models/opt/llama_2-7b-$1.gguf

