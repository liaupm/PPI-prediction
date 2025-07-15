La #!/bin/sh

if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: $0 input_dir output_dir"
  exit 1
fi

# Work around XLA issue causing compilation time to greatly increase
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
# Do not use unified memory, to be faster:
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95


AF3_ENV_PATH="/home/jovyan/miniforge3/envs/af3_env/bin"

$AF3_ENV_PATH/python ~/alphafold3/run_alphafold.py --input_dir=$1 --model_dir=/home/jovyan/alphafold3/weights --output_dir=$2 --norun_data_pipeline --save_embeddings