#!/bin/bash
#SBATCH --job-name=eap-ig-all-tests
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-unknown}"
echo "[info] host=$(hostname)"
echo "[info] submit_dir=${SLURM_SUBMIT_DIR:-$PWD}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Keep caches on node-local storage when possible.
export TMPBASE="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${TMPBASE}/pip-cache-${USER}"
export HF_HOME="${TMPBASE}/hf-${USER}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}"

# Enable all heavy test gates.
export EAP_TL_PARITY_DEVICE=cuda
export EAP_RUN_TL_PARITY=1
export EAP_RUN_MODEL_AUDIT_SMOKE=1
export EAP_RUN_MODEL_AUDIT_CERT=1
export EAP_RUN_MODEL_AUDIT_PARITY=1
export EAP_MODEL_AUDIT_DEVICE=cuda

# Use a per-job virtualenv to guarantee pytest + dependencies exist.
VENV_DIR="${TMPBASE}/eap-ig-venv-${SLURM_JOB_ID:-manual}"
python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[test]"

python --version
python -m pytest -q
