#!/bin/bash
#SBATCH --job-name=eap-ig-targeted
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-unknown}"
echo "[info] host=$(hostname)"
echo "[info] submit_dir=${SLURM_SUBMIT_DIR:-$PWD}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export TMPBASE="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${TMPBASE}/pip-cache-${USER}"
export HF_HOME="${TMPBASE}/hf-${USER}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}"

export EAP_RUN_MODEL_AUDIT_CERT=1
export EAP_RUN_MODEL_AUDIT_PARITY=1
export EAP_MODEL_AUDIT_DEVICE=cuda

./.venv311/bin/python --version
./.venv311/bin/python -m pytest -q \
  tests/test_model_audit_math.py::test_completeness_axiom[gpt2-smoke] \
  tests/test_model_audit_math.py::test_exact_patching_faithfulness[gpt2-smoke] \
  tests/test_model_audit_certification.py::test_hooked_and_bridge_parity_on_certification_models[gpt2-cert] \
  -vv
