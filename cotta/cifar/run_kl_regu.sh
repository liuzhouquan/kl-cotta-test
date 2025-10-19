#!/bin/bash

# KL Regularized CoTTA Test Script
# Usage examples:
# bash run_kl_regu.sh baseline    # Run baseline CoTTA (tau=0.0)
# bash run_kl_regu.sh kl_regu     # Run KL-Regu CoTTA (tau=1.0)
# bash run_kl_regu.sh custom 0.5  # Run KL-Regu CoTTA with custom tau
# bash run_kl_regu.sh custom 0.5 1e-6  # Run KL-Regu CoTTA with custom tau and eps

export PYTHONPATH=

# Check if we're in the right directory
if [ ! -f "cifar10c_KL_regu.py" ]; then
    echo "Error: Please run this script from the cifar directory"
    exit 1
fi

# Parse arguments
MODE=${1:-"kl_regu"}
TAU=${2:-"1.0"}
EPS=${3:-"1e-8"}

case $MODE in
    "baseline")
        echo "Running baseline CoTTA (KL regularization disabled)..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_regu.py --cfg cfgs/cifar10/kl_regu_cotta.yaml --tau 0.0
        ;;
    "kl_regu")
        echo "Running KL-Regu CoTTA with tau=1.0, eps=1e-8..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_regu.py --cfg cfgs/cifar10/kl_regu_cotta.yaml --tau 1.0 --eps 1e-8
        ;;
    "custom")
        echo "Running KL-Regu CoTTA with custom tau=$TAU, eps=$EPS..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_regu.py --cfg cfgs/cifar10/kl_regu_cotta.yaml --tau $TAU --eps $EPS
        ;;
    "test")
        echo "Testing imports..."
        python test_imports.py
        ;;
    *)
        echo "Usage: $0 {baseline|kl_regu|custom [tau] [eps]|test}"
        echo "  baseline  - Run baseline CoTTA (tau=0.0)"
        echo "  kl_regu   - Run KL-Regu CoTTA (tau=1.0, eps=1e-8)"
        echo "  custom    - Run KL-Regu CoTTA with custom tau and eps"
        echo "  test      - Test imports"
        echo ""
        echo "Examples:"
        echo "  $0 baseline"
        echo "  $0 kl_regu"
        echo "  $0 custom 0.5"
        echo "  $0 custom 0.5 1e-6"
        exit 1
        ;;
esac
