#!/bin/bash

# KL Inverse CoTTA Test Script
# Usage examples:
# bash run_kl_inverse.sh baseline                    # Run baseline CoTTA (tau=0.0)
# bash run_kl_inverse.sh inverse_exp                # Run KL-Inverse CoTTA with exp strategy (tau=1.0)
# bash run_kl_inverse.sh inverse_reciprocal         # Run KL-Inverse CoTTA with reciprocal strategy (tau=1.0)
# bash run_kl_inverse.sh inverse_linear             # Run KL-Inverse CoTTA with linear strategy (tau=1.0)
# bash run_kl_inverse.sh custom 2.0 exp             # Run KL-Inverse CoTTA with custom tau and strategy
# bash run_kl_inverse.sh custom 2.0 reciprocal 1e-6 # Run KL-Inverse CoTTA with custom tau, strategy and eps

export PYTHONPATH=

# Check if we're in the right directory
if [ ! -f "cifar10c_KL_inverse.py" ]; then
    echo "Error: Please run this script from the cifar directory"
    exit 1
fi

# Parse arguments
MODE=${1:-"inverse_exp"}
TAU=${2:-"1.0"}
STRATEGY=${3:-"exp"}
EPS=${4:-"1e-8"}

case $MODE in
    "baseline")
        echo "Running baseline CoTTA (KL inverse weighting disabled)..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_inverse.py --cfg cfgs/cifar10/kl_inverse_cotta.yaml --disable_kl_inverse
        ;;
    "inverse_exp")
        echo "Running KL-Inverse CoTTA with exp strategy, tau=1.0, eps=1e-8..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_inverse.py --cfg cfgs/cifar10/kl_inverse_cotta.yaml --tau 1.0 --eps 1e-8 --strategy exp
        ;;
    "inverse_reciprocal")
        echo "Running KL-Inverse CoTTA with reciprocal strategy, tau=1.0, eps=1e-8..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_inverse.py --cfg cfgs/cifar10/kl_inverse_cotta.yaml --tau 1.0 --eps 1e-8 --strategy reciprocal
        ;;
    "inverse_linear")
        echo "Running KL-Inverse CoTTA with linear strategy, tau=1.0, eps=1e-8..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_inverse.py --cfg cfgs/cifar10/kl_inverse_cotta.yaml --tau 1.0 --eps 1e-8 --strategy linear
        ;;
    "custom")
        echo "Running KL-Inverse CoTTA with custom tau=$TAU, strategy=$STRATEGY, eps=$EPS..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_inverse.py --cfg cfgs/cifar10/kl_inverse_cotta.yaml --tau $TAU --eps $EPS --strategy $STRATEGY
        ;;
    "test")
        echo "Testing imports..."
        python test_imports.py
        ;;
    *)
        echo "Usage: $0 {baseline|inverse_exp|inverse_reciprocal|inverse_linear|custom [tau] [strategy] [eps]|test}"
        echo "  baseline           - Run baseline CoTTA (tau=0.0)"
        echo "  inverse_exp        - Run KL-Inverse CoTTA with exp strategy (tau=1.0)"
        echo "  inverse_reciprocal - Run KL-Inverse CoTTA with reciprocal strategy (tau=1.0)"
        echo "  inverse_linear     - Run KL-Inverse CoTTA with linear strategy (tau=1.0)"
        echo "  custom             - Run KL-Inverse CoTTA with custom parameters"
        echo "  test               - Test imports"
        echo ""
        echo "Weighting Strategies:"
        echo "  exp        - Exponential inverse: weights = exp(kl_div / tau)"
        echo "  reciprocal - Reciprocal inverse: weights = 1 / (kl_div + eps)"
        echo "  linear     - Linear inverse: weights = 1 + kl_div / tau"
        echo ""
        echo "Examples:"
        echo "  $0 baseline"
        echo "  $0 inverse_exp"
        echo "  $0 inverse_reciprocal"
        echo "  $0 inverse_linear"
        echo "  $0 custom 2.0 exp"
        echo "  $0 custom 2.0 reciprocal 1e-6"
        echo "  $0 custom 5.0 linear"
        exit 1
        ;;
esac
