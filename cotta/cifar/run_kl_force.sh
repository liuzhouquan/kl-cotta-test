#!/bin/bash

# KL-Force CoTTA Test Script
# Usage examples:
# bash run_kl_force.sh baseline    # Run baseline CoTTA (thr=0.0)
# bash run_kl_force.sh force       # Run KL-Force CoTTA (thr=0.1, interval=10)
# bash run_kl_force.sh adaptive    # Run KL-Force CoTTA with adaptive threshold
# bash run_kl_force.sh custom 0.1 5 True  # Run with custom parameters

export PYTHONPATH=

# Check if we're in the right directory
if [ ! -f "cifar10c_KL_force.py" ]; then
    echo "Error: Please run this script from the cifar directory"
    exit 1
fi

# Parse arguments
MODE=${1:-"force"}
THRESHOLD=${2:-"0.1"}
INTERVAL=${3:-"10"}
ADAPTIVE=${4:-"False"}

case $MODE in
    "baseline")
        echo "Running baseline CoTTA (KL-Force disabled)..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_force.py --cfg cfgs/cifar10/kl_force_cotta.yaml --thr 0.0
        ;;
    "force")
        echo "Running KL-Force CoTTA with threshold=0.1, interval=10..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_force.py --cfg cfgs/cifar10/kl_force_cotta.yaml --thr 0.1 --force_interval 10
        ;;
    "adaptive")
        echo "Running KL-Force CoTTA with adaptive threshold..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_force.py --cfg cfgs/cifar10/kl_force_cotta.yaml --thr 0.1 --force_interval 10 --adaptive
        ;;
    "custom")
        echo "Running KL-Force CoTTA with custom parameters: threshold=$THRESHOLD, interval=$INTERVAL, adaptive=$ADAPTIVE..."
        if [ "$ADAPTIVE" = "True" ]; then
            CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_force.py --cfg cfgs/cifar10/kl_force_cotta.yaml --thr $THRESHOLD --force_interval $INTERVAL --adaptive
        else
            CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_force.py --cfg cfgs/cifar10/kl_force_cotta.yaml --thr $THRESHOLD --force_interval $INTERVAL
        fi
        ;;
    "test")
        echo "Testing imports..."
        python test_imports.py
        ;;
    *)
        echo "Usage: $0 {baseline|force|adaptive|custom [threshold] [interval] [adaptive]|test}"
        echo "  baseline  - Run baseline CoTTA (thr=0.0)"
        echo "  force     - Run KL-Force CoTTA (thr=0.1, interval=10)"
        echo "  adaptive  - Run KL-Force CoTTA with adaptive threshold"
        echo "  custom    - Run KL-Force CoTTA with custom parameters"
        echo "  test      - Test imports"
        echo ""
        echo "Examples:"
        echo "  $0 baseline"
        echo "  $0 force"
        echo "  $0 adaptive"
        echo "  $0 custom 0.1 5 True"
        echo "  $0 custom 0.05 8 False"
        exit 1
        ;;
esac
