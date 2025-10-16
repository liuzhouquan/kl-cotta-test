#!/bin/bash

# KL-Gate CoTTA Test Script
# Usage examples:
# bash run_kl_gate.sh baseline    # Run baseline CoTTA (thr=0.0)
# bash run_kl_gate.sh kl_gate     # Run KL-Gate CoTTA (thr=0.1)
# bash run_kl_gate.sh custom 0.05 # Run KL-Gate CoTTA with custom threshold

export PYTHONPATH=

# Check if we're in the right directory
if [ ! -f "cifar10c_KL.py" ]; then
    echo "Error: Please run this script from the cifar directory"
    exit 1
fi

# Parse arguments
MODE=${1:-"kl_gate"}
THRESHOLD=${2:-"0.1"}

case $MODE in
    "baseline")
        echo "Running baseline CoTTA (KL-Gate disabled)..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL.py --cfg cfgs/cifar10/kl_gate_cotta.yaml --thr 0.0
        ;;
    "kl_gate")
        echo "Running KL-Gate CoTTA with threshold 0.1..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL.py --cfg cfgs/cifar10/kl_gate_cotta.yaml --thr 0.1
        ;;
    "custom")
        echo "Running KL-Gate CoTTA with custom threshold $THRESHOLD..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL.py --cfg cfgs/cifar10/kl_gate_cotta.yaml --thr $THRESHOLD
        ;;
    "test")
        echo "Testing imports..."
        python test_imports.py
        ;;
    *)
        echo "Usage: $0 {baseline|kl_gate|custom [threshold]|test}"
        echo "  baseline  - Run baseline CoTTA (thr=0.0)"
        echo "  kl_gate   - Run KL-Gate CoTTA (thr=0.1)"
        echo "  custom    - Run KL-Gate CoTTA with custom threshold"
        echo "  test      - Test imports"
        exit 1
        ;;
esac
