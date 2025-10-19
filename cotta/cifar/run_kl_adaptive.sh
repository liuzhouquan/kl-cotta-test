#!/bin/bash

# KL-Adaptive CoTTA Test Script
# Usage examples:
# bash run_kl_adaptive.sh baseline    # Run baseline CoTTA (thr=0.0)
# bash run_kl_adaptive.sh adaptive    # Run KL-Adaptive CoTTA (default params)
# bash run_kl_adaptive.sh custom 0.1 10 0.2 0.8 1.2 100  # Run with custom parameters

export PYTHONPATH=

# Check if we're in the right directory
if [ ! -f "cifar10c_KL_adaptive.py" ]; then
    echo "Error: Please run this script from the cifar directory"
    exit 1
fi

# Parse arguments
MODE=${1:-"adaptive"}
THRESHOLD=${2:-"0.1"}
CHECK_INTERVAL=${3:-"10"}
LOW_THRESHOLD=${4:-"0.2"}
HIGH_THRESHOLD=${5:-"0.8"}
SCALE_FACTOR=${6:-"1.2"}
WARMUP_SAMPLES=${7:-"100"}

case $MODE in
    "baseline")
        echo "Running baseline CoTTA (KL-Adaptive disabled)..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_adaptive.py --cfg cfgs/cifar10/kl_adaptive_cotta.yaml --thr 0.0
        ;;
    "adaptive")
        echo "Running KL-Adaptive CoTTA with default parameters..."
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_adaptive.py --cfg cfgs/cifar10/kl_adaptive_cotta.yaml --thr 0.1 --check_interval 10 --low_threshold 0.2 --high_threshold 0.8 --scale_factor 1.2 --warmup_samples 100
        ;;
    "custom")
        echo "Running KL-Adaptive CoTTA with custom parameters:"
        echo "  Threshold: $THRESHOLD"
        echo "  Check interval: $CHECK_INTERVAL"
        echo "  Low threshold: $LOW_THRESHOLD"
        echo "  High threshold: $HIGH_THRESHOLD"
        echo "  Scale factor: $SCALE_FACTOR"
        echo "  Warmup samples: $WARMUP_SAMPLES"
        CUDA_VISIBLE_DEVICES=0 python cifar10c_KL_adaptive.py --cfg cfgs/cifar10/kl_adaptive_cotta.yaml --thr $THRESHOLD --check_interval $CHECK_INTERVAL --low_threshold $LOW_THRESHOLD --high_threshold $HIGH_THRESHOLD --scale_factor $SCALE_FACTOR --warmup_samples $WARMUP_SAMPLES
        ;;
    "test")
        echo "Testing imports..."
        python test_imports.py
        ;;
    *)
        echo "Usage: $0 {baseline|adaptive|custom [threshold] [check_interval] [low_threshold] [high_threshold] [scale_factor] [warmup_samples]|test}"
        echo "  baseline  - Run baseline CoTTA (thr=0.0)"
        echo "  adaptive  - Run KL-Adaptive CoTTA (default parameters)"
        echo "  custom    - Run KL-Adaptive CoTTA with custom parameters"
        echo "  test      - Test imports"
        echo ""
        echo "Examples:"
        echo "  $0 baseline"
        echo "  $0 adaptive"
        echo "  $0 custom 0.1 10 0.2 0.8 1.2 100"
        echo "  $0 custom 0.05 5 0.1 0.9 1.5 50"
        echo ""
        echo "Parameter explanations:"
        echo "  threshold      - Initial KL threshold (overridden by warmup)"
        echo "  check_interval - Samples between threshold checks"
        echo "  low_threshold  - Low update ratio threshold (decrease KL threshold)"
        echo "  high_threshold - High update ratio threshold (increase KL threshold)"
        echo "  scale_factor   - Scale factor for threshold adjustment"
        echo "  warmup_samples - Number of warmup samples for initial threshold learning"
        exit 1
        ;;
esac
