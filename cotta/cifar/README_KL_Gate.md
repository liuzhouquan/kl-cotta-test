# KL-Gate CoTTA Implementation

This implementation adds a KL divergence gate to the CoTTA (Continual Test-Time Adaptation) method for CIFAR-10C evaluation.

## Overview

The KL-Gate CoTTA method introduces a threshold-based mechanism to skip model updates when the KL divergence between teacher and student models exceeds a specified threshold. This is based on the hypothesis that high KL divergence indicates the teacher model is not trustworthy, so updates should be skipped to avoid harmful adaptations.

## Files Created

- `kl_gate_cotta.py` - KL-Gate CoTTA implementation
- `cifar10c_KL.py` - Main test script with baseline and KL-Gate modes
- `cfgs/cifar10/kl_gate_cotta.yaml` - Configuration file
- `run_kl_gate.sh` - Convenient run script
- `test_imports.py` - Import verification script

## Usage

### Quick Start

```bash
# Test imports first
bash run_kl_gate.sh test

# Run baseline CoTTA (original method)
bash run_kl_gate.sh baseline

# Run KL-Gate CoTTA with default threshold (0.1)
bash run_kl_gate.sh kl_gate

# Run KL-Gate CoTTA with custom threshold
bash run_kl_gate.sh custom 0.05
```

### Direct Python Usage

```bash
# Baseline CoTTA (KL-Gate disabled)
python cifar10c_KL.py --cfg cfgs/cifar10/kl_gate_cotta.yaml --thr 0.0

# KL-Gate CoTTA with threshold 0.1
python cifar10c_KL.py --cfg cfgs/cifar10/kl_gate_cotta.yaml --thr 0.1

# KL-Gate CoTTA with custom threshold
python cifar10c_KL.py --cfg cfgs/cifar10/kl_gate_cotta.yaml --thr 0.05

# Disable KL-Gate (alternative syntax)
python cifar10c_KL.py --cfg cfgs/cifar10/kl_gate_cotta.yaml --disable_kl_gate
```

## Key Features

### 1. KL Divergence Gate
- Computes KL divergence between student and teacher model outputs
- Skips updates when KL divergence > threshold
- Default threshold: 0.1 (configurable via command line)

### 2. Performance Tracking
- `forward_count`: Total number of forward passes
- `update_count`: Number of actual updates performed
- `efficiency`: update_count/forward_count ratio

### 3. Baseline Comparison
- `--thr 0.0`: Runs original CoTTA for baseline comparison
- Same tracking metrics for fair comparison
- Identical codebase ensures consistent evaluation

## Output Format

```
[timestamp] KL-Gate CoTTA Results (threshold=0.1):
Corruption: gaussian_noise5, Error: 12.34%, Updates: 45/50 (90.0%)
Corruption: shot_noise5, Error: 15.67%, Updates: 38/50 (76.0%)
...
============================================================
KL-Gate CoTTA Results (threshold=0.1):
Overall Error: 14.23%
Total Efficiency: 82.5% (825/1000 updates)
Average Error per Corruption: 14.23%
============================================================
```

## Configuration

The `kl_gate_cotta.yaml` file contains all configuration parameters:

```yaml
MODEL:
  ADAPTATION: kl_gate_cotta
  ARCH: Standard
  EPISODIC: False
TEST:
  BATCH_SIZE: 200
CORRUPTION:
  DATASET: cifar10
  NUM_EX: 10000
  SEVERITY: [5]
  TYPE: [gaussian_noise, shot_noise, ...]
OPTIM:
  METHOD: Adam
  STEPS: 1
  LR: 1e-3
  MT: 0.999
  RST: 0.01
  AP: 0.92
KL_GATE:
  THRESHOLD: 0.1  # Default threshold
```

## Expected Results

- **Accuracy**: Should be comparable to original CoTTA
- **Efficiency**: Should show computational savings (efficiency < 100%)
- **Speed**: Faster execution due to skipped updates
- **Robustness**: Better handling of untrustworthy teacher predictions

## RunPod Setup

For RunPod deployment:

1. Install dependencies:
```bash
pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu113
pip install addict==2.4.0 attrs==21.2.0 imagecorruptions==1.1.2 imageio==2.10.1
pip install numpy==1.19.5 scikit-image==0.18.3 scikit-learn==1.0.1 scipy==1.7.1
pip install tqdm==4.56.2 yacs==0.1.8
pip install git+https://github.com/robustbench/robustbench@v0.1#egg=robustbench
```

2. Test imports:
```bash
python test_imports.py
```

3. Run experiments:
```bash
bash run_kl_gate.sh baseline
bash run_kl_gate.sh kl_gate
```

## Troubleshooting

- **Import errors**: Run `python test_imports.py` to verify dependencies
- **CUDA errors**: Ensure CUDA is available and `CUDA_VISIBLE_DEVICES` is set correctly
- **Memory issues**: Reduce `BATCH_SIZE` in the config file
- **Slow execution**: This is expected for the first run as models and data are downloaded
