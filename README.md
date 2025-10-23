# Enhanced CoTTA: KL-Gate Based Continual Test-Time Adaptation
This repository contains enhanced variants of CoTTA with KL-Gate mechanism for selective updates during test-time adaptation.

## Background
The original CoTTA method performs continuous updates on all test samples, which can be computationally expensive and potentially noisy. This work introduces a KL-Gate mechanism that selectively updates the model based on the KL divergence between teacher and student predictions. After initial experiments with the basic KL-Gate, several enhanced variants were developed to address the static threshold limitation and improve adaptation effectiveness.

## Enhanced CoTTA Variants
This repository includes several enhanced variants of CoTTA with KL-Gate mechanism:

### 1. KL-Gate CoTTA (Original)
- **File**: `kl_gate_cotta.py`, `cifar10c_KL.py`
- **Script**: `run_kl_gate.sh`
- **Description**: Original KL-Gate mechanism - skips updates when KL > threshold (high disagreement)
- **Rationale & Hypothesis**:
  - **Core Idea**: When teacher and student predictions disagree significantly (high KL divergence), the update might be unreliable or noisy
  - **Assumption**: High KL divergence indicates either model uncertainty or potential noise in the sample
  - **Expected Benefit**: Skip potentially harmful updates, maintain model stability
  - **Limitation**: Static threshold can cause learning stagnation if model gets stuck in high-disagreement state
- **Usage**: `bash run_kl_gate.sh baseline|kl_gate|custom [threshold]`

### 2. KL-Gate-Rev CoTTA (Reverse Logic)
- **File**: `kl_gate_cotta_rev.py`, `cifar10c_KL_rev.py`
- **Script**: `run_kl_gate_rev.sh`
- **Description**: Reverse KL-Gate logic - skips updates when KL < threshold (high agreement)
- **Rationale & Hypothesis**:
  - **Core Idea**: When teacher and student agree well (low KL divergence), the update might be redundant or potentially noisy
  - **Assumption**: High agreement samples might contain less informative signal for adaptation
  - **Expected Benefits**: 
    - Save computation by skipping redundant updates
    - Filter out potentially noisy consistent predictions
    - Focus learning on samples where teacher-student disagreement indicates room for improvement
  - **Hypothesis**: This approach may reduce overfitting and improve generalization
- **Usage**: `bash run_kl_gate_rev.sh baseline|kl_gate|custom [threshold]`

### 3. KL-Regu CoTTA (Soft Weighting)
- **File**: `kl_regu_cotta.py`, `cifar10c_KL_regu.py`
- **Script**: `run_kl_regu.sh`
- **Description**: Soft weighting based on KL divergence instead of hard gating
- **Rationale & Hypothesis**:
  - **Core Idea**: Instead of binary skip/update decisions, use continuous weights based on KL divergence
  - **Mathematical Formulation**: 
    - Sample weights: `w_i = exp(-KL_i / τ)`
    - Weighted loss: `L = (Σ_i w_i L_i) / (Σ_i w_i + ε)`
  - **Assumptions**:
    - All samples contain some useful information, but with varying reliability
    - Smooth weighting is more robust than hard decisions
    - Temperature parameter τ controls the sensitivity of weighting
  - **Expected Benefits**:
    - Prevents learning stagnation (no complete skipping)
    - Automatically down-weights potentially noisy samples
    - Maintains continuous learning while filtering noise
- **Parameters**:
  - `--tau`: Temperature parameter (higher = smoother weighting)
  - `--eps`: Numerical stability parameter
- **Usage**: `bash run_kl_regu.sh baseline|kl_regu|custom [tau] [eps]`

### 4. KL-Inverse CoTTA (Inverse Weighting)
- **File**: `kl_inverse_cotta.py`, `cifar10c_KL_inverse.py`
- **Script**: `run_kl_inverse.sh`
- **Description**: Inverse weighting mechanism that emphasizes samples with high KL divergence
- **Rationale & Hypothesis**:
  - **Core Idea**: Instead of down-weighting high KL samples (like KL-Regu), emphasize them with higher weights
  - **Mathematical Formulation**: 
    - Exponential: `w_i = exp(KL_i / τ)`
    - Reciprocal: `w_i = 1 / (KL_i + ε)`
    - Linear: `w_i = 1 + KL_i / τ`
    - Weighted loss: `L = (Σ_i w_i L_i) / (Σ_i w_i + ε)`
  - **Assumptions**:
    - High disagreement samples contain more informative signal for adaptation
    - Emphasizing difficult samples can improve model robustness
    - Different weighting strategies may work better for different scenarios
  - **Expected Benefits**:
    - Focus learning on challenging samples where teacher-student disagree
    - Improve adaptation to difficult test-time conditions
    - Multiple weighting strategies provide flexibility
- **Key Features**:
  - Three weighting strategies: exponential, reciprocal, and linear
  - Configurable temperature parameter for controlling emphasis strength
  - Numerical stability with epsilon parameter
  - Comprehensive statistics tracking for monitoring adaptation behavior
- **Parameters**:
  - `--tau`: Temperature parameter (higher = stronger emphasis on high KL)
  - `--eps`: Numerical stability parameter
  - `--strategy`: Weighting strategy (exp, reciprocal, linear)
- **Usage**: `bash run_kl_inverse.sh baseline|inverse_exp|inverse_reciprocal|inverse_linear|custom [tau] [strategy] [eps]`
  
## Tasks
+ CIFAR10 -> CIFAR10C (standard)

## Prerequisite
Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate cotta 
```

## Experiments
### Enhanced CoTTA Variants on CIFAR10C
```bash
cd cifar

# Original KL-Gate CoTTA
bash run_kl_gate.sh baseline      # Baseline CoTTA (no gating)
bash run_kl_gate.sh kl_gate       # KL-Gate CoTTA (threshold=0.1)
bash run_kl_gate.sh custom 0.05   # Custom threshold

# KL-Gate-Rev CoTTA (Reverse Logic)
bash run_kl_gate_rev.sh baseline      # Baseline CoTTA
bash run_kl_gate_rev.sh kl_gate       # KL-Gate-Rev (threshold=0.1)
bash run_kl_gate_rev.sh custom 0.05   # Custom threshold

# KL-Regu CoTTA (Soft Weighting)
bash run_kl_regu.sh baseline      # Baseline CoTTA
bash run_kl_regu.sh kl_regu       # KL-Regu (tau=1.0)
bash run_kl_regu.sh custom 0.5    # Custom tau

# KL-Inverse CoTTA (Inverse Weighting)
bash run_kl_inverse.sh baseline           # Baseline CoTTA
bash run_kl_inverse.sh inverse_exp        # KL-Inverse with exp strategy (tau=1.0)
bash run_kl_inverse.sh inverse_reciprocal # KL-Inverse with reciprocal strategy (tau=1.0)
bash run_kl_inverse.sh inverse_linear     # KL-Inverse with linear strategy (tau=1.0)
bash run_kl_inverse.sh custom 2.0 exp     # Custom tau and strategy
bash run_kl_inverse.sh custom 2.0 reciprocal 1e-6  # Custom parameters
```


## Enhanced CoTTA Variants Summary

The enhanced CoTTA variants address the static threshold limitation in the original KL-Gate mechanism, which can cause learning stagnation when the model encounters high KL divergence samples. Here's a comparison of the approaches:

| Method | Core Idea | Advantages | Best For |
|--------|-----------|------------|----------|
| **KL-Gate** | Hard gating with static threshold | Simple, efficient | Known optimal threshold |
| **KL-Gate-Rev** | Reverse logic (skip high agreement) | Filters noise, saves computation | Noisy environments |
| **KL-Regu** | Soft weighting based on KL | Smooth adaptation, no stagnation | Fine-grained control needed |
| **KL-Inverse** | Inverse weighting emphasizing high KL | Focuses on challenging samples | Difficult test conditions |

### Key Improvements:
1. **Prevents Learning Stagnation**: All variants address the core issue of static thresholds
2. **Maintains Efficiency**: Most variants preserve the computational benefits of selective updates
3. **Flexible Weighting**: KL-Regu and KL-Inverse provide continuous weighting options
4. **Multiple Strategies**: KL-Inverse offers three different weighting approaches

### Recommended Usage:
- **KL-Gate-Rev**: When you want to filter out potentially noisy consistent predictions
- **KL-Regu**: When you need smooth, continuous adaptation with down-weighting of high KL samples
- **KL-Inverse**: When you want to emphasize challenging samples with high teacher-student disagreement

## Acknowledgement 
+ Original CoTTA implementation: [official](https://github.com/qinenergy/cotta)
+ Robustbench for evaluation framework: [official](https://github.com/RobustBench/robustbench)