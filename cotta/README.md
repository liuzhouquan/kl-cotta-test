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

### 4. KL-Force CoTTA (Forced Updates)
- **File**: `kl_force_cotta.py`, `cifar10c_KL_force.py`
- **Script**: `run_kl_force.sh`
- **Description**: Forced update mechanism to prevent learning stagnation
- **Rationale & Hypothesis**:
  - **Core Idea**: Address the static threshold problem by forcing updates after consecutive skips
  - **Problem Addressed**: Original KL-Gate can get stuck in high-disagreement states, completely halting learning
  - **Assumptions**:
    - Even high-disagreement samples might contain useful information after some time
    - Periodic forced updates can help escape local minima
    - Adaptive threshold adjustment based on recent KL values is more intelligent than fixed threshold
  - **Expected Benefits**:
    - Prevents complete learning stagnation
    - Maintains computational efficiency most of the time
    - Adaptive variant can learn optimal thresholds from data
- **Key Features**:
  - Forces updates after consecutive skips reach a threshold
  - Optional adaptive threshold adjustment based on recent KL values
  - Maintains efficiency while preventing complete learning halt
- **Parameters**:
  - `--thr`: Initial KL threshold
  - `--force_interval`: Number of consecutive skips before forced update
  - `--adaptive`: Enable adaptive threshold adjustment
- **Usage**: `bash run_kl_force.sh baseline|force|adaptive|custom [threshold] [interval] [adaptive]`

### 5. KL-Adaptive CoTTA (Adaptive Threshold)
- **File**: `kl_adaptive_cotta.py`, `cifar10c_KL_adaptive.py`
- **Script**: `run_kl_adaptive.sh`
- **Description**: Adaptive threshold adjustment based on update ratio and warmup phase
- **Rationale & Hypothesis**:
  - **Core Idea**: Learn optimal threshold from data and adapt it based on update patterns
  - **Warmup Phase Hypothesis**: Initial samples can provide a good estimate of typical KL divergence distribution
  - **Adaptive Adjustment Hypothesis**: 
    - Low update ratio (< 20%) suggests threshold is too strict → decrease threshold
    - High update ratio (> 80%) suggests threshold is too loose → increase threshold
  - **Assumptions**:
    - Optimal update ratio should be in a reasonable range (e.g., 20%-80%)
    - KL divergence distribution is relatively stable within a corruption type
    - Dynamic threshold adjustment can better adapt to different data characteristics
  - **Expected Benefits**:
    - Data-driven threshold selection (no manual tuning)
    - Automatic adaptation to different corruption types
    - Balanced learning efficiency and effectiveness
- **Key Features**:
  - Warmup phase learns initial threshold from first 100 samples
  - Dynamically adjusts threshold based on update frequency
  - Balances update efficiency and learning effectiveness
- **Parameters**:
  - `--thr`: Initial threshold (overridden by warmup)
  - `--check_interval`: Samples between threshold checks
  - `--low_threshold`: Low update ratio threshold (decrease KL threshold)
  - `--high_threshold`: High update ratio threshold (increase KL threshold)
  - `--scale_factor`: Scale factor for threshold adjustment
  - `--warmup_samples`: Number of warmup samples for initial threshold learning
- **Usage**: `bash run_kl_adaptive.sh baseline|adaptive|custom [threshold] [check_interval] [low_threshold] [high_threshold] [scale_factor] [warmup_samples]`
  
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

# KL-Force CoTTA (Forced Updates)
bash run_kl_force.sh baseline         # Baseline CoTTA
bash run_kl_force.sh force            # Basic forced updates
bash run_kl_force.sh adaptive         # Adaptive threshold
bash run_kl_force.sh custom 0.1 5 True  # Custom parameters

# KL-Adaptive CoTTA (Adaptive Threshold)
bash run_kl_adaptive.sh baseline      # Baseline CoTTA
bash run_kl_adaptive.sh adaptive      # Default adaptive settings
bash run_kl_adaptive.sh custom 0.1 10 0.2 0.8 1.2 100  # Custom parameters
```


## Enhanced CoTTA Variants Summary

The enhanced CoTTA variants address the static threshold limitation in the original KL-Gate mechanism, which can cause learning stagnation when the model encounters high KL divergence samples. Here's a comparison of the approaches:

| Method | Core Idea | Advantages | Best For |
|--------|-----------|------------|----------|
| **KL-Gate** | Hard gating with static threshold | Simple, efficient | Known optimal threshold |
| **KL-Gate-Rev** | Reverse logic (skip high agreement) | Filters noise, saves computation | Noisy environments |
| **KL-Regu** | Soft weighting based on KL | Smooth adaptation, no stagnation | Fine-grained control needed |
| **KL-Force** | Forced updates after skips | Prevents complete stagnation | Environments prone to stagnation |
| **KL-Adaptive** | Dynamic threshold adjustment | Self-tuning, data-driven | Unknown optimal threshold |

### Key Improvements:
1. **Prevents Learning Stagnation**: All variants address the core issue of static thresholds
2. **Maintains Efficiency**: Most variants preserve the computational benefits of selective updates
3. **Data-Driven Adaptation**: KL-Adaptive learns optimal thresholds from data
4. **Flexible Control**: Multiple parameters allow fine-tuning for different scenarios

### Recommended Usage:
- **KL-Gate-Rev**: When you want to filter out potentially noisy consistent predictions
- **KL-Regu**: When you need smooth, continuous adaptation without hard decisions
- **KL-Force**: When you're concerned about complete learning halt
- **KL-Adaptive**: When you want automatic threshold optimization

## Citation
If you find these enhanced CoTTA variants useful, please cite the original CoTTA paper:
```bibtex
@inproceedings{wang2022continual,
  title={Continual Test-Time Domain Adaptation},
  author={Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin},
  booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgement 
+ Original CoTTA implementation: [official](https://github.com/qinenergy/cotta)
+ Robustbench for evaluation framework: [official](https://github.com/RobustBench/robustbench)
