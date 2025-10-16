#!/usr/bin/env python3
"""
Test script to verify all imports work correctly for KL-Gate CoTTA.
This script can be run to check dependencies before running on RunPod.
"""

def test_imports():
    """Test all required imports."""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        print("✓ PyTorch imports successful")
        
        # Test robustbench imports
        from robustbench.data import load_cifar10c
        from robustbench.model_zoo.enums import ThreatModel
        from robustbench.utils import load_model, clean_accuracy
        print("✓ RobustBench imports successful")
        
        # Test local imports
        from conf import cfg, load_cfg_fom_args
        print("✓ Configuration imports successful")
        
        from cotta import (
            get_tta_transforms, update_ema_variables, softmax_entropy,
            collect_params, copy_model_and_optimizer, load_model_and_optimizer, configure_model
        )
        print("✓ CoTTA imports successful")
        
        # Test our new KL-Gate imports
        from kl_gate_cotta import KLGateCoTTA, setup_kl_gate_cotta
        print("✓ KL-Gate CoTTA imports successful")
        
        print("\n🎉 All imports successful! Ready to run on RunPod.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
