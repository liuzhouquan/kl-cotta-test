import logging
import argparse
import sys
import os

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

from kl_adaptive_cotta import setup_kl_adaptive_cotta
from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate_kl_adaptive_cotta(description):
    """Evaluate KL-Adaptive CoTTA on CIFAR-10C dataset."""
    # Parse command line arguments for KL-Adaptive parameters
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("--thr", dest="kl_threshold", type=float, default=0.1,
                        help="Initial KL divergence threshold (will be overridden by warmup)")
    parser.add_argument("--check_interval", dest="check_interval", type=int, default=10,
                        help="Number of samples between threshold checks")
    parser.add_argument("--low_threshold", dest="low_update_threshold", type=float, default=0.2,
                        help="Low update ratio threshold for decreasing KL threshold")
    parser.add_argument("--high_threshold", dest="high_update_threshold", type=float, default=0.8,
                        help="High update ratio threshold for increasing KL threshold")
    parser.add_argument("--scale_factor", dest="scale_factor", type=float, default=1.2,
                        help="Scale factor for threshold adjustment")
    parser.add_argument("--warmup_samples", dest="warmup_samples", type=int, default=100,
                        help="Number of warmup samples for initial threshold learning")
    parser.add_argument("--disable_adaptive", action="store_true",
                        help="Disable adaptive threshold (use fixed threshold)")
    parser.add_argument("--disable_kl_adaptive", action="store_true",
                        help="Disable KL-Adaptive (equivalent to baseline CoTTA)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    
    args = parser.parse_args()
    
    # Set KL-Adaptive parameters
    if args.disable_kl_adaptive:
        kl_threshold = 0.0
        check_interval = 10
        low_update_threshold = 0.2
        high_update_threshold = 0.8
        scale_factor = 1.2
        warmup_samples = 100
        adaptive_threshold = False
    else:
        kl_threshold = args.kl_threshold
        check_interval = args.check_interval
        low_update_threshold = args.low_update_threshold
        high_update_threshold = args.high_update_threshold
        scale_factor = args.scale_factor
        warmup_samples = args.warmup_samples
        adaptive_threshold = not args.disable_adaptive
    
    # Load configuration
    # Ensure config parser sees only config-related args
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--cfg", args.cfg_file] + (args.opts or [])
        load_cfg_fom_args(description)
    finally:
        sys.argv = original_argv

    cfg.defrost()
    cfg.KL_ADAPTIVE.THRESHOLD = kl_threshold
    cfg.KL_ADAPTIVE.CHECK_INTERVAL = check_interval
    cfg.KL_ADAPTIVE.LOW_UPDATE_THRESHOLD = low_update_threshold
    cfg.KL_ADAPTIVE.HIGH_UPDATE_THRESHOLD = high_update_threshold
    cfg.KL_ADAPTIVE.SCALE_FACTOR = scale_factor
    cfg.KL_ADAPTIVE.WARMUP_SAMPLES = warmup_samples
    cfg.KL_ADAPTIVE.ADAPTIVE_THRESHOLD = adaptive_threshold
    cfg.freeze()
    
    # Configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                           cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    
    # Setup optimizer
    optimizer = setup_optimizer(base_model.parameters())
    
    # Setup KL-Adaptive CoTTA model
    model = setup_kl_adaptive_cotta(
        base_model, 
        optimizer,
        kl_threshold=kl_threshold,
        check_interval=check_interval,
        low_update_threshold=low_update_threshold,
        high_update_threshold=high_update_threshold,
        scale_factor=scale_factor,
        warmup_samples=warmup_samples,
        adaptive_threshold=adaptive_threshold,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        mt_alpha=cfg.OPTIM.MT,
        rst_m=cfg.OPTIM.RST,
        ap=cfg.OPTIM.AP
    )
    
    # Log configuration
    if kl_threshold == 0.0:
        logger.info("Running baseline CoTTA (KL-Adaptive disabled)")
    else:
        logger.info(f"Running KL-Adaptive CoTTA with:")
        logger.info(f"  Initial threshold: {kl_threshold}")
        logger.info(f"  Check interval: {check_interval}")
        logger.info(f"  Update thresholds: [{low_update_threshold}, {high_update_threshold}]")
        logger.info(f"  Scale factor: {scale_factor}")
        logger.info(f"  Warmup samples: {warmup_samples}")
        logger.info(f"  Adaptive threshold: {adaptive_threshold}")
    
    # Evaluate on each severity and type of corruption in turn
    total_errors = []
    total_forward_count = 0
    total_update_count = 0
    total_threshold_adjustments = 0
    
    prev_ct = "x0"
    prev_forward_count = 0
    prev_update_count = 0
    prev_threshold_adjustments = 0
    
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # Reset model for each corruption type
            if i_c == 0:
                try:
                    model.reset()
                    model.reset_stats()  # Reset efficiency stats
                    prev_forward_count = 0
                    prev_update_count = 0
                    prev_threshold_adjustments = 0
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            
            # Load corruption data
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            
            # Evaluate model
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            err = 1. - acc
            total_errors.append(err)
            
            # Get efficiency stats for this corruption
            efficiency, update_count, forward_count, threshold_adjustments, warmup_ratio = model.get_efficiency_stats()
            
            # Calculate incremental stats for this corruption
            incremental_forward = forward_count - prev_forward_count
            incremental_update = update_count - prev_update_count
            incremental_threshold_adjustments = threshold_adjustments - prev_threshold_adjustments
            
            # Accumulate incremental stats
            total_forward_count += incremental_forward
            total_update_count += incremental_update
            total_threshold_adjustments += incremental_threshold_adjustments
            
            # Update previous values for next iteration
            prev_forward_count = forward_count
            prev_update_count = update_count
            prev_threshold_adjustments = threshold_adjustments
            
            # Log results for this corruption
            logger.info(f"Corruption: {corruption_type}{severity}, "
                       f"Error: {err:.2%}, "
                       f"Updates: {incremental_update}/{incremental_forward} ({efficiency:.1%}), "
                       f"Threshold_Adjustments: {incremental_threshold_adjustments}, "
                       f"Warmup_Ratio: {warmup_ratio:.1%}")
    
    # Calculate overall statistics
    overall_error = sum(total_errors) / len(total_errors)
    overall_efficiency = total_update_count / total_forward_count if total_forward_count > 0 else 0.0
    
    # Log final results
    logger.info("=" * 60)
    if kl_threshold == 0.0:
        logger.info("BASELINE CoTTA Results:")
    else:
        logger.info(f"KL-Adaptive CoTTA Results:")
        logger.info(f"  Initial threshold: {kl_threshold}")
        logger.info(f"  Check interval: {check_interval}")
        logger.info(f"  Update thresholds: [{low_update_threshold}, {high_update_threshold}]")
        logger.info(f"  Scale factor: {scale_factor}")
        logger.info(f"  Warmup samples: {warmup_samples}")
        logger.info(f"  Adaptive threshold: {adaptive_threshold}")
    logger.info(f"Overall Error: {overall_error:.2%}")
    logger.info(f"Total Efficiency: {overall_efficiency:.1%} ({total_update_count}/{total_forward_count} updates)")
    logger.info(f"Total Threshold Adjustments: {total_threshold_adjustments}")
    logger.info(f"Average Error per Corruption: {overall_error:.2%}")
    logger.info("=" * 60)


def setup_optimizer(params):
    """Set up optimizer for KL-Adaptive CoTTA adaptation.

    KL-Adaptive CoTTA needs an optimizer for test-time entropy minimization.
    In principle, KL-Adaptive CoTTA could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    training, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate_kl_adaptive_cotta('KL-Adaptive CoTTA CIFAR-10-C evaluation.')
