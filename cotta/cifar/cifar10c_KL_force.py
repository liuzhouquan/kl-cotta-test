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

from kl_force_cotta import setup_kl_force_cotta
from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate_kl_force_cotta(description):
    """Evaluate KL-Force CoTTA on CIFAR-10C dataset."""
    # Parse command line arguments for KL-Force parameters
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("--thr", dest="kl_threshold", type=float, default=0.1,
                        help="KL divergence threshold (0.0 = baseline CoTTA, >0.0 = KL-Force enabled)")
    parser.add_argument("--force_interval", dest="force_interval", type=int, default=10,
                        help="Number of consecutive skips before forced update")
    parser.add_argument("--adaptive", dest="adaptive_threshold", action="store_true",
                        help="Enable adaptive threshold adjustment")
    parser.add_argument("--disable_kl_force", action="store_true",
                        help="Disable KL-Force (equivalent to baseline CoTTA)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    
    args = parser.parse_args()
    
    # Set KL-Force parameters
    if args.disable_kl_force:
        kl_threshold = 0.0
        force_update_interval = 10
        adaptive_threshold = False
    else:
        kl_threshold = args.kl_threshold
        force_update_interval = args.force_interval
        adaptive_threshold = args.adaptive_threshold
    
    # Load configuration
    # Ensure config parser sees only config-related args (avoid '--thr', '--force_interval' etc.)
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--cfg", args.cfg_file] + (args.opts or [])
        load_cfg_fom_args(description)
    finally:
        sys.argv = original_argv

    cfg.defrost()
    cfg.KL_FORCE.THRESHOLD = kl_threshold
    cfg.KL_FORCE.FORCE_INTERVAL = force_update_interval
    cfg.KL_FORCE.ADAPTIVE_THRESHOLD = adaptive_threshold
    cfg.freeze()
    
    # Configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                           cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    
    # Setup optimizer
    optimizer = setup_optimizer(base_model.parameters())
    
    # Setup KL-Force CoTTA model
    model = setup_kl_force_cotta(
        base_model, 
        optimizer,
        kl_threshold=kl_threshold,
        force_update_interval=force_update_interval,
        adaptive_threshold=adaptive_threshold,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        mt_alpha=cfg.OPTIM.MT,
        rst_m=cfg.OPTIM.RST,
        ap=cfg.OPTIM.AP
    )
    
    # Log configuration
    if kl_threshold == 0.0:
        logger.info("Running baseline CoTTA (KL-Force disabled)")
    else:
        logger.info(f"Running KL-Force CoTTA with threshold={kl_threshold}, "
                   f"force_interval={force_update_interval}, "
                   f"adaptive={adaptive_threshold}")
    
    # Evaluate on each severity and type of corruption in turn
    total_errors = []
    total_forward_count = 0
    total_update_count = 0
    total_force_update_count = 0
    
    prev_ct = "x0"
    prev_forward_count = 0
    prev_update_count = 0
    prev_force_update_count = 0
    
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # Reset model for each corruption type
            if i_c == 0:
                try:
                    model.reset()
                    model.reset_stats()  # Reset efficiency stats
                    prev_forward_count = 0
                    prev_update_count = 0
                    prev_force_update_count = 0
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
            efficiency, update_count, forward_count, force_update_count, force_ratio = model.get_efficiency_stats()
            
            # Calculate incremental stats for this corruption
            incremental_forward = forward_count - prev_forward_count
            incremental_update = update_count - prev_update_count
            incremental_force_update = force_update_count - prev_force_update_count
            
            # Accumulate incremental stats
            total_forward_count += incremental_forward
            total_update_count += incremental_update
            total_force_update_count += incremental_force_update
            
            # Update previous values for next iteration
            prev_forward_count = forward_count
            prev_update_count = update_count
            prev_force_update_count = force_update_count
            
            # Log results for this corruption
            logger.info(f"Corruption: {corruption_type}{severity}, "
                       f"Error: {err:.2%}, "
                       f"Updates: {incremental_update}/{incremental_forward} ({efficiency:.1%}), "
                       f"Force_Updates: {incremental_force_update}, "
                       f"Force_Ratio: {force_ratio:.1%}")
    
    # Calculate overall statistics
    overall_error = sum(total_errors) / len(total_errors)
    overall_efficiency = total_update_count / total_forward_count if total_forward_count > 0 else 0.0
    overall_force_ratio = total_force_update_count / total_forward_count if total_forward_count > 0 else 0.0
    
    # Log final results
    logger.info("=" * 60)
    if kl_threshold == 0.0:
        logger.info("BASELINE CoTTA Results:")
    else:
        logger.info(f"KL-Force CoTTA Results (threshold={kl_threshold}, "
                   f"force_interval={force_update_interval}, "
                   f"adaptive={adaptive_threshold}):")
    logger.info(f"Overall Error: {overall_error:.2%}")
    logger.info(f"Total Efficiency: {overall_efficiency:.1%} ({total_update_count}/{total_forward_count} updates)")
    logger.info(f"Total Force Updates: {total_force_update_count}")
    logger.info(f"Force Update Ratio: {overall_force_ratio:.1%}")
    logger.info(f"Average Error per Corruption: {overall_error:.2%}")
    logger.info("=" * 60)


def setup_optimizer(params):
    """Set up optimizer for KL-Force CoTTA adaptation.

    KL-Force CoTTA needs an optimizer for test-time entropy minimization.
    In principle, KL-Force CoTTA could make use of any gradient optimizer.
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
    evaluate_kl_force_cotta('KL-Force CoTTA CIFAR-10-C evaluation.')
