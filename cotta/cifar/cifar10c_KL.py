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

from kl_gate_cotta import setup_kl_gate_cotta
from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate_kl_gate_cotta(description):
    """Evaluate KL-Gate CoTTA on CIFAR-10C dataset."""
    # Parse command line arguments for KL threshold
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("--thr", dest="kl_threshold", type=float, default=0.1,
                        help="KL divergence threshold (0.0 = baseline CoTTA, >0.0 = KL-Gate enabled)")
    parser.add_argument("--disable_kl_gate", action="store_true",
                        help="Disable KL-Gate (equivalent to thr=0.0)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    
    args = parser.parse_args()
    
    # Set KL threshold
    if args.disable_kl_gate:
        kl_threshold = 0.0
    else:
        kl_threshold = args.kl_threshold
    
    # Load configuration
    # Ensure config parser sees only config-related args (avoid '--thr' etc.)
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--cfg", args.cfg_file] + (args.opts or [])
        load_cfg_fom_args(description)
    finally:
        sys.argv = original_argv
    
    # Configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                           cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    
    # Setup optimizer
    optimizer = setup_optimizer(base_model.parameters())
    
    # Setup KL-Gate CoTTA model
    model = setup_kl_gate_cotta(
        base_model, 
        optimizer,
        kl_threshold=kl_threshold,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        mt_alpha=cfg.OPTIM.MT,
        rst_m=cfg.OPTIM.RST,
        ap=cfg.OPTIM.AP
    )
    
    # Log configuration
    if kl_threshold == 0.0:
        logger.info("Running baseline CoTTA (KL-Gate disabled)")
    else:
        logger.info(f"Running KL-Gate CoTTA with threshold: {kl_threshold}")
    
    # Evaluate on each severity and type of corruption in turn
    total_errors = []
    total_forward_count = 0
    total_update_count = 0
    
    prev_ct = "x0"
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # Reset model for each corruption type
            if i_c == 0:
                try:
                    model.reset()
                    model.reset_stats()  # Reset efficiency stats
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
            efficiency, update_count, forward_count = model.get_efficiency_stats()
            total_forward_count += forward_count
            total_update_count += update_count
            
            # Log results for this corruption
            logger.info(f"Corruption: {corruption_type}{severity}, "
                       f"Error: {err:.2%}, "
                       f"Updates: {update_count}/{forward_count} ({efficiency:.1%})")
    
    # Calculate overall statistics
    overall_error = sum(total_errors) / len(total_errors)
    overall_efficiency = total_update_count / total_forward_count if total_forward_count > 0 else 0.0
    
    # Log final results
    logger.info("=" * 60)
    if kl_threshold == 0.0:
        logger.info("BASELINE CoTTA Results:")
    else:
        logger.info(f"KL-Gate CoTTA Results (threshold={kl_threshold}):")
    logger.info(f"Overall Error: {overall_error:.2%}")
    logger.info(f"Total Efficiency: {overall_efficiency:.1%} ({total_update_count}/{total_forward_count} updates)")
    logger.info(f"Average Error per Corruption: {overall_error:.2%}")
    logger.info("=" * 60)


def setup_optimizer(params):
    """Set up optimizer for KL-Gate CoTTA adaptation.

    KL-Gate CoTTA needs an optimizer for test-time entropy minimization.
    In principle, KL-Gate CoTTA could make use of any gradient optimizer.
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
    evaluate_kl_gate_cotta('KL-Gate CoTTA CIFAR-10-C evaluation.')
