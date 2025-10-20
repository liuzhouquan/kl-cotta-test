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

from kl_inverse_cotta import setup_kl_inverse_cotta
from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate_kl_inverse_cotta(description):
    """Evaluate KL Inverse CoTTA on CIFAR-10C dataset."""
    # Parse command line arguments for KL inverse parameters
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("--tau", dest="tau", type=float, default=1.0,
                        help="Temperature parameter for inverse weighting (higher = smoother weighting)")
    parser.add_argument("--eps", dest="eps", type=float, default=1e-8,
                        help="Numerical stability parameter for weighted loss")
    parser.add_argument("--strategy", dest="strategy", type=str, default="exp",
                        choices=["exp", "reciprocal", "linear"],
                        help="Inverse weighting strategy: exp, reciprocal, or linear")
    parser.add_argument("--disable_kl_inverse", action="store_true",
                        help="Disable KL inverse weighting (equivalent to baseline CoTTA)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    
    args = parser.parse_args()
    
    # Set KL inverse parameters
    if args.disable_kl_inverse:
        tau = 0.0  # Disable inverse weighting
        eps = 1e-8
        strategy = "exp"
    else:
        tau = args.tau
        eps = args.eps
        strategy = args.strategy
    
    # Load configuration
    # Ensure config parser sees only config-related args (avoid '--tau', '--eps', '--strategy' etc.)
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--cfg", args.cfg_file] + (args.opts or [])
        load_cfg_fom_args(description)
    finally:
        sys.argv = original_argv

    cfg.defrost()
    cfg.KL_INVERSE.TAU = tau
    cfg.KL_INVERSE.EPS = eps
    cfg.KL_INVERSE.STRATEGY = strategy
    cfg.freeze()
    
    # Configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                           cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    
    # Setup optimizer
    optimizer = setup_optimizer(base_model.parameters())
    
    # Setup KL Inverse CoTTA model
    model = setup_kl_inverse_cotta(
        base_model, 
        optimizer,
        tau=tau,
        eps=eps,
        strategy=strategy,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        mt_alpha=cfg.OPTIM.MT,
        rst_m=cfg.OPTIM.RST,
        ap=cfg.OPTIM.AP
    )
    
    # Log configuration
    if tau == 0.0:
        logger.info("Running baseline CoTTA (KL inverse weighting disabled)")
    else:
        logger.info(f"Running KL-Inverse CoTTA with tau={tau}, eps={eps}, strategy={strategy}")
    
    # Evaluate on each severity and type of corruption in turn
    total_errors = []
    total_forward_count = 0
    total_update_count = 0
    total_kl_sum = 0.0
    total_weight_sum = 0.0
    
    prev_ct = "x0"
    prev_forward_count = 0
    prev_update_count = 0
    prev_kl_sum = 0.0
    prev_weight_sum = 0.0
    
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # Reset model for each corruption type
            if i_c == 0:
                try:
                    model.reset()
                    model.reset_stats()  # Reset efficiency stats
                    prev_forward_count = 0
                    prev_update_count = 0
                    prev_kl_sum = 0.0
                    prev_weight_sum = 0.0
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
            
            # Get detailed stats for this corruption
            efficiency, update_count, forward_count, avg_kl, avg_weight, kl_sum, weight_sum = model.get_detailed_stats()
            
            # Calculate incremental stats for this corruption
            incremental_forward = forward_count - prev_forward_count
            incremental_update = update_count - prev_update_count
            incremental_kl_sum = kl_sum - prev_kl_sum
            incremental_weight_sum = weight_sum - prev_weight_sum
            
            # Accumulate incremental stats
            total_forward_count += incremental_forward
            total_update_count += incremental_update
            total_kl_sum += incremental_kl_sum
            total_weight_sum += incremental_weight_sum
            
            # Update previous values for next iteration
            prev_forward_count = forward_count
            prev_update_count = update_count
            prev_kl_sum = kl_sum
            prev_weight_sum = weight_sum
            
            # Log results for this corruption
            logger.info(f"Corruption: {corruption_type}{severity}, "
                       f"Error: {err:.2%}, "
                       f"Updates: {incremental_update}/{incremental_forward} ({efficiency:.1%}), "
                       f"Avg_KL: {avg_kl:.4f}, "
                       f"Avg_Weight: {avg_weight:.4f}")
    
    # Calculate overall statistics
    overall_error = sum(total_errors) / len(total_errors)
    overall_efficiency = total_update_count / total_forward_count if total_forward_count > 0 else 0.0
    overall_avg_kl = total_kl_sum / total_forward_count if total_forward_count > 0 else 0.0
    overall_avg_weight = total_weight_sum / total_forward_count if total_forward_count > 0 else 0.0
    
    # Log final results
    logger.info("=" * 60)
    if tau == 0.0:
        logger.info("BASELINE CoTTA Results:")
    else:
        logger.info(f"KL-Inverse CoTTA Results (tau={tau}, eps={eps}, strategy={strategy}):")
    logger.info(f"Overall Error: {overall_error:.2%}")
    logger.info(f"Total Efficiency: {overall_efficiency:.1%} ({total_update_count}/{total_forward_count} updates)")
    logger.info(f"Average Error per Corruption: {overall_error:.2%}")
    logger.info(f"Overall Average KL: {overall_avg_kl:.4f}")
    logger.info(f"Overall Average Weight: {overall_avg_weight:.4f}")
    logger.info("=" * 60)


def setup_optimizer(params):
    """Set up optimizer for KL Inverse CoTTA adaptation.

    KL Inverse CoTTA needs an optimizer for test-time entropy minimization.
    In principle, KL Inverse CoTTA could make use of any gradient optimizer.
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
    evaluate_kl_inverse_cotta('KL Inverse CoTTA CIFAR-10-C evaluation.')
