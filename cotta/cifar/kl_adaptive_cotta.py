from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import logging

from cotta import (
    get_tta_transforms, update_ema_variables, softmax_entropy,
    collect_params, copy_model_and_optimizer, load_model_and_optimizer, configure_model
)

logger = logging.getLogger(__name__)


class KLAdaptiveCoTTA(nn.Module):
    """KL-Adaptive CoTTA adapts a model by entropy minimization during testing.
    
    Uses adaptive threshold adjustment based on update ratio and warmup phase.
    Automatically learns initial threshold from warmup samples and adjusts based on update frequency.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, 
                 kl_threshold=0.1, check_interval=10, low_update_threshold=0.2, high_update_threshold=0.8,
                 scale_factor=1.2, warmup_samples=100, adaptive_threshold=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        # KL-Adaptive parameters
        self.kl_threshold = kl_threshold
        self.check_interval = check_interval
        self.low_update_threshold = low_update_threshold
        self.high_update_threshold = high_update_threshold
        self.scale_factor = scale_factor
        self.warmup_samples = warmup_samples
        self.adaptive_threshold = adaptive_threshold
        
        self.forward_count = 0
        self.update_count = 0
        
        # Tracking variables for adaptive threshold
        self.warmup_kl_values = []
        self.last_update_count = 0
        self.threshold_adjustments = 0
        self.original_threshold = kl_threshold
        self.warmup_completed = False
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def compute_kl_divergence(self, student_logits, teacher_logits):
        """Compute KL divergence between student and teacher logits."""
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # KL(student || teacher) = sum(student * log(student/teacher))
        kl_div = F.kl_div(
            F.log_softmax(student_logits, dim=1), 
            teacher_probs, 
            reduction='batchmean'
        )
        return kl_div

    def adjust_threshold(self):
        """Adjust threshold based on recent update ratio."""
        if not self.adaptive_threshold:
            return
            
        # Calculate recent update ratio
        recent_updates = self.update_count - self.last_update_count
        recent_forwards = self.check_interval
        update_ratio = recent_updates / recent_forwards
        
        old_threshold = self.kl_threshold
        
        if update_ratio < self.low_update_threshold:
            # Too few updates, decrease threshold (increase updates)
            self.kl_threshold *= self.scale_factor
            self.threshold_adjustments += 1
            logger.info(f"KL-Adaptive: Low update ratio ({update_ratio:.2%}): "
                       f"threshold {old_threshold:.4f} -> {self.kl_threshold:.4f}")
        elif update_ratio > self.high_update_threshold:
            # Too many updates, increase threshold (decrease updates)
            self.kl_threshold /= self.scale_factor
            self.threshold_adjustments += 1
            logger.info(f"KL-Adaptive: High update ratio ({update_ratio:.2%}): "
                       f"threshold {old_threshold:.4f} -> {self.kl_threshold:.4f}")
        else:
            logger.info(f"KL-Adaptive: Update ratio ({update_ratio:.2%}) in normal range, "
                       f"threshold unchanged: {self.kl_threshold:.4f}")
        
        self.last_update_count = self.update_count

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        # Increment forward count
        self.forward_count += 1
        
        # Get student (current model) outputs
        student_outputs = self.model(x)
        
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        for i in range(N):
            outputs_ = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_)
            
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0) < self.ap:
            teacher_outputs = torch.stack(outputs_emas).mean(0)
        else:
            teacher_outputs = standard_ema
        
        # Compute KL divergence between student and teacher
        kl_div = self.compute_kl_divergence(student_outputs, teacher_outputs)
        
        # Warmup phase: collect KL values and set initial threshold
        if self.forward_count <= self.warmup_samples:
            self.warmup_kl_values.append(kl_div.item())
            if self.forward_count == self.warmup_samples:
                # Set initial threshold as average of warmup KL values
                self.kl_threshold = sum(self.warmup_kl_values) / len(self.warmup_kl_values)
                self.warmup_completed = True
                logger.info(f"KL-Adaptive warmup completed: initial threshold = {self.kl_threshold:.4f} "
                           f"(from {len(self.warmup_kl_values)} samples)")
        
        # Check if threshold adjustment is needed (after warmup)
        if (self.warmup_completed and 
            self.forward_count > self.warmup_samples and 
            (self.forward_count - self.warmup_samples) % self.check_interval == 0):
            self.adjust_threshold()
        
        # Decide whether to update based on KL threshold
        should_update = (self.kl_threshold > 0 and kl_div <= self.kl_threshold)
        
        if should_update:
            # Perform normal CoTTA update
            self.update_count += 1
            
            # Student update
            loss = (softmax_entropy(student_outputs, teacher_outputs)).mean(0) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Teacher update
            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
            
            # Stochastic restore
            if True:
                for nm, m in self.model.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape) < self.rst).float().cuda() 
                            with torch.no_grad():
                                p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        else:
            # Skip update
            if self.warmup_completed:
                logger.info(f"KL-Adaptive skip: KL={kl_div:.4f} > threshold={self.kl_threshold:.4f}")
            return teacher_outputs.detach()
        
        return teacher_outputs

    def get_efficiency_stats(self):
        """Get efficiency statistics."""
        if self.forward_count == 0:
            return 0.0, 0, 0, 0, 0.0
        efficiency = self.update_count / self.forward_count
        warmup_ratio = min(self.warmup_samples, self.forward_count) / self.forward_count
        return efficiency, self.update_count, self.forward_count, self.threshold_adjustments, warmup_ratio

    def reset_stats(self):
        """Reset efficiency statistics."""
        self.forward_count = 0
        self.update_count = 0
        self.warmup_kl_values.clear()
        self.last_update_count = 0
        self.threshold_adjustments = 0
        self.warmup_completed = False
        # Reset threshold to original value
        self.kl_threshold = self.original_threshold


def setup_kl_adaptive_cotta(model, optimizer, kl_threshold=0.1, check_interval=10, 
                           low_update_threshold=0.2, high_update_threshold=0.8, scale_factor=1.2,
                           warmup_samples=100, adaptive_threshold=True, steps=1, episodic=False, 
                           mt_alpha=0.999, rst_m=0.01, ap=0.92):
    """Set up KL-Adaptive CoTTA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then wrap with KL-Adaptive CoTTA.
    """
    model = configure_model(model)
    params, param_names = collect_params(model)
    kl_adaptive_cotta_model = KLAdaptiveCoTTA(
        model, optimizer,
        steps=steps,
        episodic=episodic, 
        mt_alpha=mt_alpha, 
        rst_m=rst_m, 
        ap=ap,
        kl_threshold=kl_threshold,
        check_interval=check_interval,
        low_update_threshold=low_update_threshold,
        high_update_threshold=high_update_threshold,
        scale_factor=scale_factor,
        warmup_samples=warmup_samples,
        adaptive_threshold=adaptive_threshold
    )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    logger.info(f"KL threshold: %s", kl_threshold)
    logger.info(f"Check interval: %s", check_interval)
    logger.info(f"Update thresholds: [{low_update_threshold}, {high_update_threshold}]")
    logger.info(f"Scale factor: %s", scale_factor)
    logger.info(f"Warmup samples: %s", warmup_samples)
    logger.info(f"Adaptive threshold: %s", adaptive_threshold)
    return kl_adaptive_cotta_model
