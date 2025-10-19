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


class KLForceCoTTA(nn.Module):
    """KL-Force CoTTA adapts a model by entropy minimization during testing.
    
    Adds a forced update mechanism to prevent learning stagnation.
    When consecutive skips reach a threshold, forces an update or adapts the KL threshold.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, 
                 kl_threshold=0.1, force_update_interval=10, adaptive_threshold=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        # KL-Force parameters
        self.kl_threshold = kl_threshold
        self.force_update_interval = force_update_interval
        self.adaptive_threshold = adaptive_threshold
        self.forward_count = 0
        self.update_count = 0
        self.force_update_count = 0
        
        # Tracking variables
        self.consecutive_skips = 0
        self.recent_kl_values = []
        self.original_threshold = kl_threshold
        
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
        
        # Check if forced update should be triggered
        force_update = False
        force_reason = ""
        
        if self.consecutive_skips >= self.force_update_interval:
            if self.adaptive_threshold:
                # Adaptive threshold: adjust threshold based on recent KL values
                if len(self.recent_kl_values) > 0:
                    old_threshold = self.kl_threshold
                    self.kl_threshold = sum(self.recent_kl_values) / len(self.recent_kl_values)
                    self.recent_kl_values.clear()
                    force_reason = f"adaptive threshold: {old_threshold:.4f} -> {self.kl_threshold:.4f}"
                    logger.info(f"KL-Force adaptive threshold: {old_threshold:.4f} -> {self.kl_threshold:.4f}")
                else:
                    force_reason = "no recent KL values for adaptation"
            else:
                # Simple forced update
                force_reason = "consecutive skips limit reached"
                logger.info(f"KL-Force triggered: {self.consecutive_skips} consecutive skips")
            
            force_update = True
            self.consecutive_skips = 0
            self.force_update_count += 1
        
        # Decide whether to update based on KL threshold or forced update
        should_update = force_update or (self.kl_threshold > 0 and kl_div <= self.kl_threshold)
        
        if should_update:
            # Perform normal CoTTA update
            self.update_count += 1
            self.consecutive_skips = 0
            
            if force_update:
                logger.info(f"KL-Force update: KL={kl_div:.4f}, reason: {force_reason}")
            
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
            # Skip update and record KL value for potential threshold adaptation
            if self.adaptive_threshold and self.kl_threshold > 0 and kl_div > self.kl_threshold:
                self.recent_kl_values.append(kl_div.item())
                # Keep only recent values to avoid memory issues
                if len(self.recent_kl_values) > self.force_update_interval * 2:
                    self.recent_kl_values = self.recent_kl_values[-self.force_update_interval:]
            
            self.consecutive_skips += 1
            logger.info(f"KL-Force skip: KL={kl_div:.4f} > threshold={self.kl_threshold:.4f}, "
                       f"consecutive_skips={self.consecutive_skips}")
            return teacher_outputs.detach()
        
        return teacher_outputs

    def get_efficiency_stats(self):
        """Get efficiency statistics."""
        if self.forward_count == 0:
            return 0.0, 0, 0, 0, 0.0
        efficiency = self.update_count / self.forward_count
        force_ratio = self.force_update_count / self.forward_count if self.forward_count > 0 else 0.0
        return efficiency, self.update_count, self.forward_count, self.force_update_count, force_ratio

    def reset_stats(self):
        """Reset efficiency statistics."""
        self.forward_count = 0
        self.update_count = 0
        self.force_update_count = 0
        self.consecutive_skips = 0
        self.recent_kl_values.clear()
        # Reset threshold to original value
        self.kl_threshold = self.original_threshold


def setup_kl_force_cotta(model, optimizer, kl_threshold=0.1, force_update_interval=10, 
                        adaptive_threshold=True, steps=1, episodic=False, 
                        mt_alpha=0.999, rst_m=0.01, ap=0.92):
    """Set up KL-Force CoTTA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then wrap with KL-Force CoTTA.
    """
    model = configure_model(model)
    params, param_names = collect_params(model)
    kl_force_cotta_model = KLForceCoTTA(
        model, optimizer,
        steps=steps,
        episodic=episodic, 
        mt_alpha=mt_alpha, 
        rst_m=rst_m, 
        ap=ap,
        kl_threshold=kl_threshold,
        force_update_interval=force_update_interval,
        adaptive_threshold=adaptive_threshold
    )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    logger.info(f"KL threshold: %s", kl_threshold)
    logger.info(f"Force update interval: %s", force_update_interval)
    logger.info(f"Adaptive threshold: %s", adaptive_threshold)
    return kl_force_cotta_model
