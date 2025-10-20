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


class KLReguCoTTA(nn.Module):
    """KL Regularized CoTTA adapts a model by entropy minimization during testing.
    
    Uses soft weighting based on KL divergence instead of hard gating.
    All samples participate in updates, but with weights inversely proportional to KL divergence.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, tau=1.0, eps=1e-8):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        # KL Regularization parameters
        self.tau = tau  # Temperature parameter for soft weighting
        self.eps = eps  # Numerical stability parameter
        self.forward_count = 0
        self.update_count = 0
        
        # Statistics for monitoring
        self.kl_sum = 0.0
        self.weight_sum = 0.0
        self.avg_kl = 0.0
        self.avg_weight = 0.0
        
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
            reduction='none'  # Return per-sample KL divergence
        )
        return kl_div.sum(dim=1)  # Sum over classes to get per-sample KL

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
        
        # Compute KL divergence between student and teacher (per-sample)
        kl_div = self.compute_kl_divergence(student_outputs, teacher_outputs)
        
        # Compute soft weights based on KL divergence
        weights = torch.exp(-kl_div / self.tau)
        
        # Compute individual losses
        individual_losses = softmax_entropy(student_outputs, teacher_outputs)
        
        # Compute weighted loss
        weighted_loss = (weights * individual_losses).sum() / (weights.sum() + self.eps)
        
        # Update statistics
        self.kl_sum += kl_div.sum().item()
        self.weight_sum += weights.sum().item()
        self.avg_kl = self.kl_sum / self.forward_count
        self.avg_weight = self.weight_sum / self.forward_count
        
        # Log soft weighting information
        logger.info(f"KL-Regu: avg_KL={kl_div.mean().item():.4f}, "
                   f"avg_weight={weights.mean().item():.4f}, "
                   f"weighted_loss={weighted_loss.item():.4f}")
        
        # Perform weighted update
        self.update_count += 1
        
        # Student update
        weighted_loss.backward()
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
        
        return teacher_outputs

    def get_efficiency_stats(self):
        """Get efficiency statistics."""
        if self.forward_count == 0:
            return 0.0, 0, 0, 0.0, 0.0
        efficiency = self.update_count / self.forward_count
        return efficiency, self.update_count, self.forward_count, self.avg_kl, self.avg_weight
    
    def get_detailed_stats(self):
        """Get detailed statistics including sums."""
        if self.forward_count == 0:
            return 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0
        efficiency = self.update_count / self.forward_count
        return efficiency, self.update_count, self.forward_count, self.avg_kl, self.avg_weight, self.kl_sum, self.weight_sum

    def reset_stats(self):
        """Reset efficiency statistics."""
        self.forward_count = 0
        self.update_count = 0
        self.kl_sum = 0.0
        self.weight_sum = 0.0
        self.avg_kl = 0.0
        self.avg_weight = 0.0


def setup_kl_regu_cotta(model, optimizer, tau=1.0, eps=1e-8, steps=1, episodic=False, 
                       mt_alpha=0.999, rst_m=0.01, ap=0.92):
    """Set up KL Regularized CoTTA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then wrap with KL Regularized CoTTA.
    """
    model = configure_model(model)
    params, param_names = collect_params(model)
    kl_regu_cotta_model = KLReguCoTTA(
        model, optimizer,
        steps=steps,
        episodic=episodic, 
        mt_alpha=mt_alpha, 
        rst_m=rst_m, 
        ap=ap,
        tau=tau,
        eps=eps
    )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    logger.info(f"KL regularization tau: %s", tau)
    logger.info(f"KL regularization eps: %s", eps)
    return kl_regu_cotta_model
