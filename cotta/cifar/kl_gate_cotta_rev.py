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


class KLGateCoTTARev(nn.Module):
    """KL-Gate CoTTA Rev adapts a model by entropy minimization during testing.
    
    Adds a reversed KL divergence gate to skip updates when teacher-student agreement is high.
    This is the opposite of the original KL-Gate: skip updates when KL < threshold.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, kl_threshold=0.1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        # KL-Gate Rev parameters
        self.kl_threshold = kl_threshold
        self.forward_count = 0
        self.update_count = 0
        
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
        
        # KL-Gate Rev: Skip update if KL divergence is too low (high agreement)
        if self.kl_threshold > 0 and kl_div < self.kl_threshold:
            logger.info(f"KL-Gate-Rev triggered: KL={kl_div:.4f} < threshold={self.kl_threshold:.4f}, skipping update")
            # Return teacher outputs without updating
            return teacher_outputs.detach()
        
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
        
        return teacher_outputs

    def get_efficiency_stats(self):
        """Get efficiency statistics."""
        if self.forward_count == 0:
            return 0.0, 0, 0
        efficiency = self.update_count / self.forward_count
        return efficiency, self.update_count, self.forward_count

    def reset_stats(self):
        """Reset efficiency statistics."""
        self.forward_count = 0
        self.update_count = 0


def setup_kl_gate_cotta_rev(model, optimizer, kl_threshold=0.1, steps=1, episodic=False, 
                           mt_alpha=0.999, rst_m=0.01, ap=0.92):
    """Set up KL-Gate CoTTA Rev adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then wrap with KL-Gate CoTTA Rev.
    """
    model = configure_model(model)
    params, param_names = collect_params(model)
    kl_gate_cotta_rev_model = KLGateCoTTARev(
        model, optimizer,
        steps=steps,
        episodic=episodic, 
        mt_alpha=mt_alpha, 
        rst_m=rst_m, 
        ap=ap,
        kl_threshold=kl_threshold
    )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    logger.info(f"KL threshold (rev): %s", kl_threshold)
    return kl_gate_cotta_rev_model
