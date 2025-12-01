from torch.optim import lr_scheduler

class Scheduler:
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        target_lr: float,
        warmup_steps: int = 1000,
        max_steps: int = 10_000,
    ):
        self.step_count = 0
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        self.warmup = lr_scheduler.LinearLR(
            optimizer,
            start_factor=initial_lr / target_lr,
            end_factor=1,
            total_iters=warmup_steps
        )
        self.decay = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=1e-7
        )
    
    def step(self):
        # --- Step scheduler ---
        if self.step_count < self.warmup_steps:
            self.warmup.step() # Step the warm-up scheduler
        else:
            self.decay.step() # Step the main scheduler
