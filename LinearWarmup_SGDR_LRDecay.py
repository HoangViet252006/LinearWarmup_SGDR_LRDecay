import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import matplotlib.pyplot as plt


class WarmupThenConstantLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_lr, step_per_epoch, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.step_per_epoch = step_per_epoch
        self.last_step = last_epoch if last_epoch >= 0 else -1
        super(WarmupThenConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_stage = (self.last_step + 1) // self.step_per_epoch
        max_warmup_stage = self.warmup_steps // self.step_per_epoch

        if warmup_stage < max_warmup_stage:
            current_lr = self.max_lr * (warmup_stage + 1) / max_warmup_stage
        else:
            current_lr = self.max_lr

        return [current_lr for _ in self.base_lrs]

    def step(self):
        self.last_step += 1
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group["lr"] = lr


class CosineAnnealingWarmRestartsModified(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, max_lr=0.1, gamma=0.9):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.gamma = gamma
        self.max_lr = max_lr
        self.restart_count = 0

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        if self.T_cur == 0:
            self.restart_count += 1
            self.base_lrs = [self.max_lr * self.gamma ** self.restart_count for _ in self.base_lrs]

        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]


class CombinedWarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_lr, T_0, T_mult=1, eta_min=0, step_per_epoch=0, gamma=0.9):
        self.warmup_steps = warmup_epochs * step_per_epoch
        self.current_step = 0
        self.warmup_scheduler = WarmupThenConstantLR(
            optimizer, warmup_steps=self.warmup_steps, max_lr=max_lr, step_per_epoch=step_per_epoch
        )
        self.cosine_scheduler = CosineAnnealingWarmRestartsModified(
            optimizer, T_0=T_0 * step_per_epoch, T_mult=T_mult, eta_min=eta_min, max_lr=max_lr, gamma=gamma
        )
        super().__init__(optimizer)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.warmup_scheduler.get_lr()
        return self.cosine_scheduler.get_lr()

    def step(self):
        if self.current_step < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.current_step += 1
        super().step()

if __name__ == '__main__':
    lr = 0.1
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    step_per_epoch = 10
    epochs = 200
    scheduler = CombinedWarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_lr=lr,
        T_0=10,
        T_mult=2,
        eta_min=0.002,
        step_per_epoch=step_per_epoch,
        gamma=0.9
    )

    list_lr = []

    for epoch in range(epochs):
        for batch in range(step_per_epoch):
            optimizer.step()
            scheduler.step()

        for param_group in optimizer.param_groups:
            list_lr.append(param_group["lr"])

    def plot_learning_rate(list_lr):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(list_lr) + 1), list_lr, label="Learning Rate")
        plt.xlabel("Batch")
        plt.ylabel("LR Value")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid()
        plt.savefig("Plot", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()



    plot_learning_rate(list_lr)
