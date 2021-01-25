from torch.optim.adam import Adam


class CustomAdam(Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, lr_scheduler=None):
        super(CustomAdam, self).__init__(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                                         weight_decay=0, amsgrad=False)
        self.lr_scheduler = lr_scheduler

    @property
    def params(self):
        for group in self.param_groups:
            for param in group:
                yield param

    def multiply_grads(self, multiplier):
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(multiplier)

    def clip_grad_norm(self, max_norm):
        pass

    def set_lr(self, num_updates):
        lr = self.lr_scheduler.lr_step_forward(num_updates)
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.param_groups[0]['lr']



