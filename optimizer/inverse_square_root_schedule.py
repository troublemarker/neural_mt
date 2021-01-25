
class InverseSquareRootScheduler(object):

    def __init__(self, args):
        self.warmup_updates = args.warmup_updates
        self.warmup_end_lr = args.lr[0]
        self.warmup_init_lr = args.init_warmup_lr if args.warmup_updates > 0 else self.warmup_end_lr
        self.warmup_velocity = (self.warmup_end_lr - self.warmup_init_lr) / args.warmup_updates

    def lr_step_forward(self, num_updates):

        if num_updates < self.warmup_updates:
            lr = self.warmup_init_lr + self.warmup_velocity * num_updates
        else:
            lr = self.warmup_end_lr * (self.warmup_updates / num_updates) ** 0.5
        return lr


