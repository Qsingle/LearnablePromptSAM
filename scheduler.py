#coding:utf-8
from torch.optim import Optimizer
from abc import ABCMeta


class LRScheduler(metaclass=ABCMeta):
    def __init__(self, optimizer: Optimizer, start=-1):

        self.optimizer = optimizer
        self.current_step = start

        if start == -1:
            for group in self.optimizer.param_groups:
                if hasattr(group, "initial_lr"):
                    continue
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified in "
                        "param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], self.optimizer.param_groups)
        )

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.current_step += 1
        values = self.get_lr()
        for groups, lr in zip(self.optimizer.param_groups, values):
            groups["lr"] = lr

class PolyLRScheduler(LRScheduler):
    def __init__(self, optimizer,  num_images, batch_size, epochs, gamma=0.9, start=-1, drop_last=False):
        super(PolyLRScheduler, self).__init__(optimizer, start)
        if num_images % batch_size == 0 or drop_last:
            total_iterations = num_images // batch_size * epochs
        else:
            total_iterations = (num_images // batch_size + 1) * epochs

        self.total_iterations = total_iterations
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_images = num_images
        self.epochs = epochs
        print("Initial learning rate set to:{}".format([group["initial_lr"] for group
                                                        in self.optimizer.param_groups]))

    def get_lr(self):
        def calc_lr(group):
            lr = group["initial_lr"] * (1-self.current_step/self.total_iterations)**self.gamma
            return lr
        return [calc_lr(group) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            key:value
            for key, value in self.__dict__.items()
            if key in ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs"]
        }

    def load_state_dict(self, state_dict):
        tmp_state = {}
        keys = ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs"]
        for key in keys:
            if key not in state_dict:
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_state[key] = state_dict[key]
        self.__dict__.update(tmp_state)