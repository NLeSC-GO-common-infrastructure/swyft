from typing import Callable, Optional, Union, Collection
from abc import ABC, abstractmethod
from functools import partial

import torch

from .types import Tensor, Array, ScalarFloat


class Hook(ABC):
    def __init__(self):
        self._stop_training = False

    @property
    @abstractmethod
    def state_dict(self):
        pass

    @state_dict.setter
    @abstractmethod
    def state_dict(self, state_dict: dict):
        pass

    def load_state_dict(self, state_dict: dict):
        self.state_dict = state_dict

    @property
    def stop_training(self):
        return self._stop_training

    def on_optimizer_init(self, optimizer: torch.optim.Optimizer):
        pass

    def on_val_loss(self, val_loss: float):
        pass

    def on_epoch_end(self):
        pass
    
    def on_x(self, x: Tensor, z: Tensor):
        return x
    
    def on_z(self, x: Tensor, z: Tensor):
        return z


class EarlyStopping(Hook):
    def __init__(self, patience: int):
        super().__init__()
        self.best_loss = float("Inf")
        self.counter = 0
        self.patience = patience

    @property
    def state_dict(self):
        return {"counter": self.counter}
    
    @state_dict.setter
    def state_dict(self, state_dict: dict):
        self.counter = state_dict["counter"]

    def on_val_loss(self, val_loss: float):
        if val_loss > self.best_loss:
            self.counter += 1
        else:
            self.best_loss = val_loss
            self.counter = 0

        if self.counter > self.patience:
            self._stop_training = True


class StepNoise(Hook):
    def __init__(
        self, 
        noise_model: Callable[[Array, Array, ScalarFloat], Array],
        step_size: int,
        gamma: ScalarFloat,
        noise_max: ScalarFloat,
        initial_noise_level: ScalarFloat = 0.,
        verbose: bool = False,
):
        """Increases the noise by gamma upto noise_max every step_size epochs.
        
        Args:
            noise_model: noise_model(x, z, noise_level) yields x with noise. It must be able to handle batches.
            step_size: Number of epochs before noise is increased.
            gamma: Amount to increase noise.
            noise_max: Maximum noise level.
            initial_noise_level: Noise level at the start.
            verbose: Prints whenever noise level is increased.
        """
        super().__init__()
        self.counter = 0
        self.noise_level = initial_noise_level
        self.noise_model = noise_model
        self.step_size = step_size
        self.gamma = gamma
        self.noise_max = noise_max
        self.verbose = verbose
    
    @property
    def state_dict(self):
        return {
            "counter": self.counter,
            "noise_level": self.noise_level,
            'noise_model': self.noise_model,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'noise_max': self.noise_max,
            'verbose': self.verbose
        }
    
    @state_dict.setter
    def state_dict(self, state_dict: dict):
        self.counter = state_dict["counter"]
        self.noise_level = state_dict["noise_level"]
        self.noise_model = state_dict["noise_model"]
        self.step_size = state_dict["step_size"]
        self.gamma = state_dict["gamma"]
        self.noise_max = state_dict["noise_max"]
        self.verbose = state_dict["verbose"]

    def on_epoch_end(self):
        self.counter += 1
        if self.noise_level >= self.noise_max:
            return None

        if self.counter % self.step_size == 0:
            self.noise_level += self.gamma
            if self.verbose:
                print(f"At counter={self.counter}, noise level increased to {self.noise_level}.")
    
    def on_x(self, x: Tensor, z: Tensor):
        return self.noise_model(x.cpu(), z.cpu(), self.noise_level)


class StepLR(Hook):
    def __init__(
        self, 
        step_size: int, 
        gamma: ScalarFloat = 0.1, 
        last_epoch: int = -1,
        verbose: bool = False,
):
        """Wraps torch.optim.lr_scheduler.StepLR
        
        Args:
            optimizer: Wrapped optimizer.
            step_size: Period of learning rate decay.
            gamma: Multiplicative factor of learning rate decay. Default: 0.1.
            last_epoch: The index of last epoch. Default: -1.
        """
        super().__init__()
        self._pre_optimizer_scheduler = partial(
            torch.optim.lr_scheduler.StepLR,
            step_size=step_size, 
            gamma=gamma, 
            last_epoch=last_epoch,
        )
        self.scheduler = None

    @property
    def state_dict(self):
        return self.scheduler.state_dict
    
    @state_dict.setter
    def state_dict(self, state_dict: dict):
        self.scheduler.load_state_dict(state_dict)

    def on_optimizer_init(self, optimizer: torch.optim.Optimizer):
        self.scheduler = self._pre_optimizer_scheduler(optimizer)

    def on_epoch_end(self):
        self.scheduler.step()


class ReduceLROnPlateau(Hook):
    def __init__(
        self, 
        mode: str = 'min', 
        factor: float = 0.1, 
        patience: int = 10, 
        threshold: float = 0.0001, 
        threshold_mode: str = 'rel', 
        cooldown: int = 0, 
        min_lr: Union[float, Collection[float]] = 0, 
        eps: float = 1e-08,
):
        """Wraps torch.optim.lr_scheduler.ReduceLROnPlateau
            
        Args:
            optimizer: Wrapped optimizer.
            mode: One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
            factor: Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            patience: Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
            threshold: Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
            threshold_mode: One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
            cooldown: Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
            min_lr: A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
            eps: Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
        """
        super().__init__()
        self._pre_optimizer_scheduler = partial(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            mode=mode, 
            factor=factor, 
            patience=patience, 
            threshold=threshold, 
            threshold_mode=threshold_mode, 
            cooldown=cooldown, 
            min_lr=min_lr, 
            eps=eps,
        )
        self.scheduler = None

    @property
    def state_dict(self):
        return self.scheduler.state_dict
    
    @state_dict.setter
    def state_dict(self, state_dict: dict):
        self.scheduler.load_state_dict(state_dict)

    def on_optimizer_init(self, optimizer: torch.optim.Optimizer):
        self.scheduler = self._pre_optimizer_scheduler(optimizer)

    def on_val_loss(self, val_loss):
        self.scheduler.step(val_loss)


if __name__ == "__main__":
    pass
