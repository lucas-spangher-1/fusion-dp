# torch
import torch

# project
import ckconv
from ckconv.nn.scheduler import ChainedScheduler

# built-in
import math

# typing
from omegaconf import OmegaConf


def construct_optimizer(
    model,
    optim_cfg: OmegaConf,
):
    """
    Constructs an optimizer for a given model
    :param model: a list of parameters to be trained
    :param optim_cfg:
    :return: optimizer
    """
    # Unpack values from optim_cfg
    optimizer_type = optim_cfg.name
    lr = optim_cfg.lr
    mask_lr = optim_cfg.lr * optim_cfg.mask_lr_ratio
    s4_lr = optim_cfg.s4_lr

    # mask params have the name "mask_params" in the FlexConv
    mask_param_names = []
    # s4 params that should use the lower lr have an attribute "_optim"
    s4_param_names = []
    s4_params_hps = []
    for name, p in model.named_parameters():
        if "mask_params" in name:
            mask_param_names.append(name)
        elif hasattr(p, "_optim"):
            s4_param_names.append(name)
            s4_params_hps.append(p._optim)

    all_params = dict(model.named_parameters())
    other_params = [
        v
        for k, v in model.named_parameters()
        if (k not in mask_param_names) and (k not in s4_param_names)
    ]
    mask_params = [all_params[k] for k in mask_param_names]
    # as list
    parameters = [
        {"params": other_params},
        {"params": mask_params, "lr": mask_lr},
    ]
    # want one group of parameters for each set of custom optimization parameters in hps
    group_hps = []
    group_param_names = []
    for hp, name in zip(s4_params_hps, s4_param_names):
        if hp in group_hps:
            group_param_names[group_hps.index(hp)].append(name)
        else:
            group_hps.append(hp)
            group_param_names.append([name])

    # print out everything
    print(f"Mask parameters, lr={mask_lr}: {mask_param_names}")

    for hp, names in zip(group_hps, group_param_names):
        print(f"S4 group, {repr(hp)}: {names}")
        parameters.append({"params": [all_params[k] for k in names], **hp})

    # Construct optimizer
    if optimizer_type == "SGD":
        # Unpack values from optim_cfg.params
        momentum = optim_cfg.momentum
        nesterov = optim_cfg.nesterov

        optimizer = torch.optim.SGD(
            params=parameters,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            # Weight decay is manually calculated on the train step
            weight_decay=0.0,
        )
    else:
        optimizer = getattr(torch.optim, optimizer_type)(
            params=parameters,
            lr=lr,
            # Weight decay is manually calculated on the train step
            weight_decay=0.0,
        )
    return optimizer


def construct_scheduler(
    optimizer,
    scheduler_cfg: OmegaConf,
):
    """
    Creates a learning rate scheduler for a given model
    :param optimizer: the optimizer to be used
    :return: scheduler
    """

    # Unpack values from cfg.train.scheduler_params
    scheduler_type = scheduler_cfg.name
    factor = scheduler_cfg.factor
    decay_steps = scheduler_cfg.decay_steps
    patience = scheduler_cfg.patience
    mode = scheduler_cfg.mode

    # Get iterations for warmup
    warmup_epochs = scheduler_cfg.warmup_epochs
    warmup_iterations = (
        scheduler_cfg.warmup_epochs * scheduler_cfg.iters_per_train_epoch
    )

    # Get total iterations (used for CosineScheduler)
    total_iterations = scheduler_cfg.total_train_iters

    # Create warm_up scheduler
    if warmup_epochs != -1:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_iterations,
        )
    else:
        warmup_scheduler = None

    # Check consistency
    if scheduler_type != "cosine" and factor == -1:
        raise ValueError(
            f"The factor cannot be {factor} for scheduler {scheduler_type}"
        )

    # Create scheduler
    if scheduler_type == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_cfg.iters_per_train_epoch * decay_steps,
            gamma=factor,
            last_epoch=-warmup_iterations,  # user to sync with warmup
        )
    elif scheduler_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=True,
        )
    elif scheduler_type == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=factor,
            last_epoch=-warmup_iterations,  # user to sync with warmup
        )
    elif scheduler_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_iterations - warmup_iterations,
            last_epoch=-warmup_iterations,
        )
    else:
        lr_scheduler = None
        print(
            f"WARNING! No scheduler will be used. cfg.train.scheduler = {scheduler_type}"
        )

    # Concatenate schedulers if required
    if warmup_scheduler is not None:
        # If both schedulers are defined, concatenate them
        if lr_scheduler is not None:
            lr_scheduler = ChainedScheduler(
                [
                    warmup_scheduler,
                    lr_scheduler,
                ]
            )
        # Otherwise, return only the warmup scheduler
        else:
            lr_scheduler = lr_scheduler

    return lr_scheduler
