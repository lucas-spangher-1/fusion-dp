import pytorch_lightning as pl
from models.lightning_wrappers import OnExceptionExit
import os

# typing
import torch.cuda
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger


def construct_trainer(
    cfg: OmegaConf,
    wandb_logger: pl.loggers.WandbLogger,
) -> tuple[pl.Trainer, pl.Callback]:
    # Set up precision
    if cfg.train.mixed_precision:
        precision = 16
    else:
        precision = 32

    # Set up determinism
    if cfg.deterministic:
        deterministic = True
    else:
        deterministic = False

    # Callback to print model summary
    modelsummary_callback = pl.callbacks.ModelSummary(
        max_depth=-1,
    )

    # Metric to monitor
    if cfg.scheduler.mode == "max":
        monitor = "val/auroc"
    elif cfg.scheduler.mode == "min":
        monitor = "val/loss"

    # Callback for model checkpointing:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=cfg.scheduler.mode,  # Save on best loss or auroc
        verbose=True,
    )

    # Callback for learning rate monitoring
    lrmonitor_callback = pl.callbacks.LearningRateMonitor()

    # Callback for early stopping:
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        mode=cfg.scheduler.mode,
        patience=cfg.train.max_epochs_no_improvement,
        verbose=True,
        strict=True,
    )

    exception_callback = OnExceptionExit()

    """
    TODO:
    detect_anomaly
    limit batches
    profiler
    overfit_batches
    resume from checkpoint
    StochasticWeightAveraging
    log_every_n_steps
    """
    # Distributed training params
    if cfg.device == "cuda":
        accelerator = "auto"
        sync_batchnorm = cfg.train.distributed
        strategy = (
            "ddp_find_unused_parameters_false" if cfg.train.distributed else "auto"
        )
        devices = cfg.train.avail_gpus if cfg.train.distributed else 1
        num_nodes = cfg.train.num_nodes if (cfg.train.num_nodes != -1) else 1
    else:
        print("_____ USING CPU _______")
        accelerator = "cpu"
        devices = "auto"
        sync_batchnorm = False
        strategy = "auto"
        num_nodes = 0

    if cfg.train.track_grad_norm != -1:
        # TODO: figure this out
        None

    print("CWD is :" + os.getcwd())

    # create trainer
    trainer = pl.Trainer(
        default_root_dir=os.environ.get("TRAINER_DIR", os.getcwd()),
        accelerator=accelerator,
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        gradient_clip_val=cfg.train.grad_clip,
        accumulate_grad_batches=cfg.train.accumulate_grad_steps,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        limit_test_batches=cfg.train.limit_test_batches,
        # Callbacks
        callbacks=[
            modelsummary_callback,
            lrmonitor_callback,
            checkpoint_callback,
            early_stopping_callback,
            exception_callback,
        ],
        # Multi-GPU
        num_nodes=num_nodes,
        devices=devices,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        # auto_select_gpus=True,
        # Precision
        precision=precision,
        # Determinism
        deterministic=deterministic,
    )
    return trainer, checkpoint_callback
