# torch
import torch
import pytorch_lightning as pl

# Project
import wandb

import ckconv
from dataset_constructor import construct_datamodule
from model_constructor import construct_model
from trainer_constructor import construct_trainer

from functools import partial
from hook_registration import register_hooks

# Loggers
from pytorch_lightning.loggers import WandbLogger

# Configs
import hydra
from omegaconf import OmegaConf
import os
from pathlib import Path


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.3")
def main(
    cfg: OmegaConf,
):
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)

    # Set seed
    # IMPORTANT! This does not make training entirely deterministic! Trainer(deterministic=True) required!
    pl.seed_everything(cfg.seed, workers=True)

    # Check number of available gpus
    cfg.train.avail_gpus = torch.cuda.device_count()
    torch.set_float32_matmul_precision("high")

    # Construct data_module
    datamodule = construct_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    # Append no of iteration to the cfg file for the definition of the schedulers
    distrib_batch_size = cfg.train.batch_size
    if cfg.train.distributed:
        distrib_batch_size *= cfg.train.avail_gpus
    cfg.scheduler.iters_per_train_epoch = (
        len(datamodule.train_dataset) // distrib_batch_size
    )
    cfg.scheduler.total_train_iters = (
        cfg.scheduler.iters_per_train_epoch * cfg.train.epochs
    )

    # Construct model
    model = construct_model(cfg, datamodule)

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        save_dir=os.environ.get("WANDB_LOGGER_DIR", "."),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity != -1 else None,
        config=ckconv.utils.flatten_configdict(cfg),
        log_model=False
        if cfg.offline
        else "all",  # used to save models to wandb during training
        offline=cfg.offline,
        id=cfg.wandb.run_id if cfg.wandb.run_id != -1 else None,
        save_code=False,
    )
    print(f"Wandb id is {wandb.run.id}")

    # Before start training. Verify arguments in the cfg.
    verify_config(cfg)

    # Recreate the command that instantiated this run.
    if isinstance(wandb_logger.experiment.settings, wandb.Settings):
        args = wandb_logger.experiment.settings._args
        command = " ".join(args)

        # Log the command.
        wandb_logger.experiment.config.update(
            {"command": command}, allow_val_change=True
        )

    # Print the cfg files prior to training
    print(f"Input arguments \n {OmegaConf.to_yaml(cfg)}")

    # Create trainer
    trainer, checkpoint_callback = construct_trainer(cfg, wandb_logger)

    # Load checkpoint
    if cfg.pretrained.load:
        # Construct artifact path.
        checkpoint_path = (
            hydra.utils.get_original_cwd() + f"/artifacts/{cfg.pretrained.filename}"
        )

        # Load model from artifact
        print(
            f'IGNORE this validation run. Required due to problem with Lightning model loading \n {"#" * 200}'
        )
        trainer.validate(model, datamodule=datamodule)
        print("#" * 200)
        checkpoint_path += "/model.ckpt"
        model = model.__class__.load_from_checkpoint(
            checkpoint_path,
            network=model.network,
            cfg=cfg,
        )

    # Test before training
    if cfg.test.before_train:
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

    # register hooks
    if cfg.hooks_enabled:
        model.configure_callbacks = partial(register_hooks, cfg, model)

    # Train
    if cfg.train.do:
        if cfg.pretrained.load:
            # From preloaded point
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
        else:
            # Load from wand checkpoint
            resume_ckpt = None
            if cfg.train.resume_wandb and not cfg.offline:
                wb = cfg.wandb
                checkpoint_ref = f"model-{wb.run_id}:{cfg.train.resume_wandb}"
                try:
                    artifact_dir = wandb_logger.download_artifact(
                        checkpoint_ref, artifact_type="model"
                    )
                    resume_ckpt = str(Path(artifact_dir) / "model.ckpt")
                except Exception as e:
                    print(e)
                    print("No checkpoint found in wandb. Training from scratch.")

            elif cfg.train.resume_local:
                resume_ckpt = cfg.train.resume_local

            trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt)

        # Load state dict from best performing model
        model.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
        )

    # Validate and test before finishing
    model.eval()
    trainer.validate(
        model,
        datamodule=datamodule,
    )
    trainer.test(
        model,
        datamodule=datamodule,
    )


def verify_config(cfg: OmegaConf):
    if cfg.train.distributed and cfg.train.avail_gpus < 2:
        raise ValueError(
            f"Distributed only available with more than 1 GPU. Avail={cfg.train.avail_gpus}"
        )
    if cfg.conv.causal and cfg.net.data_dim != 1:
        raise ValueError("Causal conv is only supported in 1D.")
    if (
        cfg.conv.type in ["SeparableFlexConv", "FlexConv"]
        and cfg.mask.type != "gaussian"
        and cfg.net.data_dim != 1
    ):
        raise ValueError(f"Only gaussian masks are supported in {cfg.net.data_dim}.")
    if cfg.train.batch_size % cfg.train.accumulate_grad_steps:
        raise ValueError(
            f"Batch size must be divisible by the number of grad accumulation steps.\n"
            f"Values: batch_size:{cfg.train.batch_size}, "
            f"accumulate_grad_steps:{cfg.train.accumulate_grad_steps}",
        )


if __name__ == "__main__":
    main()
