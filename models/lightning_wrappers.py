# torch
import torch
import pytorch_lightning as pl
import glob
import hydra
import torchmetrics
from . import seqseq_utils
from torchmetrics.classification import Accuracy, Recall, F1Score, AUROC, ROC
from pytorch_lightning.callbacks import Callback
import sys
from . import plotting
import traceback

# project
from optim import construct_optimizer, construct_scheduler
import ckconv

# logger
import wandb

# typing
from omegaconf import OmegaConf


class LightningWrapperBase(pl.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
    ):
        super().__init__()
        # Define network
        self.network = network
        # Save optimizer & scheduler parameters
        self.optim_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        self.disruptivity_plot_cfg = cfg.train.disruptivity_plot
        # Regularization metrics
        if self.optim_cfg.weight_decay != 0.0:
            self.weight_regularizer = ckconv.nn.LnLoss(
                weight_loss=self.optim_cfg.weight_decay,
                norm_type=2,
            )
        else:
            self.weight_regularizer = None
        # Placeholders for logging of best train & validation values
        self.no_params = -1
        # Explicitly define whether we are in distributed mode.
        self.distributed = cfg.train.distributed and cfg.train.avail_gpus != 1
        self.save_hyperparameters(ignore=["network"])

    def forward(self, *args):
        return self.network(*args)

    def configure_optimizers(self):
        # Construct optimizer & scheduler
        optimizer = construct_optimizer(
            model=self,
            optim_cfg=self.optim_cfg,
        )
        scheduler = construct_scheduler(
            optimizer=optimizer,
            scheduler_cfg=self.scheduler_cfg,
        )
        # Construct output dictionary
        output_dict = {"optimizer": optimizer}
        if scheduler is not None:
            output_dict["lr_scheduler"] = {}
            output_dict["lr_scheduler"]["scheduler"] = scheduler
            output_dict["lr_scheduler"]["interval"] = "step"

            # If we use a ReduceLROnPlateu scheduler, we must monitor val/acc
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.scheduler_cfg.mode == "min":
                    output_dict["lr_scheduler"]["monitor"] = "val/loss"
                else:
                    output_dict["lr_scheduler"]["monitor"] = "val/acc"
                output_dict["lr_scheduler"]["reduce_on_plateau"] = True
                output_dict["lr_scheduler"]["interval"] = "epoch"

            # TODO: ReduceLROnPlateau with warmup
            if isinstance(
                scheduler, ckconv.nn.scheduler.ChainedScheduler
            ) and isinstance(
                scheduler._schedulers[-1], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                raise NotImplementedError("cannot use ReduceLROnPlateau with warmup")
        # Return
        return output_dict

    def on_train_start(self):
        if self.global_rank == 0:
            # Calculate and log the size of the model
            if self.no_params == -1:
                with torch.no_grad():
                    # Log parameters
                    no_params = ckconv.utils.no_params(self.network)
                    self.logger.experiment.summary["no_params"] = no_params
                    self.no_params = no_params

                    # Log code
                    code = wandb.Artifact(
                        f"source-code-{self.logger.experiment.name}", type="code"
                    )
                    # Get paths
                    paths = glob.glob(
                        hydra.utils.get_original_cwd() + "/**/*.py",
                        recursive=True,
                    )
                    paths += glob.glob(
                        hydra.utils.get_original_cwd() + "/**/*.yaml",
                        recursive=True,
                    )
                    # Filter paths
                    paths = list(filter(lambda x: "outputs" not in x, paths))
                    paths = list(filter(lambda x: "venv" not in x, paths))
                    paths = list(filter(lambda x: "wandb" not in x, paths))
                    # Get all source files
                    for path in paths:
                        code.add_file(
                            path,
                            name=path.replace(f"{hydra.utils.get_original_cwd()}/", ""),
                        )
                    # Use the artifact
                    if not self.logger.experiment.offline:
                        wandb.run.use_artifact(code)


class ClassificationWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
        **kwargs,
    ):
        super().__init__(
            network=network,
            cfg=cfg,
        )

        # Binary problem?
        n_classes = network.out_layer.out_channels
        self.multiclass = n_classes != 1
        self.seq_out = network.OUTPUT_TYPE == "sequence"

        # Other metrics
        task = "multiclass" if self.multiclass else "binary"

        def make_metrics(stage, **kwargs):
            metrics = {
                "acc": Accuracy(**kwargs),
                "recall": Recall(**kwargs),
                "f1": F1Score(**kwargs),
                "auroc": AUROC(thresholds=100, **kwargs),
                "roc": ROC(thresholds=100, **kwargs),
            }
            # Each metric also has to be set as an attribute on the module
            for name, metric in metrics.items():
                setattr(self, f"{stage}_{name}", metric)
            return metrics

        kwargs = {"num_classes": n_classes, "task": task}
        self.train_metrics = make_metrics("train", **kwargs)
        self.val_metrics = make_metrics("val", **kwargs)
        self.test_metrics = make_metrics("test", **kwargs)

        # loss_metric should accept (logits, labels) or
        #   (logits, labels, lens) if seq_out
        # get_predictions should accept (logits, lengths)
        # get_probabilities should accept (logits, lenghts)
        if self.multiclass:
            self.loss_metric = torch.nn.CrossEntropyLoss()
            self.get_predictions = self.multiclass_prediction
            self.get_probabilities = self.multiclass_probabilities
        elif self.seq_out:
            # If we output a seq, we expect a loss function of (logits, labels, lengths)
            self.loss_metric = seqseq_utils.make_masked_shotmean_loss_fn(
                torch.nn.BCEWithLogitsLoss()
            )
            self.get_predictions = seqseq_utils.get_preds_any
            self.get_probabilities = seqseq_utils.get_preds_any  # TODO: revisit this
        else:
            self.loss_metric = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(cfg.train.pos_weight, dtype=torch.float)
            )
            self.get_predictions = self.binary_prediction
            self.get_probabilities = self.binary_probabilities

        # Placeholders for logging of best train & validation values
        self.best_train_metrics = {}
        self.best_val_metrics = {}
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def _preprocess_batch(self, batch):
        if len(batch) < 3:
            x, labels = batch[:2]
            return x, labels, x.shape[-1] * torch.ones_like(labels)
        else:
            return batch

    def _step(self, batch, metrics_dict: dict, compute_metrics: bool = False):
        """Run a single step of the model

        Args:
            batch (Tensor): the batch to run the step on
            metrics_dict (dict): The dictionary of metrics (self.train_metrics, etc...)
            compute_metrics (bool, optional): Whether to compute the value of each
                metric. Otherwise, we just call `.update` to update its internal state.
                Defaults to False.

        Returns:
            probalities (Tensor), logits (Tensor), loss (Tensor)
        """
        # batch can contain either x, labels or x, labels, lens
        x, labels, lens = self._preprocess_batch(batch)
        logits = self.forward(x, lens)

        # Probabilities
        probabilities = self.get_probabilities(logits, lens)

        # Calculate metrics
        for name, metric in metrics_dict.items():
            # We don't want to compute roc on every step, since
            # we only log it per epoch
            if name == "roc":
                metric.update(probabilities, labels.round().int())
            elif not compute_metrics:
                metric.update(probabilities, labels)
            else:
                metric(probabilities, labels)

        # For binary classification, the labels must be float
        if not self.multiclass:
            labels = labels.float()  # N
            logits = logits.view(-1)  # N

        loss = None
        if self.seq_out:
            # loss requires the lens also
            loss = self.loss_metric(logits, labels, lens)
        else:
            loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return probabilities, logits, loss

    def _log_metrics(self, stage: str, metrics_dict, **kwargs):
        for name, metric in metrics_dict.items():
            if name != "roc":  # we just log ROC plots
                self.log(f"{stage}/{name}", metric, **kwargs)

    def training_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(
            batch, self.train_metrics, compute_metrics=True
        )
        # Add regularization
        if self.weight_regularizer is not None:
            reg_loss = self.weight_regularizer(self.network)
        else:
            reg_loss = 0.0
        # Log metrics
        kwargs = {"on_step": True, "on_epoch": True, "sync_dist": self.distributed}
        self.log("train/loss", loss, prog_bar=True, **kwargs)
        self._log_metrics("train", self.train_metrics, **kwargs)
        # Store loss and logits for on_train_epoch_end
        if self.seq_out:  # we do this to save memory, not sure the impact
            logits = torch.mean(logits, dim=-1)
        self.train_step_outputs.append(
            {"loss": loss + reg_loss, "logits": logits.detach()}
        )
        # Do I still need the logits in this?
        return {"loss": loss + reg_loss, "logits": logits.detach()}

    def validation_step(self, batch, batch_idx):
        # Perform step
        with torch.no_grad():
            predictions, logits, loss = self._step(
                batch, self.val_metrics, compute_metrics=False
            )
            # Log and return loss (Required in training step)

            kwargs = {"on_epoch": True, "sync_dist": self.distributed}
            self.log("val/loss", loss, **kwargs)
            self._log_metrics("val", self.val_metrics, **kwargs)

            if self.seq_out:
                logits = torch.mean(logits, dim=0)

            # used to log histograms in validation_epoch_step
            self.validation_step_outputs.append({"logits": logits})

            # Do disruptivity plotting
            dpcfg = self.disruptivity_plot_cfg
            if dpcfg.enabled and dpcfg.batch_idx == batch_idx:
                x, labels, lens = self._preprocess_batch(batch)
                out = (
                    self(x, lens)
                    if self.seq_out
                    else self.network.forward_unrolled(x, lens)
                ).cpu()
                fig = plotting.plot_disruption_predictions(out, batch, dpcfg)
                self.logger.experiment.log({"val/disruptivity_plot": wandb.Image(fig)})

            return logits

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, _, loss = self._step(
            batch, self.test_metrics, compute_metrics=False
        )
        self.log("test/loss", loss, on_epoch=True, sync_dist=self.distributed)
        self._log_metrics(
            "test",
            self.test_metrics,
            on_epoch=True,
            sync_dist=self.distributed,
        )

    def on_train_epoch_end(self):
        flattened_logits = torch.flatten(
            torch.cat(
                [step_output["logits"] for step_output in self.train_step_outputs]
            )
        )
        self.logger.experiment.log(
            {
                "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
            }
        )
        # Log best accuracy
        for name, _ in self.train_metrics.items():
            if name == "roc":
                continue
            this_epoch = self.trainer.callback_metrics[f"train/{name}_epoch"]
            prev_best = self.best_train_metrics.get(name, None)
            if not prev_best or this_epoch > prev_best:
                self.best_train_metrics[name] = this_epoch.item()
                self.logger.experiment.log(
                    {
                        f"train/best_{name}": self.best_train_metrics[name],
                    }
                )

        fig, _ = self.train_metrics["roc"].plot()
        self.logger.experiment.log({"train/roc": fig})

        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        # Gather logits from validation set and construct a histogram of them.
        flattened_logits = torch.flatten(
            torch.cat([output["logits"] for output in self.validation_step_outputs])
        )
        self.logger.experiment.log(
            {
                "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "val/logit_max_abs_value": flattened_logits.abs().max().item(),
            }
        )
        # Log best accuracy
        for name, _ in self.val_metrics.items():
            if name == "roc":
                continue
            this_epoch = self.trainer.callback_metrics[f"val/{name}"]
            prev_best = self.best_val_metrics.get(name, None)
            if not prev_best or this_epoch > prev_best:
                self.best_val_metrics[name] = this_epoch.item()
                self.logger.experiment.log(
                    {
                        f"val/best_{name}": self.best_val_metrics[name],
                    }
                )

        fig, _ = self.val_metrics["roc"].plot()
        self.logger.experiment.log({"val/roc": fig})

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        fig, _ = self.test_metrics["roc"].plot()
        self.logger.experiment.log({"test/roc": fig})

    # This has a *args to ignore lengths if they get passed
    @staticmethod
    def multiclass_prediction(logits, *args):
        return torch.argmax(logits, 1)

    # This has a *args to ignore lengths if they get passed
    @staticmethod
    def multiclass_probabilities(logits, *args):
        return torch.softmax(logits, 1)

    # This has a *args to ignore lengths if they get passed
    @staticmethod
    def binary_prediction(logits, *args):
        return (logits > 0.0).squeeze().long()

    # This has a *args to ignore lengths if they get passed
    @staticmethod
    def binary_probabilities(logits, *args):
        return torch.sigmoid(logits).squeeze()


class RegressionWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
        metric: str,
        **kwargs,
    ):
        super().__init__(
            network=network,
            cfg=cfg,
        )
        if metric == "MAE":
            MetricClass = torchmetrics.MeanAbsoluteError
            LossMetricClass = torch.nn.L1Loss
        elif metric == "MSE":
            MetricClass = torchmetrics.MeanSquaredError
            LossMetricClass = torch.nn.MSELoss
        else:
            raise ValueError(f"Metric {metric} not recognized")

        # Other metrics
        self.train_metric = MetricClass()
        self.val_metric = MetricClass()
        self.test_metric = MetricClass()
        # Loss metric
        self.loss_metric = LossMetricClass()
        # Placeholders for logging of best train & validation values
        self.best_train_loss = 1e9
        self.best_val_loss = 1e9

    def _step(self, batch, metric_calculator):
        x, labels = batch
        prediction = self(x)
        # Calculate loss
        metric_calculator(prediction, labels)
        loss = self.loss_metric(prediction, labels)
        # Return predictions and loss
        return prediction, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        _, loss = self._step(batch, self.train_metric)
        # Log loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)

        # Add regularization
        if self.regularizer is None:
            # Return loss (required in training step)
            return loss
        else:
            reg_loss = self.regularizer(self.network)
            self.log("train/reg_loss", reg_loss, on_epoch=True, prog_bar=True)
            return loss + reg_loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, loss = self._step(batch, self.val_metric)
        # Log and return loss (Required in training step)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, loss = self._step(batch, self.test_metric)
        # Log and return loss (Required in training step)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def training_epoch_end(self, train_step_outputs):
        # Log best accuracy
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def validation_epoch_end(self, validation_step_outputs):
        # Log best accuracy
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.logger.experiment.log(
                {
                    "val/best_loss": self.best_val_loss,
                    "global_step": self.global_step,
                }
            )


#############################
#    Point Cloud Models     #
#############################
class PyGClassificationWrapper(ClassificationWrapper):
    def _step(self, batch, accuracy_calculator, recall_calculator, f1_calculator):
        logits = self(batch)
        # Predictions
        predictions = torch.argmax(logits, 1)
        # Calculate accuracy and loss
        accuracy_calculator(predictions, batch.y)
        recall_calculator(predictions, batch.y)
        f1_calculator(predictions, batch.y)

        loss = self.loss_metric(logits, batch.y)
        # Return predictions and loss
        return predictions, logits, loss


class PyGRegressionWrapper(RegressionWrapper):
    def _step(self, batch, metric_calculator):
        prediction = self(batch)
        # Calculate loss
        metric_calculator(prediction, batch.y)
        loss = self.loss_metric(prediction, batch.y)
        # Return predictions and loss
        return prediction, loss


class OnExceptionExit(Callback):
    def on_exception(self, trainer, module, exception):
        traceback.print_exception(exception)
        # print(f"exception caught, gracefully shutting down: {exception}")
        sys.exit("Graceful shutdown")
