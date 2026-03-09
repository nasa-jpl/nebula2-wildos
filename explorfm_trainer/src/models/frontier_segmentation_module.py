from typing import Any, Dict, Tuple

import torch
import wandb
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.segmentation import MeanIoU

from .binary_segmentation_module import BinarySegmentationLitModule
from .components.frontier_viz_utils import gen_logging_image

class FrontierSegmentationLitModule(BinarySegmentationLitModule):
    """`LightningModule` for Frontier Detection as a Binary Segmentation task.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        pred_threshold: float = 0.5,
        num_log_imgs: int = 4,
        validation_img_log_idx: int = 0,
        strict_loading: bool = True,
        vmax: float = 1,
        pos_weight: float = 1.0,
    ) -> None:
        """Initialize a `BinarySegmentationLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model for faster training with PyTorch 2.0.
        :param pred_threshold: The threshold for binary predictions.
        :param num_log_imgs: The number of images to log during training.
        :param validation_img_log_idx: The index of the batch to log images during validation.
        :param strict_loading: Whether to strictly check for missing keys when loading the model state.
        :param vmax: The maximum value for the colormap in logging probability heatmaps.
        """
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            pred_threshold=pred_threshold,
            num_log_imgs=num_log_imgs,
            validation_img_log_idx=validation_img_log_idx,
            strict_loading=strict_loading,
            vmax=vmax
        )
        
        # use bcewithlogits
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))


    def model_step(
        self, batch: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of probabilities.
        """
        x = batch["image"]
        y = batch["mask"]
        probs = self.forward(x)
        loss = self.criterion(probs, y.float())

        probs = torch.sigmoid(probs)
        pred = (probs > self.pred_threshold).int()

        return loss, pred, probs

    def training_step(
        self, batch: Dict, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, probs = self.model_step(batch)
        targets = batch["mask"]

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_miou(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/miou", self.train_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Visualize predictions and targets on the second last batch of the epoch
        if batch_idx == len(self.trainer.train_dataloader) - 2:
            self.logger.experiment.log(
                {
                    "train/log_imgs": [
                        wandb.Image(img) for img in gen_logging_image(
                            batch_data={
                                **batch,
                                "preds": preds,
                                "probs": probs.detach(),
                            },
                            num_log_imgs=self.hparams.num_log_imgs,
                            vmax=self.hparams.vmax
                        )
                    ]
                },
                step=self.global_step
            )

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs = self.model_step(batch)
        targets = batch["mask"]

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_miou(preds, targets)
        self.val_f1(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/miou", self.val_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Visualize predictions and targets
        if batch_idx == self.hparams.validation_img_log_idx:
            self.logger.experiment.log(
                {
                    "val/log_imgs": [
                        wandb.Image(img) for img in gen_logging_image(
                            batch_data={
                                **batch,
                                "preds": preds,
                                "probs": probs,
                            },
                            num_log_imgs=self.hparams.num_log_imgs,
                            vmax=self.hparams.vmax
                        )
                    ]
                },
                step=self.global_step
            )

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs = self.model_step(batch)
        targets = batch["mask"]

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_miou(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/miou", self.test_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Visualize predictions and targets
        if batch_idx == self.hparams.validation_img_log_idx:
            self.logger.experiment.log(
                {
                    "test/log_imgs": [
                        wandb.Image(img) for img in gen_logging_image(
                            batch_data={
                                **batch,
                                "preds": preds,
                                "probs": probs,
                            },
                            num_log_imgs=self.hparams.num_log_imgs,
                            vmax=self.hparams.vmax
                        )
                    ]
                },
                step=self.global_step
            )


if __name__ == "__main__":
    from .components.radio_cnn import RADIO_CNN
    _ = FrontierSegmentationLitModule(
        net=RADIO_CNN(
            model_version="c-radio_v3-b",
            adaptor_version=None,
        ),
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.StepLR,
        compile=False,
        pred_threshold=0.5,
        num_log_imgs=4,
        validation_img_log_idx=0
    )
