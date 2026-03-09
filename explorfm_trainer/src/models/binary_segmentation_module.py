from typing import Any, Dict, Tuple

import torch
import wandb
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.segmentation import MeanIoU
from .components.radio_utils import gen_logging_image

class BinarySegmentationLitModule(LightningModule):
    """`LightningModule` for Binary Semantic Segmentation.
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
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            ignore=["net"],
            logger=False
        )

        self.net = net

        # loss function
        self.criterion = torch.nn.BCELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.phases = ["train", "val", "test"]
        for phase in self.phases:
            setattr(self, f"{phase}_acc", Accuracy(task="binary"))
            setattr(self, f"{phase}_loss", MeanMetric())
            setattr(self, f"{phase}_miou", MeanIoU())
            setattr(self, f"{phase}_f1", BinaryF1Score(threshold=pred_threshold))

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_iou_best = MaxMetric()

        self.pred_threshold = pred_threshold

        self.strict_loading=strict_loading

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_miou.reset()
        self.val_f1.reset()
        self.val_acc_best.reset()
        self.val_iou_best.reset()

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
        x = batch["raw_img"]
        y = batch["gt_traversability"]
        probs = self.forward(x)
        loss = self.criterion(probs, y.float())

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
        targets = batch["gt_traversability"]

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
                                "raw_img": batch["raw_img"],
                                "gt_segmentation": batch["gt_img"],
                                "gt_traversability": batch["gt_traversability"],
                                "preds": preds,
                                "probs": probs.detach(),
                                "img_path": batch["img_path"],
                            },
                            seg_colormap=self.trainer.train_dataloader.dataset.seg_colormap,
                            num_log_imgs=self.hparams.num_log_imgs,
                            vmax=self.hparams.vmax
                        )
                    ]
                },
                step=self.global_step
            )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs = self.model_step(batch)
        targets = batch["gt_traversability"]

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
                                "raw_img": batch["raw_img"],
                                "gt_segmentation": batch["gt_img"],
                                "gt_traversability": batch["gt_traversability"],
                                "preds": preds,
                                "probs": probs,
                                "img_path": batch["img_path"],
                            },
                            seg_colormap=self.trainer.val_dataloaders.dataset.seg_colormap,
                            num_log_imgs=self.hparams.num_log_imgs,
                            vmax=self.hparams.vmax
                        )
                    ]
                },
                step=self.global_step
            )


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        self.val_iou_best(self.val_miou.compute())
        self.log("val/miou_best", self.val_iou_best.compute(), sync_dist=True, prog_bar=True)


    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs = self.model_step(batch)
        targets = batch["gt_traversability"]

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
                                "raw_img": batch["raw_img"],
                                "gt_segmentation": batch["gt_img"],
                                "gt_traversability": batch["gt_traversability"],
                                "preds": preds,
                                "probs": probs,
                                "img_path": batch["img_path"],
                            },
                            seg_colormap=self.trainer.test_dataloaders.dataset.seg_colormap,
                            num_log_imgs=self.hparams.num_log_imgs,
                            vmax=self.hparams.vmax
                        )
                    ]
                },
                step=self.global_step
            )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose optimizers and learning-rate schedulers to use in optimization.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    from .components.radio_cnn import RADIO_CNN
    _ = BinarySegmentationLitModule(
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
