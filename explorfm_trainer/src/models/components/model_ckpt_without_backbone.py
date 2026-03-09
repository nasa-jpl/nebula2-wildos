from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

class ModelCheckpointWithoutBackbone(ModelCheckpoint):
    """Custom ModelCheckpoint that does not save the backbone weights.

    This is useful when you want to save only the head of the model, e.g., for transfer learning.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the custom ModelCheckpoint."""
        self.backbone_name = kwargs.get('backbone_name', 'backbone')
        kwargs.pop('backbone_name', None)
        super().__init__(*args, **kwargs)


    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Override to remove backbone weights from the checkpoint."""
        new_state_dict = {}
        for key in list(checkpoint['state_dict'].keys()):
            if self.backbone_name not in key:
                new_state_dict[key] = checkpoint['state_dict'][key]
            else:
                log.warning(f"Ignoring backbone weights: {key}")
        checkpoint['state_dict'] = new_state_dict

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)