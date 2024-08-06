import torch
from backbone import Backbone, Joiner
from position_encoding import PositionEmbeddingSine
from deformable_transformer import DeformableTransformer
from mdef_detr import MDefDETR
from modulated_detection import ModulatedDetection


class Model:
    """
    This class initiates the specified model.
    """

    def __init__(self, checkpoint):
        model = MDefDETR(
            backbone=Joiner(
                Backbone(name="resnet101", train_backbone=True, return_interm_layers=True, dilation=False),
                PositionEmbeddingSine(num_pos_feats=128, normalize=True),
            ),
            transformer=DeformableTransformer(
                text_encoder_type="roberta-base",
                d_model=256,
                num_feature_levels=4,
                dim_feedforward=1024,
                return_intermediate_dec=False,
            ),
            num_classes=255,
            num_queries=300,
            num_feature_levels=4,
        )
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        self.model = ModulatedDetection(model)

    def get_model(self):
        """
        This function returns the selected models
        """
        return self.model
