"""BirdCLEF modeling package."""

from birdclef.modeling.ast.model import ASTClassifier
from birdclef.modeling.conformer.model import (
    ConformerClassifier,
    ConformerModelForPreTraining,
)
from birdclef.modeling.conv1d.model import Conv1DClassifier
from birdclef.modeling.conv2d.model import Conv2DClassifier
from birdclef.modeling.convnext.model import ConvNextV2Classifier
from birdclef.modeling.efficientformer.model import EfficientFormerClassifier
from birdclef.modeling.efficientnet.model import EfficientNetClassifier
from birdclef.modeling.efficientvit.model import EfficientViTClassifier
from birdclef.modeling.encodec.model import EncodecClassifier
from birdclef.modeling.gem.model import GeMClassifier
from birdclef.modeling.nfnet.model import NFNetClassifier
from birdclef.modeling.rexnet.model import ReXNetClassifier
from birdclef.modeling.swiftformer.model import SwiftFormerClassifier
from birdclef.modeling.vit.model import ViTClassifier
from birdclef.modeling.wavlm.model import WavLMClassifier

__all__ = [
    "Conv2DClassifier",
    "ConvNextV2Classifier",
    "EfficientFormerClassifier",
    "EfficientNetClassifier",
    "EfficientViTClassifier",
    "EncodecClassifier",
    "WavLMClassifier",
    "ConformerClassifier",
    "ConformerModelForPreTraining",
    "NFNetClassifier",
    "GeMClassifier",
    "SwiftFormerClassifier",
    "Conv1DClassifier",
    "ReXNetClassifier",
    "ViTClassifier",
    "ASTClassifier",
]
