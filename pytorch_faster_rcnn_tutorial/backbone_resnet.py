import logging
from enum import Enum
from typing import Dict, List, Optional

import torch
import torchvision.models as models
from torch import nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork

from pytorch_faster_rcnn_tutorial.CoordAttention import CoordinateAttention

logger: logging.Logger = logging.getLogger(__name__)


class ResNetBackbones(Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"

class CA(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=32):
        super().__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, in_dim // reduction)
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0).cuda()
        self.bn1 = nn.BatchNorm2d(hidden_dim).cuda()
        self.act = nn.ReLU(inplace=True).cuda()
        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0).cuda()
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0).cuda()
        
    def forward(self, x):
        dtype=torch.cuda.FloatTensor
        x=torch.autograd.Variable(x.type(dtype))
        identity = x
        b,c,h,w = x.shape
        x_h = self.pool_h(x).cuda()
        x_w = self.pool_w(x).transpose(-1, -2).cuda()
        y = torch.cat([x_h, x_w], dim=2).cuda()
        y = self.conv1(y).cuda()
        y = self.bn1(y).cuda()
        y = self.act(y).cuda()
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(-1, -2)
        a_h = self.conv_h(x_h).cuda()
        a_w = self.conv_w(x_w).cuda()
        out = identity * a_h * a_w
        return out


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(BackboneWithFPN, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.body = IntermediateLayerGetter(model=backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        #x = torch.randn(2, 64, 32, 32)
        b,c,h,w = x.shape
        attn = CA(c, c)
        x = attn(x)
        x = x.cuda()

        x = self.body(x)
        for key,value in x.items():
          x[key] = value.cuda()
        
        
        x = self.fpn(x)
        return x


def get_resnet_backbone(backbone_name: ResNetBackbones) -> torch.nn.Sequential:
    """
    Returns a resnet backbone pretrained on ImageNet.
    Removes the average-pooling layer and the linear layer at the end.
    """
    pretrained_model, out_channels = None, None

    if backbone_name == ResNetBackbones.RESNET18:
        pretrained_model = models.resnet18(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == ResNetBackbones.RESNET34:
        pretrained_model = models.resnet34(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == ResNetBackbones.RESNET50:
        pretrained_model = models.resnet50(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == ResNetBackbones.RESNET101:
        pretrained_model = models.resnet101(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == ResNetBackbones.RESNET152:
        pretrained_model = models.resnet152(pretrained=True, progress=False)
        out_channels = 2048

    if not pretrained_model and not out_channels:
        raise ValueError(
            f"Your backbone_name is {backbone_name}, "
            f"but should be one of the following:"
            f"{[i.name for i in list(ResNetBackbones)]}"
        )

    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    backbone.out_channels = out_channels

    return backbone


def get_resnet_fpn_backbone(
    backbone_name: ResNetBackbones, pretrained: bool = True, trainable_layers: int = 5
) -> BackboneWithFPN:
    """
    Returns a resnet backbone with fpn pretrained on ImageNet.
    """
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        trainable_layers=trainable_layers,
    )

    backbone.out_channels = 256
    return backbone


def resnet_fpn_backbone(
    backbone_name: ResNetBackbones,
    pretrained: bool,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: List[int] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
):
    # Slight adaptation from the original pytorch vision package
    # Changes: Removed extra_blocks parameter - This parameter invokes LastLevelMaxPool(), which I don't need
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Arguments:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    backbone = resnet.__dict__[backbone_name.value](
        pretrained=pretrained, norm_layer=norm_layer
    )

    if torch.cuda.is_available():
        backbone = backbone.cuda()

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    #modelBackbone = BackboneWithFPN(
    #    backbone=backbone,
    #    return_layers=return_layers,
    #    in_channels_list=in_channels_list,
    #    out_channels=out_channels,
    #    extra_blocks=extra_blocks,
    #)
    #if torch.cuda.is_available():
    #    modelBackbone.cuda()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return BackboneWithFPN(
        backbone=backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=extra_blocks,
    ).to(dev)