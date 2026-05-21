import json
import os
import sys


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from torch import nn

from src.models.slot_attn import SlotAttention, SoftPositionEmbed


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _build_cnn_layers(in_channels, enc_channels, encode_depth, kernel_size=3):
    """Convolutional feature extractor shared by both encoders.

    A stem conv followed by ``encode_depth`` residual blocks, each halving the
    spatial resolution and doubling the channel count. Returns the layer list,
    the output channel count and the (square) output resolution given a square
    input of ``image_size``.
    """
    activ = nn.ReLU
    layers = [
        nn.Conv2d(
            in_channels, enc_channels, kernel_size=kernel_size, stride=1, padding=1
        ),
        activ(),
    ]
    enc = enc_channels
    for _ in range(encode_depth):
        layers.append(BasicBlock(enc, enc))
        layers.append(activ())
        layers.append(nn.Conv2d(enc, 2 * enc, kernel_size, stride=2, padding=1))
        layers.append(activ())
        enc *= 2
    return layers, enc


def _reduced_size(image_size, encode_depth):
    """Spatial resolution after ``encode_depth`` stride-2 convs (ceil-halving)."""
    reduced = image_size
    for _ in range(encode_depth):
        reduced = (1 + reduced) // 2
    return reduced


class LatentEncoder(ModelMixin, ConfigMixin):
    """Naive baseline encoder: CNN feature map flattened + linearly read out
    into K slot vectors. No slot attention -- kept intentionally as the
    baseline the SlotAttentionEncoder is compared against (see ROADMAP.md).
    """

    def __init__(
        self,
        in_channels=3,
        enc_channels=128,
        num_components=4,
        image_size=128,
        latent_dim=64,
    ):
        super().__init__()
        self.num_components = num_components
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.latent_dim_expand = self.latent_dim * self.num_components
        kernel_size = 3
        encode_depth = 3
        out_dim = self.latent_dim_expand

        reduced_size = image_size
        for i in range(encode_depth):
            reduced_size = (1 + reduced_size) // 2  # halved (ceiling) after each conv
        activ = nn.ReLU
        layers = [
            nn.Conv2d(
                in_channels, enc_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            activ(),
        ]
        enc = enc_channels

        for i in range(encode_depth):
            layers.append(BasicBlock(enc, enc))
            layers.append(activ())
            layers.append(nn.Conv2d(enc, 2 * enc, kernel_size, stride=2, padding=1))
            layers.append(activ())
            enc *= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(enc * reduced_size * reduced_size, out_dim))
        layers.append(activ())  # can experiment with removing

        self.latent_encoder = nn.Sequential(*layers)

    def forward(self, x):
        slots = self.latent_encoder(x)
        slots = slots.view(slots.shape[0], self.num_components, self.latent_dim)
        return slots


class SlotAttentionEncoder(ModelMixin, ConfigMixin):
    """Object-centric encoder: CNN feature map -> Slot Attention -> slots.

    Keeps the convolutional feature extractor of ``LatentEncoder`` (image ->
    feature map) but replaces its Flatten + Linear slot read-out with a proper
    Slot Attention module. The feature map is augmented with a soft positional
    embedding, flattened to a token sequence, projected to ``latent_dim`` and
    bound into ``num_components`` slots by iterative attention.

    Drop-in for ``LatentEncoder``: ``forward`` returns ``[B, num_components,
    latent_dim]`` slots, with ``latent_dim`` matching the UNet cross-attention
    dim. This is the canonical Latent Slot Diffusion pattern.
    """

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        enc_channels=128,
        num_components=4,
        image_size=128,
        latent_dim=64,
        encode_depth=3,
        slot_iters=3,
        slot_hidden_dim=128,
    ):
        super().__init__()
        self.num_components = num_components
        self.image_size = image_size
        self.latent_dim = latent_dim

        # CNN: image -> feature map [B, cnn_channels, feat_size, feat_size].
        layers, cnn_channels = _build_cnn_layers(
            in_channels, enc_channels, encode_depth
        )
        self.cnn = nn.Sequential(*layers)
        feat_size = _reduced_size(image_size, encode_depth)

        # Slots localise only through attention -- inject spatial position.
        self.pos_embed = SoftPositionEmbed(cnn_channels, (feat_size, feat_size))

        # Per-token read-in: normalise, then project feature dim -> latent_dim.
        self.layer_norm = nn.LayerNorm(cnn_channels)
        self.mlp = nn.Sequential(
            nn.Linear(cnn_channels, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        self.slot_attention = SlotAttention(
            num_slots=num_components,
            dim=latent_dim,
            iters=slot_iters,
            hidden_dim=slot_hidden_dim,
        )

    def forward(self, x):
        features = self.cnn(x)  # [B, C, h, w]
        features = features.permute(0, 2, 3, 1)  # [B, h, w, C]
        features = self.pos_embed(features)
        features = features.flatten(1, 2)  # [B, h*w, C]
        features = self.mlp(self.layer_norm(features))  # [B, h*w, latent_dim]
        slots = self.slot_attention(features)  # [B, num_components, latent_dim]
        return slots


# --- Encoder selection -------------------------------------------------------
# Both encoders share the [B, num_components, latent_dim] output contract, so
# they are interchangeable. Which one a run uses is decided by the
# ``_class_name`` field of the latent-encoder config json.

ENCODER_REGISTRY = {
    "LatentEncoder": LatentEncoder,
    "SlotAttentionEncoder": SlotAttentionEncoder,
}

# For isinstance() checks that need to tell an encoder from the UNet.
LATENT_ENCODER_CLASSES = tuple(ENCODER_REGISTRY.values())


def _encoder_class_from_name(class_name):
    if class_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown latent encoder '{class_name}'. "
            f"Expected one of {sorted(ENCODER_REGISTRY)}."
        )
    return ENCODER_REGISTRY[class_name]


def build_latent_encoder(config_path):
    """Build a latent encoder from a config json, dispatching on ``_class_name``.

    Falls back to ``LatentEncoder`` when ``_class_name`` is absent so older
    configs keep working.
    """
    with open(config_path) as f:
        class_name = json.load(f).get("_class_name", "LatentEncoder")
    cls = _encoder_class_from_name(class_name)
    return cls.from_config(cls.load_config(config_path))


def load_latent_encoder(ckpt_path):
    """Load a trained latent encoder from a checkpoint folder.

    The training save hook stores each model under a subfolder named after its
    lowercased class name, so the encoder type is recovered by probing for the
    subfolder that exists.
    """
    for class_name, cls in ENCODER_REGISTRY.items():
        subfolder = class_name.lower()
        if os.path.isdir(os.path.join(ckpt_path, subfolder)):
            return cls.from_pretrained(ckpt_path, subfolder=subfolder)
    raise ValueError(
        f"No known latent encoder subfolder ({sorted(c.lower() for c in ENCODER_REGISTRY)}) "
        f"found under {ckpt_path}."
    )
