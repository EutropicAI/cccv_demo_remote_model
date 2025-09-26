import torch
from torch import nn
from torch.nn import functional as F

from cccv import ARCH_REGISTRY


@ARCH_REGISTRY.register(name="demo_srcnn")
class DEMOSRCNN(nn.Module):
    def __init__(self, num_channels: int = 1, scale: int = 2) -> None:
        super(DEMOSRCNN, self).__init__()
        print("I'm a demo SRCNN architecture, I'm initializing...")
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear")

        if self.num_channels == 1 and x.size(1) == 3:
            # RGB -> YUV
            x = rgb_to_yuv(x)
            y, u, v = x[:, 0:1, ...], x[:, 1:2, ...], x[:, 2:3, ...]

            y = self.relu(self.conv1(y))
            y = self.relu(self.conv2(y))
            y = self.conv3(y)

            x = torch.cat([y, u, v], dim=1)
            # YUV -> RGB
            x = yuv_to_rgb(x)
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)

        return x


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of :math:`(0, 1)`. The range of the output is of
    :math:`(0, 1)` to luma and the ranges of U and V are :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`,
    respectively.

    The YUV model adopted here follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"[CCCV] Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"[CCCV] Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"[CCCV] Input type is not a Tensor. Got {type(image)}")

    if image.dim() < 3 or image.shape[-3] != 3:
        raise ValueError(f"[CCCV] Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    g: torch.Tensor = y + -0.396 * u - 0.581 * v
    b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out


print("New Arch Registered: demo_srcnn")

