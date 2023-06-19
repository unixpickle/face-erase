import torch
import torch.nn.functional as F


def adaptive_blur(
    image: torch.Tensor,
    mask: torch.Tensor,
    iterations: int,
    rate: float = 0.5,
) -> torch.Tensor:
    """
    :param image: an [N x C x H x W] image.
    :param mask: an [N x 1 x H x W] mask where 0 means the pixel should be
                 blurred.
    """
    assert iterations >= 1, "must apply at least one blurring iteration"
    edge_rate = (1 - rate) / 4
    kernel = (
        torch.tensor(
            [[0, edge_rate, 0], [edge_rate, rate, edge_rate], [0, edge_rate, 0]],
            device=image.device,
            dtype=image.dtype,
        )
        .view(1, 1, 3, 3)
        .repeat(image.shape[1], 1, 1, 1)
    )
    mask = mask.float()
    for _ in range(iterations):
        blurred = F.conv2d(
            F.pad(image, (1, 1, 1, 1), mode="reflect"), kernel, groups=image.shape[1]
        )
        mask_blurred = F.conv2d(
            F.pad(mask, (1, 1, 1, 1), mode="reflect"), kernel, groups=mask.shape[1]
        )
        image = mask * image + (1 - mask) * blurred
        mask = torch.minimum(mask, mask_blurred)
    return image
