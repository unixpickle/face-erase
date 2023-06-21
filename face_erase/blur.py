import torch
import torch.nn.functional as F


def adaptive_blur(
    image: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 10,
    kernel_size: int = 13,
    sigma: float = 5.0,
) -> torch.Tensor:
    """
    :param image: an [N x C x H x W] image.
    :param mask: an [N x 1 x H x W] mask where 0 means the pixel should be
                 blurred.
    """
    assert iterations >= 1, "must apply at least one blurring iteration"
    assert kernel_size % 2 == 1, "kernel size must be odd"

    in_dtype = image.dtype
    image = image.float()
    mask = mask.float()

    # Create a Gaussian blur kernel.
    dists = torch.arange(
        -(kernel_size // 2),
        kernel_size // 2 + 1,
        dtype=image.dtype,
        device=image.device,
    )
    dists = dists[None, None, :, None] ** 2 + dists[None, None, None, :] ** 2
    kernel = (-dists / (2 * sigma**2)).exp()
    kernel /= kernel.sum()
    kernel = kernel.repeat(3, 1, 1, 1)

    # Not only apply blur repeatedly, but also expand the mask.
    for _ in range(iterations):
        mask_blurred = F.conv2d(
            F.pad(mask, (kernel_size // 2,) * 4, mode="reflect"),
            kernel,
            groups=mask.shape[1],
        )
        mask = torch.minimum(mask, mask_blurred)

        blurred = F.conv2d(
            F.pad(image, (kernel_size // 2,) * 4, mode="reflect"),
            kernel,
            groups=image.shape[1],
        )
        image = mask * image + (1 - mask) * blurred

    return image.to(in_dtype)
