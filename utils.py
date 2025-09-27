import math
import torch
import torch.nn.functional as F

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted

def kl_divergence_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    calculate KL divergence loss
    """
    # norm
    output_norm = output - output.mean()
    target_norm = target - target.mean()

    output_log_prob = F.log_softmax(output_norm, dim=1)
    target_prob = F.softmax(target_norm, dim=1)

    loss = F.kl_div(output_log_prob, target_prob, reduction='batchmean')
    return loss


def _compute_laplacian(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Compute the 2D Laplacian with periodic boundary conditions using central differences."""
    dx_tensor = field.real.new_tensor(dx)
    dy_tensor = field.real.new_tensor(dy)

    laplace_x = (
        torch.roll(field, shifts=-1, dims=-1)
        - 2.0 * field
        + torch.roll(field, shifts=1, dims=-1)
    ) / (dx_tensor ** 2)

    laplace_y = (
        torch.roll(field, shifts=-1, dims=-2)
        - 2.0 * field
        + torch.roll(field, shifts=1, dims=-2)
    ) / (dy_tensor ** 2)

    return laplace_x + laplace_y


def _prepare_wavenumber(field: torch.Tensor, wavenumber) -> torch.Tensor:
    """Broadcast wavenumber tensors to match the field dimensionality."""
    if isinstance(wavenumber, torch.Tensor):
        k_tensor = wavenumber.to(field.real.dtype).to(field.device)
    else:
        k_tensor = field.real.new_tensor(wavenumber)

    if k_tensor.ndim == 0:
        k_tensor = k_tensor.view(1, 1, 1, 1)

    expand_shape = [1] * (field.ndim - k_tensor.ndim) + list(k_tensor.shape)
    k_tensor = k_tensor.view(*expand_shape)
    return k_tensor


def evaluate_wave_residual(
    field: torch.Tensor,
    dx: float,
    dy: float,
    wavenumber,
    noise_std: float = 0.0,
    phase_std: float = 0.0,
    num_samples: int = 1,
) -> torch.Tensor:
    """Estimate weak-form residuals of the Helmholtz equation for a complex field."""
    if not torch.is_complex(field):
        raise TypeError("Optical field for residual evaluation must be a complex tensor.")

    k_tensor = _prepare_wavenumber(field, wavenumber)
    area_element = field.real.new_tensor(dx * dy)

    residual_accum = field.real.new_zeros(())

    for _ in range(max(1, num_samples)):
        perturbed = field

        if noise_std > 0.0:
            noise_real = torch.randn_like(field.real)
            noise_imag = torch.randn_like(field.imag)
            complex_noise = torch.complex(noise_real, noise_imag)
            perturbed = perturbed + noise_std * complex_noise

        if phase_std > 0.0:
            phase_noise = phase_std * torch.randn_like(field.real)
            phase_factor = torch.complex(torch.cos(phase_noise), torch.sin(phase_noise))
            perturbed = perturbed * phase_factor

        laplacian = _compute_laplacian(perturbed, dx, dy)
        residual = laplacian + (k_tensor ** 2) * perturbed
        residual_density = torch.abs(residual) ** 2

        sample_residual = residual_density.mean(dim=(-1, -2)) * area_element
        residual_accum = residual_accum + sample_residual.mean()

    return residual_accum / max(1, num_samples)


def compute_optical_residual_penalty(
    model: torch.nn.Module,
    dx: float,
    dy: float,
    noise_std: float = 0.0,
    phase_std: float = 0.0,
    num_samples: int = 1,
) -> torch.Tensor:
    """Aggregate residual penalties from stored optical module outputs."""
    residual_values = []

    for block in getattr(model, "DD", []):
        field = getattr(block, "_last_output", None)
        wavenumber = getattr(block, "wavenumber", None)

        if field is None or wavenumber is None:
            continue

        residual_values.append(
            evaluate_wave_residual(
                field,
                dx=dx,
                dy=dy,
                wavenumber=wavenumber,
                noise_std=noise_std,
                phase_std=phase_std,
                num_samples=num_samples,
            )
        )

    if not residual_values:
        device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
        return torch.zeros((), device=device)

    stacked = torch.stack(residual_values)
    return stacked.mean()
