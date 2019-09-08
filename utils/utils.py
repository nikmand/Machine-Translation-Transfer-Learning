"""Utilities."""

import os
import sys

import torch
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))


def to_device(t, device='cpu', dtype=torch.float32):
    """Copy torch tensor to device and convert the data type

    If the tensor is already in the specified device the same tensor is
    returned without copying.

    Args:
        t (torch.Tensor): The tensor to be converted
        device (str): 'cpu' or 'gpu'
        dtype: The type of the resulting tensor. The tensor is casted to the
            specified type. See (https://pytorch.org/docs/stable/tensors.html)
            for a comprehensive listing of available types.

    Returns:
        (torch.Tensor): Tensor on the specified device with specified type.
    """
    if t.device.type != device:
        return t.to(device=device, dtype=dtype)
    return t
