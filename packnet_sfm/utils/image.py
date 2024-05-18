# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import torch
import torch.nn.functional as F
from functools import lru_cache
from PIL import Image
import numpy as np
import random

from packnet_sfm.utils.misc import same_shape


def load_image(path):
    """
    Read an image using PIL

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    image : PIL.Image
        Loaded image
    """
    return Image.open(path)


def write_image(filename, image):
    """
    Write an image to file.

    Parameters
    ----------
    filename : str
        File where image will be saved
    image : np.array [H,W,3]
        RGB image
    """
    cv2.imwrite(filename, image[:, :, ::-1])


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

def flip_model(model, image, flip):
    """
    Flip input image and flip output inverse depth map

    Parameters
    ----------
    model : nn.Module
        Module to be used
    image : torch.Tensor [B, 3, H, W]
        Input image
    flip : bool
        True if the flip is happening

    Returns
    -------
    inv_depths : list of torch.Tensor [B,1,H,W]
        List of predicted inverse depth maps
    """
    if flip:
        inv_depths_feature = model(flip_lr(image))
        return [flip_lr(inv_depth) for inv_depth in inv_depths_feature]
    else:
        return model(image)


########################################################################################################################

def gradient_x(image):
    """
    Calculates the gradient of an image in the x dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_x : torch.Tensor [B,3,H,W-1]
        Gradient of image with respect to x
    """
    return image[:, :, :, :-1] - image[:, :, :, 1:]

def gradient_y(image):
    """
    Calculates the gradient of an image in the y dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_y : torch.Tensor [B,3,H-1,W]
        Gradient of image with respect to y
    """
    return image[:, :, :-1, :] - image[:, :, 1:, :]

########################################################################################################################

def interpolate_image(image, shape, mode='bilinear', align_corners=True):
    """
    Interpolate an image to a different resolution

    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Image to be interpolated
    shape : tuple (H, W)
        Output shape
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    image : torch.Tensor [B,?,H,W]
        Interpolated image
    """
    # Take last two dimensions as shape
    if len(shape) > 2:
        shape = shape[-2:]
    # If the shapes are the same, do nothing
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        # Interpolate image to match the shape
        return F.interpolate(image, size=shape, mode=mode,
                                 align_corners=align_corners)

def interpolate_scales(images, shape=None, mode='bilinear', align_corners=False):
    """
    Interpolate list of images to the same shape

    Parameters
    ----------
    images : list of torch.Tensor [B,?,?,?]
        Images to be interpolated, with different resolutions
    shape : tuple (H, W)
        Output shape
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    images : list of torch.Tensor [B,?,H,W]
        Interpolated images, with the same resolution
    """
    # If no shape is provided, interpolate to highest resolution
    if shape is None:
        shape = images[0].shape
    # Take last two dimensions as shape
    if len(shape) > 2:
        shape = shape[-2:]
    # Interpolate all images
    return [F.interpolate(image, shape, mode=mode,
                              align_corners=align_corners) for image in images]

def match_scales(image, targets, num_scales,
                 mode='bilinear', align_corners=True):
    """
    Interpolate one image to produce a list of images with the same shape as targets

    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Input image
    targets : list of torch.Tensor [B,?,?,?]
        Tensors with the target resolutions
    num_scales : int
        Number of considered scales
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    images : list of torch.Tensor [B,?,?,?]
        List of images with the same resolutions as targets
    """
    # For all scales
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        # If image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            # Otherwise, interpolate
            images.append(interpolate_image(
                image, target_shape, mode=mode, align_corners=align_corners))
    # Return scaled images
    return images

def match_feature_scales(image, targets, num_scales,
                 mode='bilinear', align_corners=True):
    """
    Interpolate one image to produce a list of images with the same shape as targets

    Parameters
    ----------
    image : list of torch.Tensor [B,?,h,w]
        Input image
    targets : torch.Tensor [B,?,?,?]
        Tensors with the target resolutions
    num_scales : int
        Number of considered scales
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    images : list of torch.Tensor [B,?,?,?]
        List of images with the same resolutions as targets
    """
    # For all scales
    images = []
    target_shape = targets.shape
    for i in range(num_scales):
        image_shape = image[i].shape[-2:]
        # If image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image[i])
        else:
            # Otherwise, interpolate
            images.append(interpolate_image(
                image[i], target_shape, mode=mode, align_corners=align_corners))
    # Return scaled images
    return images
    
########################################################################################################################

@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """
    Create meshgrid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    xs : torch.Tensor [B,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [B,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])

@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    """
    Create an image grid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    grid : torch.Tensor [B,3,H,W]
        Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid

########################################################################################################################
