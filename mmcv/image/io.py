# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import cv2
import numpy as np

from mmcv.opencv_info import USE_OPENCV2
from mmcv.utils import check_file_exist, is_str, mkdir_or_exist

from turbojpeg import TJCS_RGB, TJCS_GRAY, TJPF_BGR, TJPF_GRAY, TJPF_BGRA, TurboJPEG

jpeg = TurboJPEG()

if not USE_OPENCV2:
    from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
else:
    from cv2 import CV_LOAD_IMAGE_COLOR as IMREAD_COLOR
    from cv2 import CV_LOAD_IMAGE_GRAYSCALE as IMREAD_GRAYSCALE
    from cv2 import CV_LOAD_IMAGE_UNCHANGED as IMREAD_UNCHANGED

turbo_pixel_format = {'color': TJPF_BGR, 'grayscale': TJPF_GRAY, 'unchanged': TJPF_BGRA}


def imread(img_or_path, flag='color'):
    """Read an image.

    Args:
        img_or_path (ndarray or str): Either a numpy array or image path.
            If it is a numpy array (loaded image), then it will be returned
            as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        pixel_format = turbo_pixel_format[flag] if is_str(flag) else flag
        check_file_exist(img_or_path, 'img file does not exist: {}'.format(img_or_path))

        #change to turbojpeg
        with open(img_or_path, 'rb') as in_file:
            img = jpeg.decode(in_file.read(), pixel_format)
        return img
    else:
        raise TypeError('"img" must be a numpy array or a filename')


def imfrombytes(content, flag='color'):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.

    Returns:
        ndarray: Loaded image array.
    """
    # change to TurboJPEG
    pixel_format = turbo_pixel_format[flag] if is_str(flag) else flag
    img = jpeg.decode(content, pixel_format)
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = osp.abspath(osp.dirname(file_path))
        mkdir_or_exist(dir_name)
    return cv2.imwrite(file_path, img, params)
