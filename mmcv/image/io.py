# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import cv2
import numpy as np

from mmcv.opencv_info import USE_OPENCV2
from mmcv.utils import check_file_exist, is_str, mkdir_or_exist

if not USE_OPENCV2:
    from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
else:
    from cv2 import CV_LOAD_IMAGE_COLOR as IMREAD_COLOR
    from cv2 import CV_LOAD_IMAGE_GRAYSCALE as IMREAD_GRAYSCALE
    from cv2 import CV_LOAD_IMAGE_UNCHANGED as IMREAD_UNCHANGED

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}

imread_backend = 'cv2'

def usebackend(backend):
    """Select a backend for imread

    Args:
        backend (str): The image decoding backend type. Options are "cv2" and
            "turbojpeg".
    """
    global imread_backend
    imread_backend = backend


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
        if imread_backend == 'turbojpeg'
            from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
            jpeg = TurboJPEG()
            turbo_pixel_format = {'color': TJCS_RGB, 'grayscale': TJPF_GRAY}

            if flag == 'color' or flag == 'grayscale'
                pixel_format = turbo_pixel_format[flag]
                check_file_exist(img_or_path, 'img file does not exist: {}'.format(img_or_path))
                with open(img_or_path, 'rb') as in_file:
                    img = jpeg.decode(in_file.read(), pixel_format)
                return img
            else
                raise ValueError("flag must be 'color' or 'grayscale'")
        else
            flag = imread_flags[flag] if is_str(flag) else flag
            check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
            return cv2.imread(img_or_path, flag)
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
    if imread_backend == 'turbojpeg':
        from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
        jpeg = TurboJPEG()
        turbo_pixel_format = {'color': TJCS_RGB, 'grayscale': TJPF_GRAY}

        if flag == 'color' or flag == 'grayscale':
            pixel_format = turbo_pixel_format[flag]
            img = jpeg.decode(content, pixel_format)
            return img
        else:
            raise ValueError('flag must be "color" or "grayscale".')
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
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
