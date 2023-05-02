from typing import Union, Tuple

import cv2
import numpy as np
import torch


def resize_preserving_aspect_ratio(img: np.ndarray, img_size: int, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
    # Resize preserving aspect ratio. scale_ratio is the scaling ratio of the img_size.
    h, w = img.shape[:2]
    scale = img_size // scale_ratio / max(h, w)
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    return img, scale


def clip_coords(boxes: Union[torch.Tensor, np.ndarray], shape: tuple) -> Union[torch.Tensor, np.ndarray]:
    # MODIFIED for face detection
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clip(0, shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, shape[0])  # y2
    boxes[:, 5:15:2] = boxes[:, 5:15:2].clip(0, shape[1])  # x axis
    boxes[:, 6:15:2] = boxes[:, 6:15:2].clip(0, shape[0])  # y axis
    return boxes


def draw_prediction(img: np.ndarray, bbox: list, conf: list, kps: list = None, thickness=2, hide_conf=False):
    # Draw prediction on the image. If the keypoints is None, only draw the bbox.
    assert img.ndim == 3, f'img dimension is invalid: {img.ndim}'
    assert img.dtype == np.uint8, f'img dtype must be uint8, got {img.dtype}'
    assert img.shape[-1] == 3, 'Pass BGR images. Other Image formats are not supported.'
    assert len(bbox) == len(conf), 'bbox and conf must be equal length.'
    if kps is None:
        kps = [None] * len(bbox)
    assert len(bbox) == len(conf) == len(kps), 'bbox, conf, and kps must be equal length.'

    bbox_color = (0, 255, 0)
    conf_color = (0, 255, 0)
    kps_colors = ((0, 165, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
    for bbox_one, conf_one, kps_one in zip(bbox, conf, kps):
        # Draw bbox
        x1, y1, x2, y2 = bbox_one
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness, cv2.LINE_AA)

        # Text confidence
        if not hide_conf:
            cv2.putText(img, f'{conf_one:.2f}', (x1, y1 - 2), None, 0.6, conf_color, thickness, cv2.LINE_AA)

        # Draw keypoints
        if kps_one is not None:
            for point_x, point_y, color in zip(kps_one[::2], kps_one[1::2], kps_colors):
                cv2.circle(img, (point_x, point_y), 2, color, cv2.FILLED)
