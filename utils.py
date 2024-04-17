from typing import Tuple

import cv2
import numpy as np


def resize_preserving_aspect_ratio(img: np.ndarray, img_size: int, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
    # Resize preserving aspect ratio. scale_ratio is the scaling ratio of the img_size.
    h, w = img.shape[:2]
    scale = img_size // scale_ratio / max(h, w)
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    return img, scale


def clip_coords(boxes: np.ndarray, shape: tuple) -> np.ndarray:
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clip(0, shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, shape[0])  # y2
    boxes[:, 5:15:2] = boxes[:, 5:15:2].clip(0, shape[1])  # x axis
    boxes[:, 6:15:2] = boxes[:, 6:15:2].clip(0, shape[0])  # y axis
    return boxes


def xyxy2xywh(bbox: np.ndarray) -> np.ndarray:
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
    return bbox


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywh2xywh(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    return y


def resize_divisible(img: np.ndarray, img_size: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = img_size / min(h, w)
    h *= scale
    w *= scale
    h = round(h / 64) * 64
    w = round(w / 64) * 64
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
    return img, scale


def draw_prediction(img: np.ndarray, bbox: np.ndarray, conf: np.ndarray, landmarks: np.ndarray = None, thickness=2,
                    hide_conf=False):
    # Draw prediction on the image. If the landmarks is None, only draw the bbox.
    assert img.ndim == 3, f'img dimension is invalid: {img.ndim}'
    assert img.dtype == np.uint8, f'img dtype must be uint8, got {img.dtype}'
    assert img.shape[-1] == 3, 'Pass BGR images. Other Image formats are not supported.'
    assert len(bbox) == len(conf), 'bbox and conf must be equal length.'
    if landmarks is None:
        landmarks = [None] * len(bbox)
    assert len(bbox) == len(conf) == len(landmarks), 'bbox, conf, and landmarks must be equal length.'

    bbox_color = (0, 255, 0)
    conf_color = (0, 255, 0)
    landmarks_colors = ((0, 165, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
    for bbox_one, conf_one, landmarks_one in zip(bbox, conf, landmarks):
        # Draw bbox
        x1, y1, x2, y2 = bbox_one
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness, cv2.LINE_AA)

        # Text confidence
        if not hide_conf:
            cv2.putText(img, f'{conf_one:.2f}', (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, conf_color, thickness, cv2.LINE_AA)

        # Draw landmarks
        if landmarks_one is not None:
            for point_x, point_y, color in zip(landmarks_one[::2], landmarks_one[1::2], landmarks_colors):
                cv2.circle(img, (point_x, point_y), 2, color, cv2.FILLED)


def put_border_text(img: np.ndarray, text: str, org: tuple[int, int], font_scale: float,
                    color=(0, 0, 0), border_color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, border_color, 1, cv2.LINE_AA)
