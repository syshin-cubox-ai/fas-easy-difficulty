import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime

from utils import clip_coords, resize_preserving_aspect_ratio


class SCRFD:
    def __init__(self, model_path: str, conf_thres: float, iou_thres: float, device: str):
        """
        Args:
            model_path: Model file path.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold.
            device: Device to inference.
        """
        assert os.path.exists(model_path), f'model_path is not exists: {model_path}'
        assert 0 <= conf_thres <= 1, f'conf_thres must be between 0 and 1: {conf_thres}'
        assert 0 <= iou_thres <= 1, f'iou_thres must be between 0 and 1: {iou_thres}'
        assert device in ['cpu', 'cuda', 'openvino', 'tensorrt'], f'device is invalid: {device}'

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.mode = None
        self.session = None
        self.input_name = None
        self.request = None
        if os.path.splitext(model_path)[-1] == '.onnx':
            self.mode = 'onnx'
            if device == 'cpu':
                providers = ['CPUExecutionProvider']
            elif device == 'cuda':
                providers = ['CUDAExecutionProvider']
            elif device == 'openvino':
                providers = ['OpenVINOExecutionProvider']
            elif device == 'tensorrt':
                providers = ['TensorrtExecutionProvider']
            else:
                raise ValueError(f'device is invalid: {device}')
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            session_input = self.session.get_inputs()[0]
            assert session_input.shape[2] == session_input.shape[3], 'The input shape must be square.'
            self.img_size = session_input.shape[2]
            self.input_name = session_input.name

        elif os.path.splitext(model_path)[-1] == '.xml':
            self.mode = 'openvino'
            import openvino.runtime
            core = openvino.runtime.Core()
            compiled_model = core.compile_model(model_path, 'CPU')
            self.request = compiled_model.create_infer_request()
            input_shape = self.request.inputs[0].shape
            assert input_shape[2] == input_shape[3], 'The input shape must be square.'
            self.img_size = input_shape[2]
        else:
            raise ValueError(f'Wrong file extension: {os.path.splitext(model_path)[-1]}')

    def _transform_image(self, img: np.ndarray, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
        """
        Resizes the input image to fit img_size while preserving aspect ratio.
        This performs BGR to RGB, HWC to CHW, normalization, and adding batch dimension.
        (mean=(127.5, 127.5, 127.5), std=(128.0, 128.0, 128.0))
        """
        img, scale = resize_preserving_aspect_ratio(img, self.img_size, scale_ratio)

        pad = (0, self.img_size - img.shape[0], 0, self.img_size - img.shape[1])
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.dnn.blobFromImage(img, 1 / 128, img.shape[:2][::-1], (127.5, 127.5, 127.5), swapRB=True)
        return img, scale

    def _non_max_suppression(self, pred: np.ndarray) -> np.ndarray:
        # 임계값 초과로 confidence를 가진 항목만 남김
        keep = np.where(pred[:, 4] > self.conf_thres)[0]
        pred = pred[keep]
        if pred.size == 0:
            return pred

        # 속도 개선을 위해, 추후 IoU 계산에 사용할 각 bbox의 넓이를 한 번에 계산해둠
        x1 = pred[:, 0]
        y1 = pred[:, 1]
        x2 = pred[:, 2]
        y2 = pred[:, 3]
        score = pred[:, 4]
        area = (x2 - x1) * (y2 - y1)

        order = score.argsort()[::-1]
        keep = []
        while order.size > 0:
            # confidence가 가장 높은 항목(order[0])을 선택하고, 최종 출력 리스트에 추가
            highest_conf_item = order[0]
            keep.append(highest_conf_item.item())

            # confidence가 가장 높은 항목을 기준으로 나머지 모든 항목에 대해 IoU를 구함
            rest_item = order[1:]
            inter_x1 = np.maximum(x1[highest_conf_item], x1[rest_item])
            inter_y1 = np.maximum(y1[highest_conf_item], y1[rest_item])
            inter_x2 = np.minimum(x2[highest_conf_item], x2[rest_item])
            inter_y2 = np.minimum(y2[highest_conf_item], y2[rest_item])
            w = np.maximum(0.0, inter_x2 - inter_x1)
            h = np.maximum(0.0, inter_y2 - inter_y1)
            intersection = w * h
            union = area[highest_conf_item] + area[rest_item] - intersection
            iou = intersection / union

            # 임계값 미만으로 iou를 가진 항목만 남김
            item = np.where(iou < self.iou_thres)[0]
            order = order[item + 1]
        pred = pred[keep, :]
        return pred

    def _inference(self, img: np.ndarray) -> np.ndarray:
        if self.mode == 'onnx':
            pred = self.session.run(None, {self.input_name: img})[0]
        elif self.mode == 'openvino':
            pred = self.request.infer({0: img}).popitem()[1]
        else:
            raise ValueError(f'Wrong mode: {self.mode}')
        pred = self._non_max_suppression(pred)
        return pred

    def _padded_detect_one(self, img: np.ndarray) -> Optional[np.ndarray]:
        original_img_shape = img.shape[:2]
        transformed_img, scale = self._transform_image(img, scale_ratio=1.5)
        pred = self._inference(transformed_img)
        if pred.shape[0] > 0:
            # Rescale coordinates from inference size to input image size
            pred[:, :4] /= scale
            pred[:, 5:] /= scale
            pred = clip_coords(pred, original_img_shape)
            return pred
        else:
            return None

    def detect_one(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform face detection on a single image.
        Args:
            img: Input image read using OpenCV. (HWC, BGR)
        Return:
            pred:
                Post-processed prediction. Shape=(number of faces, 15)
                15 is composed of bbox coordinates(4), object confidence(1), and keypoints coordinates(10).
                The coordinate format is x1y1x2y2 (bbox), xy per point (keypoints).
                The unit is image pixel.
                If no face is detected, output None.
        """
        original_img_shape = img.shape[:2]
        transformed_img, scale = self._transform_image(img)
        pred = self._inference(transformed_img)
        if pred.shape[0] > 0:
            # Rescale coordinates from inference size to input image size
            pred[:, :4] /= scale
            pred[:, 5:] /= scale
            pred = clip_coords(pred, original_img_shape)
            return pred
        else:
            return None

    def parse_prediction(self, pred: np.ndarray) -> Tuple[List, List, List]:
        """Parse prediction to bbox, confidence, and keypoints."""
        bbox = pred[:, :4].round().astype(np.int32).tolist()
        conf = pred[:, 4].tolist()
        kps = pred[:, 5:].round().astype(np.int32).tolist()
        return bbox, conf, kps
