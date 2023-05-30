import os

import cv2
import numpy as np
import onnxruntime

import utils


class LDC:
    def __init__(self, model_path: str, device: str):
        """
        Args:
            model_path: Model file path.
            device: Device to inference.
        """
        assert os.path.exists(model_path), f'model_path is not exists: {model_path}'
        assert device in ['cpu', 'cuda', 'openvino', 'tensorrt'], f'device is invalid: {device}'

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

    def _transform_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resizes the input image to fit img_size while preserving aspect ratio.
        This performs BGR to RGB, HWC to CHW, normalization, and adding batch dimension.
        (mean=(123.68, 116.779, 103.939), std=(1, 1, 1))
        """
        img, scale = utils.resize_preserving_aspect_ratio(img, self.img_size)

        pad = (0, self.img_size - img.shape[0], 0, self.img_size - img.shape[1])
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.dnn.blobFromImage(img, 1, img.shape[:2][::-1], (123.68, 116.779, 103.939), swapRB=True)
        return img

    def detect_one(self, img: np.ndarray) -> np.ndarray:
        original_shape = img.shape
        img = self._transform_image(img)
        edge_map = self.session.run(None, {self.input_name: img})[0]
        # 패딩 영역 제거 및 패딩 edge 제거
        edge_map = edge_map[:original_shape[0] - 2, :original_shape[1] - 2]
        return edge_map
