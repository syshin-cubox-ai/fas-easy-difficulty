import os

import cv2
import numpy as np
import onnxruntime


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
        self.input_name = session_input.name

    def _transform_image(self, img: np.ndarray) -> np.ndarray:
        """
        This performs BGR to RGB, HWC to CHW, normalization, and adding batch dimension.
        (mean=(123.68, 116.779, 103.939), std=(1, 1, 1))
        """
        assert img.shape[0] % 8 == 0 and img.shape[1] % 8 == 0, 'Image must be divisible by 2^3=8'

        img = cv2.dnn.blobFromImage(img, 1, img.shape[:2][::-1], (123.68, 116.779, 103.939), swapRB=True)
        return img

    def detect_one(self, img: np.ndarray) -> np.ndarray:
        img = self._transform_image(img)
        edge = self.session.run(None, {self.input_name: img})[0]
        return edge
