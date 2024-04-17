import platform
import time

import cv2
import numpy as np
from spiga.demo.visualize.plotter import Plotter
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from ultralytics import YOLO

import utils


class EyeBlinkDetector:
    def __init__(self):
        # Settings
        self.face_detector = YOLO("onnx_files/yolov8n-faceperson.onnx", "pose")
        self.face_aligner = SPIGAFramework(ModelConfig("wflw"))
        self.face_detector_conf = 0.35
        self.face_detector_iou = 0.7
        self.ear_thres = 0.2
        self.ear_continuous_frames = 2

        # Internal variables
        self._counter = 0
        self._number_blinks = 0

    def predict(self, img: np.ndarray, plot=False) -> tuple[bool, int]:
        start = time.perf_counter()
        is_blinked = False

        # 얼굴 검출
        r = self.face_detector.predict(img, conf=self.face_detector_conf, iou=self.face_detector_iou,
                                       device="cuda", classes=[0])[0]
        if len(r) > 0:
            # 가장 큰 얼굴 하나만 남기기
            bbox = self.leave_max_area_item(r.boxes.xyxy.cpu().numpy())

            # 얼굴 포즈 추정
            features = self.face_aligner.inference(img, utils.xyxy2xywh(bbox.copy()))

            # 형식 변환
            bbox = bbox.squeeze(0).round().astype(np.int32)
            landmarks = np.array(features["landmarks"]).squeeze(0)

            # Eye Aspect Ratio 계산
            left_eye = landmarks[60:68]
            right_eye = landmarks[68:76]
            left_eye_ear = self.compute_eye_aspect_ratio(left_eye)
            right_eye_ear = self.compute_eye_aspect_ratio(right_eye)
            ear = (left_eye_ear + right_eye_ear) / 2

            # 눈 깜빡임 판별
            if ear < self.ear_thres:
                self._counter += 1
            else:
                if self._counter >= self.ear_continuous_frames:
                    is_blinked = True
                    self._number_blinks += 1
                self._counter = 0
            fps = 1 / (time.perf_counter() - start)

            if plot:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
                utils.put_border_text(img, f"ear={ear:.3f} blinks={self._number_blinks}", (x1, y1 - 2), 0.56)
                img = Plotter().landmarks.draw_landmarks(img, landmarks[60:76], thick=1)
                utils.put_border_text(img, f"{fps:.2f} FPS", (10, 26), 0.6)

        return is_blinked, self._number_blinks

    @staticmethod
    def leave_max_area_item(pred: np.ndarray) -> np.ndarray:
        area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        max_idx = np.argmax(area)
        pred = np.expand_dims(pred[max_idx], 0)
        return pred

    @staticmethod
    def compute_eye_aspect_ratio(eye: np.ndarray) -> float:
        # EAR = ||P1 - P7|| + ||P2 - P6|| + ||P3 - P5|| / 3 * ||P0 - P4||
        a = np.linalg.norm(eye[1] - eye[7])
        b = np.linalg.norm(eye[2] - eye[6])
        c = np.linalg.norm(eye[3] - eye[5])
        d = np.linalg.norm(eye[0] - eye[4])
        ear = (a + b + c) / (3 * d)
        return ear


def main():
    eye_blink_detector = EyeBlinkDetector()

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    assert cap.isOpened()

    while cv2.waitKey(3) != ord("q"):
        ret, img = cap.read()
        assert ret, "no frame has been grabbed."

        is_blinked, number_blinks = eye_blink_detector.predict(img, plot=True)

        cv2.imshow("0", img)
    cap.release()


if __name__ == "__main__":
    main()
