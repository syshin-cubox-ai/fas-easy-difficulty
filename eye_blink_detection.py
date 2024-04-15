import pathlib
import platform
import time

import cv2
import numpy as np
from spiga.demo.visualize.plotter import Plotter
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

import utils
import yolov8

FILE = pathlib.Path(__file__).resolve(strict=True)
ROOT = FILE.parent


class EyeBlinkDetector:
    def __init__(self):
        # Detector settings
        self.face_detector = yolov8.YOLOv8(str(ROOT.joinpath("openvino_files", "yolov8n-face-int8.xml")),
                                           0.5, 0.7, "cpu")
        self.face_aligner = SPIGAFramework(ModelConfig("wflw"))
        self.ear_thres = 0.21
        self.ear_continuous_frames = 2

        # Member variables
        self._counter = 0
        self._total = 0

    def predict(self, img: np.ndarray) -> tuple[int, np.ndarray]:
        # 얼굴 검출
        pred = self.face_detector.detect_one(img)
        if pred is None:
            return self._total, img

        # 가장 큰 얼굴 하나만 남기기
        pred = utils.leave_max_area_item(pred)
        bbox, _ = self.face_detector.parse_prediction(pred)

        # 얼굴 포즈 추정
        features = self.face_aligner.inference(img, utils.xyxy2xywh(bbox.copy()))
        landmarks = features["landmarks"]

        # Eye Aspect Ratio 계산
        left_eye = landmarks[0][60:68]
        right_eye = landmarks[0][68:76]
        left_EAR = self.compute_eye_aspect_ratio(left_eye)
        right_EAR = self.compute_eye_aspect_ratio(right_eye)
        EAR = (left_EAR + right_EAR) / 2

        # 눈 깜빡임 판별
        if EAR < self.ear_thres:
            self._counter += 1
        else:
            if self._counter >= self.ear_continuous_frames:
                self._total += 1
            self._counter = 0

        # Plot (bbox, text, landmarks)
        plotted_img = img.copy()
        x1, y1, x2, y2 = bbox[0]
        cv2.rectangle(plotted_img, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plotted_img, f"{EAR=:.4f} Blinks={self._total}", (x1, y1 - 2), cv2.FONT_HERSHEY_DUPLEX,
                    0.58, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(plotted_img, f"{EAR=:.4f} Blinks={self._total}", (x1, y1 - 2), cv2.FONT_HERSHEY_DUPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
        plotted_img = Plotter().landmarks.draw_landmarks(plotted_img, landmarks[0][60:76], thick=1)

        return self._total, plotted_img

    @staticmethod
    def compute_eye_aspect_ratio(eye: list[list[float, float]]) -> float:
        eye = np.array(eye)

        # EAR = ||P1 - P5|| + ||P2 - P4|| / 2 * ||P0 - P3||
        # a = np.linalg.norm(eye[1] - eye[5])
        # b = np.linalg.norm(eye[2] - eye[4])
        # c = np.linalg.norm(eye[0] - eye[3])
        # EAR = (a + b) / (2 * c)

        # EAR = ||P1 - P7|| + ||P2 - P6|| + ||P3 - P5|| / 3 * ||P0 - P4||
        a = np.linalg.norm(eye[1] - eye[7])
        b = np.linalg.norm(eye[2] - eye[6])
        c = np.linalg.norm(eye[3] - eye[5])
        d = np.linalg.norm(eye[0] - eye[4])
        EAR = (a + b + c) / (3 * d)
        return EAR


def main():
    eye_blink_detector = EyeBlinkDetector()

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    assert cap.isOpened()

    count = 1
    accumulated_time = 0
    fps = 0
    while cv2.waitKey(3) != ord("q"):
        ret, img = cap.read()
        assert ret, "no frame has been grabbed."

        start = time.perf_counter()
        total, plotted_img = eye_blink_detector.predict(img)
        accumulated_time += (time.perf_counter() - start)
        if count % 10 == 0:
            fps = 1 / (accumulated_time / 10)
            accumulated_time = 0
        count += 1

        # Draw FPS
        cv2.putText(plotted_img, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                    0.66, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(plotted_img, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                    0.66, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Blink Detection", plotted_img)
    cap.release()


if __name__ == "__main__":
    main()
