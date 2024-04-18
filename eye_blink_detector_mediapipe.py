import platform
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import utils


class EyeBlinkDetector:
    def __init__(self, device="cpu"):
        # Settings
        base_options = python.BaseOptions(
            model_asset_path="mediapipe_files/face_landmarker.task",
            delegate=python.BaseOptions.Delegate.CPU if device == "cpu" else python.BaseOptions.Delegate.GPU,
        )
        self.mp_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
            )
        )
        self.ear_thres = 0.2
        self.ear_continuous_frames = 2

        # Internal variables
        self._counter = 0
        self._number_blinks = 0

    def predict(self, img: np.ndarray, plot=False) -> tuple[bool, int]:
        start = time.perf_counter()
        is_blinked = False

        # 얼굴 랜드마크 검출
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.mp_detector.detect(mp_image)

        if len(result.face_landmarks) > 0:
            # 형식 변환
            landmarks = np.array(result.face_landmarks[0])
            left_eye = landmarks[[33, 160, 159, 158, 133, 153, 145, 144]]
            right_eye = landmarks[[362, 385, 386, 387, 263, 373, 374, 380]]
            left_eye = np.array([[int(round(i.x * img.shape[1])), int(round(i.y * img.shape[0]))] for i in left_eye])
            right_eye = np.array([[int(round(i.x * img.shape[1])), int(round(i.y * img.shape[0]))] for i in right_eye])

            # Eye Aspect Ratio 계산
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

            # Plot (labels, landmarks, fps)
            if plot:
                utils.put_border_text(img, f"ear={ear:.3f} blinks={self._number_blinks}", (10, 46), 0.56)
                for xy in np.concatenate([left_eye, right_eye], 0):
                    cv2.circle(img, (xy[0], xy[1]), 1, (0, 255, 0), -1)
                utils.put_border_text(img, f"{fps:.2f} FPS", (10, 26), 0.6)

        return is_blinked, self._number_blinks

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
