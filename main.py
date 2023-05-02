import argparse
import os
import platform
import time

import cv2
import numpy as np
import ultralytics
from ultralytics.yolo.engine.results import Boxes

import ldc
import scrfd
import utils


def angle(pt1, pt0, pt2):
    # 두 벡터의 각도 코사인 세타를 계산, 90도에 가까울수록 0이다.
    # 음수를 방지하기 위해 절댓값을 계산한다.
    v1 = [pt1[0] - pt0[0], pt1[1] - pt0[1]]
    v2 = [pt2[0] - pt0[0], pt2[1] - pt0[1]]
    cos_theta = np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.abs(cos_theta)
    return cos_theta


def filter_rectangles(quadrangles: list[np.ndarray], angle_threshold: float) -> list[np.ndarray]:
    rectangles = []
    for quad in quadrangles:
        quad = quad.squeeze()
        cos_theta = max([angle(quad[(k - 1) % 4], quad[k % 4], quad[(k + 1) % 4]) for k in range(1, 5)])
        if cos_theta < angle_threshold:
            rectangles.append(np.expand_dims(quad, 1))
    return rectangles


def find_rectangles(edge: np.ndarray) -> list[np.ndarray]:
    # Post-process edge map
    morph = cv2.morphologyEx(edge, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    cv2.imshow('morph', morph)

    # Find contours (+remove noise)
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = [contour for contour in contours if cv2.contourArea(contour) > (edge.shape[0] * edge.shape[1] * 0.12)]

    # Approximate quadrangles (+remove noise)
    quadrangles = [cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.06, True) for contour in contours]
    quadrangles = [quad for quad in quadrangles if quad.shape == (4, 1, 2)]

    # Filter rectangles only
    rectangles = filter_rectangles(quadrangles, 0.18)

    # Remove largest rectangle
    if len(rectangles) > 1:
        i = np.argmax([cv2.contourArea(rect) for rect in rectangles])
        del rectangles[i]

    return rectangles


def edge_is_spoofing(rectangles: list, bbox: list, threshold: int) -> bool:
    # 얼굴이 2개 이상 검출되면 fake로 간주
    if len(bbox) > 1:
        return True

    # 검출한 사각형의 중심점과 얼굴 bbox의 중심점 간의 거리가 임계값 미만이면 fake로 간주
    bbox_center = np.array([bbox[0][0] + (bbox[0][2] - bbox[0][0]) / 2, bbox[0][1] + (bbox[0][3] - bbox[0][1]) / 2])
    for quad in rectangles:
        assert quad.shape == (4, 1, 2), f'Invalid quadrangle shape: {quad.shape}'

        moments = cv2.moments(quad.squeeze(1))
        quadrangle_center = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
        distance = np.linalg.norm(bbox_center - quadrangle_center)
        print(f'edge_{distance=:.2f}')
        if distance < threshold:
            return True
    return False


def yolo_is_spoofing(od_pred: Boxes, bbox: list, threshold: int) -> bool:
    # 얼굴이 2개 이상 검출되면 fake로 간주
    if len(bbox) > 1:
        return True

    # 검출한 bbox의 중심점과 얼굴 bbox의 중심점 간의 거리가 임계값 미만이면 fake로 간주
    bbox_center = np.array([bbox[0][0] + (bbox[0][2] - bbox[0][0]) / 2, bbox[0][1] + (bbox[0][3] - bbox[0][1]) / 2])
    for od_pred_one in od_pred:
        xywh = od_pred_one.xywh.cpu().numpy()
        od_center = np.array([xywh[0][0] + (xywh[0][2] - xywh[0][0]) / 2, xywh[0][1] + (xywh[0][3] - xywh[0][1]) / 2])
        distance = np.linalg.norm(bbox_center - od_center)
        print(f'yolo_{distance=:.2f}')
        if distance < threshold:
            return True
    return False


def convert_spoofing_to_string(spoofing: bool) -> str:
    assert isinstance(spoofing, bool)

    if spoofing:
        return 'Fake'
    else:
        return 'Real'


# Global parameters
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='1', help='file/dir/webcam')
    parser.add_argument('--device', type=str, default='cuda', help='[cpu, cuda, openvino, tensorrt]')
    parser.add_argument('--line-thickness', type=int, default=2, help='drawing thickness (pixels)')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--draw-fps', action='store_true', help='Draw fps on the frame.')
    args = parser.parse_args()
    print(args)

    # Load detector
    edge_detector = ldc.LDC('onnx_files/ldc_b4.onnx', args.device)
    face_detector = scrfd.SCRFD('onnx_files/scrfd_2.5g_bnkps.onnx', 0.3, 0.5, args.device)
    yolo = ultralytics.YOLO('torch_files/yolov8n-smartphone.pt')

    # Inference
    # source: webcam or video
    if args.source.isnumeric() or args.source.lower().endswith(VID_FORMATS):
        if args.source.isnumeric():
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(args.source))
        else:
            cap = cv2.VideoCapture(args.source)
        assert cap.isOpened()

        if args.source.isnumeric():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        count = 1
        accumulated_time = 0
        fps = 0
        while cv2.waitKey(5) != ord('q'):
            # Load image
            ret, img = cap.read()
            assert ret, 'no frame has been grabbed.'

            # Detect edge
            start = time.perf_counter()
            pred = face_detector.detect_one(img)
            accumulated_time += (time.perf_counter() - start)
            if count % 10 == 0:
                fps = 1 / (accumulated_time / 10)
                accumulated_time = 0
            count += 1

            # Draw FPS
            if args.draw_fps:
                cv2.putText(img, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw prediction
            if pred is not None:
                bbox, conf, kps = face_detector.parse_prediction(pred)

                edge = edge_detector.detect_one(img)
                rectangles = find_rectangles(edge)
                od_pred = yolo.predict(img, conf=0.6)[0].boxes

                edge_spoofing = edge_is_spoofing(rectangles, bbox, 80)
                yolo_spoofing = yolo_is_spoofing(od_pred, bbox, 95)
                spoofing = convert_spoofing_to_string(edge_spoofing or yolo_spoofing)

                utils.draw_prediction(img, bbox, conf, None, args.line_thickness, args.hide_conf)
                cv2.drawContours(img, rectangles, -1, (0, 0, 255), 2)
                cv2.putText(img, f'{spoofing}', (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f'{spoofing}', (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (255, 255, 255), 1, cv2.LINE_AA)

            # Show prediction
            cv2.imshow('FAS', img)

        print('Quit inference.')
        cap.release()
        cv2.destroyAllWindows()

    # source: image
    elif args.source.lower().endswith(IMG_FORMATS):
        assert os.path.exists(args.source), f'Image not found: {args.source}'

        # Load image
        img: np.ndarray = cv2.imread(args.source)
        assert img is not None

        # Detect edge
        pred = edge_detector.detect_one(img)
        edge = edge_detector.detect_one(img)
        rectangles = find_rectangles(img, edge)

        # Save prediction
        cv2.imwrite('result.jpg', img)
        print('Save result to "result.jpg"')
    else:
        raise ValueError(f'Wrong source: {args.source}')
