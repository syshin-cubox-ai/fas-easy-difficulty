import argparse
import os
import platform
import time

import cv2
import numpy as np

import ldc
import scrfd
import utils
import yolov8


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


def find_rectangles(edge_map: np.ndarray) -> list[np.ndarray]:
    # Post-process edge map
    morph = cv2.morphologyEx(edge_map, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # Find contours (+remove noise)
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    min_thres = (edge_map.shape[0] * edge_map.shape[1] * 0.12)
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_thres]

    # Approximate quadrangles (+remove noise)
    quadrangles = [cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.06, True) for contour in contours]
    quadrangles = [quad for quad in quadrangles if quad.shape == (4, 1, 2)]

    # Filter rectangles only
    rectangles = filter_rectangles(quadrangles, 0.18)

    # Remove largest rectangle
    if len(rectangles) > 1:
        i = np.argmax([cv2.contourArea(rect) for rect in rectangles])
        del rectangles[i]

    # DEBUG
    cv2.drawContours(img, rectangles, -1, (0, 0, 255), 2)
    return rectangles


def is_paper_spoofing(rectangles: list, face_bbox: list, threshold: int) -> bool:
    # 얼굴이 2개 이상 검출되면 fake로 간주
    if len(face_bbox) > 1:
        return True

    # 검출한 사각형의 중심점과 얼굴 bbox의 중심점 간의 거리가 임계값 미만이면 fake로 간주
    bbox_center = np.array([face_bbox[0][0] + (face_bbox[0][2] - face_bbox[0][0]) / 2,
                            face_bbox[0][1] + (face_bbox[0][3] - face_bbox[0][1]) / 2])
    for quad in rectangles:
        assert quad.shape == (4, 1, 2), f'Invalid quadrangle shape: {quad.shape}'

        moments = cv2.moments(quad.squeeze(1))
        quadrangle_center = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
        distance = np.linalg.norm(bbox_center - quadrangle_center)
        print(f'{distance=:.2f}')  # DEBUG
        if distance < threshold:
            return True
    return False


def is_small_box_inside_large_box(large_box, small_box) -> bool:
    x1, y1, x2, y2 = large_box
    r_x1, r_y1, r_x2, r_y2 = small_box
    if r_x1 >= x1 and r_y1 >= y1 and r_x2 <= x2 and r_y2 <= y2:
        return True
    else:
        return False


def is_display_spoofing(display_bbox: list, face_bbox: list) -> bool:
    # 얼굴이 2개 이상 검출되면 fake로 간주
    if len(face_bbox) > 1:
        return True

    # yolo bbox 안에 얼굴 bbox가 들어있으면 fake로 간주
    for display_bbox_one in display_bbox:
        cv2.rectangle(img, display_bbox_one[:2], display_bbox_one[2:], (0, 255, 255), 2, cv2.LINE_8)  # DEBUG
        if is_small_box_inside_large_box(display_bbox_one, face_bbox[0]):
            return True
    return False


def convert_is_spoofing_to_string(spoofing: bool) -> str:
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
    parser.add_argument('--source', type=str, default='0', help='file/dir/webcam')
    parser.add_argument('--device', type=str, default='cuda', help='[cpu, cuda, openvino, tensorrt]')
    parser.add_argument('--line-thickness', type=int, default=2, help='drawing thickness (pixels)')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--draw-fps', action='store_true', help='Draw fps on the frame.')
    args = parser.parse_args()
    print(args)

    # Load detector
    edge_detector = ldc.LDC('onnx_files/ldc_b4.onnx', args.device)
    face_detector = scrfd.SCRFD('onnx_files/scrfd_2.5g_bnkps.onnx', 0.3, 0.5, args.device)
    display_detector = yolov8.YOLOv8('onnx_files/yolov8n-smartphone.onnx', 0.7, 0.7, args.device)

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
            face_pred = face_detector.detect_one(img)
            edge_map = edge_detector.detect_one(img)
            display_pred = display_detector.detect_one(img)
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
            if face_pred is not None:
                face_bbox, face_conf, _ = face_detector.parse_prediction(face_pred)
                rectangles = find_rectangles(edge_map)
                paper_spoofing = is_paper_spoofing(rectangles, face_bbox, 80)
                if display_pred is not None:
                    display_bbox, display_conf = display_detector.parse_prediction(display_pred)
                    display_spoofing = is_display_spoofing(display_bbox, face_bbox)
                else:
                    display_spoofing = False

                spoofing = convert_is_spoofing_to_string(paper_spoofing or display_spoofing)

                utils.draw_prediction(img, face_bbox, face_conf, None, args.line_thickness, args.hide_conf)
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
        face_pred = face_detector.detect_one(img)
        edge_map = edge_detector.detect_one(img)
        display_pred = display_detector.detect_one(img)
        if face_pred is not None:
            face_bbox, face_conf, _ = face_detector.parse_prediction(face_pred)
            rectangles = find_rectangles(edge_map)
            paper_spoofing = is_paper_spoofing(rectangles, face_bbox, 80)
            if display_pred is not None:
                display_bbox, display_conf = display_detector.parse_prediction(display_pred)
                display_spoofing = is_display_spoofing(display_bbox, face_bbox)
            else:
                display_spoofing = False

            spoofing = convert_is_spoofing_to_string(paper_spoofing or display_spoofing)

            utils.draw_prediction(img, face_bbox, face_conf, None, args.line_thickness, args.hide_conf)
            cv2.drawContours(img, rectangles, -1, (0, 0, 255), 2)
            cv2.putText(img, f'{spoofing}', (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                        0.66, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f'{spoofing}', (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                        0.66, (255, 255, 255), 1, cv2.LINE_AA)

        # Save prediction
        cv2.imwrite('result.jpg', img)
        print('Save result to "result.jpg"')
    else:
        raise ValueError(f'Wrong source: {args.source}')
