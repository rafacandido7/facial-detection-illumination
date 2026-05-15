import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import dlib

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = _SRC_DIR.parent / "models" / "face_detection_yunet_2023mar.onnx"


def download_yunet():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("Baixando modelo YuNet (~2 MB)...")
    tmp = MODEL_PATH.with_suffix(".onnx.tmp")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        shutil.move(tmp, MODEL_PATH)
        print("Modelo salvo em", MODEL_PATH)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Falha ao baixar YuNet: {exc}") from exc


def apply_clahe(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def init_detectors() -> dict:
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    hog = dlib.get_frontal_face_detector()
    yunet = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (1, 1))
    return {"haar": haar, "hog": hog, "yunet": yunet}


def detect_haar(img, cascade) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    elapsed = (time.perf_counter() - t0) * 1000
    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces] if len(faces) > 0 else []
    return boxes, elapsed


def detect_hog(img, detector) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    dets = detector(gray, 1)
    elapsed = (time.perf_counter() - t0) * 1000
    boxes = [(d.left(), d.top(), d.right(), d.bottom()) for d in dets]
    return boxes, elapsed


def detect_yunet(img, net) -> tuple:
    h, w = img.shape[:2]
    net.setInputSize((w, h))
    t0 = time.perf_counter()
    _, faces = net.detect(img)
    elapsed = (time.perf_counter() - t0) * 1000
    if faces is None:
        return [], elapsed
    boxes = [
        (int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])) for f in faces
    ]
    return boxes, elapsed
