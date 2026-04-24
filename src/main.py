import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import dlib
import numpy as np

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")
DEFAULT_IMAGE = (
    "dataset/lfw/lfw-deepfunneled/lfw-deepfunneled/Abba_Eban/Abba_Eban_0001.jpg"
)


def download_yunet():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("Baixando modelo YuNet (~2 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo salvo em", MODEL_PATH)


def aplicar_clahe(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def detect_haar(img):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    elapsed = (time.perf_counter() - t0) * 1000
    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces] if len(faces) > 0 else []
    return boxes, elapsed


def detect_hog(img):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    dets = detector(gray, 1)
    elapsed = (time.perf_counter() - t0) * 1000
    boxes = [(d.left(), d.top(), d.right(), d.bottom()) for d in dets]
    return boxes, elapsed


def detect_yunet(img):
    h, w = img.shape[:2]
    net = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (w, h))
    t0 = time.perf_counter()
    _, faces = net.detect(img)
    elapsed = (time.perf_counter() - t0) * 1000
    if faces is None:
        return [], elapsed
    boxes = [
        (int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])) for f in faces
    ]
    return boxes, elapsed


def _banner(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def run_pipeline(img, label):
    haar_boxes, t_haar = detect_haar(img)
    hog_boxes, t_hog = detect_hog(img)
    yunet_boxes, t_yunet = detect_yunet(img)

    n_haar = len(haar_boxes)
    n_hog = len(hog_boxes)
    n_yunet = len(yunet_boxes)

    print(f"\n  [{label}]")
    print(f"    Haar Cascade : {n_haar} face(s)  {t_haar:6.1f} ms")
    print(f"    HOG + SVM    : {n_hog} face(s)  {t_hog:6.1f} ms")
    print(f"    YuNet        : {n_yunet} face(s)  {t_yunet:6.1f} ms")

    vis = img.copy()
    for x1, y1, x2, y2 in haar_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 80, 0), 2)   # azul
    for x1, y1, x2, y2 in hog_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)    # verde
    for x1, y1, x2, y2 in yunet_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 2)    # vermelho

    return _banner(vis, label)


def main():
    parser = argparse.ArgumentParser(
        description="Comparação Haar | HOG+SVM | YuNet com e sem CLAHE"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help="Caminho para a imagem de entrada",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        download_yunet()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Erro: imagem não encontrada — {args.image}", file=sys.stderr)
        sys.exit(1)

    img_clahe = aplicar_clahe(img)

    sep = "=" * 52
    print(sep)
    print("  Detecção Facial — Haar | HOG+SVM | YuNet")
    print("  Azul = Haar   Verde = HOG   Vermelho = YuNet")
    print(sep)

    panel_orig = run_pipeline(img, "Sem CLAHE")
    panel_clahe = run_pipeline(img_clahe, "Com CLAHE")

    h = max(panel_orig.shape[0], panel_clahe.shape[0])
    w = panel_orig.shape[1]
    divider = np.full((h, 6, 3), 80, dtype=np.uint8)

    combined = np.hstack([
        cv2.resize(panel_orig, (w, h)),
        divider,
        cv2.resize(panel_clahe, (w, h)),
    ])

    print(f"\n{sep}")
    print("  Pressione qualquer tecla para fechar.")
    print(sep)

    cv2.imshow(
        "Haar=azul  HOG=verde  YuNet=vermelho  |  esq: sem CLAHE  |  dir: com CLAHE",
        combined,
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
