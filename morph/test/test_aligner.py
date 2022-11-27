import setup
from detector import FaceDetector
from aligner import FaceAligner
import cv2

def plot_alignment(filein, fileout, size):
    img = cv2.imread(filein)
    detector = FaceDetector()
    points = detector.points(img)
    aligner = FaceAligner()
    aligned, points = aligner.align(img, points, size)
    cv2.imwrite(fileout, aligned)

if __name__ == '__main__':
    plot_alignment('face.jpg', 'face_align.jpg', (500, 500))
