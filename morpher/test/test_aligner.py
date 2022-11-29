import setup
from detector import FaceDetector
from aligner import FaceAligner
import cv2

def plot_alignment(filein, size):
    img = cv2.imread(filein + '.jpg')
    detector = FaceDetector()
    points_set = detector.points(img)
    aligner = FaceAligner()
    face_index = 1
    for points in points_set:
        aligned, points = aligner.align(img, points, size)
        cv2.imwrite('aligner/' + filein + '_face' + str(face_index) + '.jpg', aligned)
        face_index = face_index + 1

if __name__ == '__main__':
    plot_alignment('female', (600, 500))
    plot_alignment('male', (600, 500))
    plot_alignment('multi', (150, 100))
