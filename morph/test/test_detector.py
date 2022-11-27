import setup
from detector import FaceDetector
import cv2

def plot_points(filein, fileout, size):
    img = cv2.imread(filein)
    detector = FaceDetector()
    points = detector.points(img)
    for point in points[:68]:
        for x in range(point[1] - size, point[1] + size):
            for y in range(point[0] - size, point[0] + size):
                img[x, y] = (0, 0, 255)
    for point in points[68:]:
        for x in range(point[1] - size, point[1] + size):
            for y in range(point[0] - size, point[0] + size):
                img[x, y] = (0, 255, 0)

    cv2.imwrite(fileout, img)

if __name__ == '__main__':
    plot_points('face.jpg', 'face_points.jpg', size=4)
