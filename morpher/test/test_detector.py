import setup
from detector import FaceDetector
import cv2

def plot_points(filein, radius=1):
    img = cv2.imread(filein + '.jpg')
    detector = FaceDetector()
    points_set = detector.points(img)
    for points in points_set:
        # red for 68 points detected
        for point in points[:68]:
            for x in range(point[1] - radius, point[1] + radius):
                for y in range(point[0] - radius, point[0] + radius):
                    img[x, y] = (0, 0, 255)
        # green for broudaries added
        for point in points[68:]:
            for x in range(point[1] - radius, point[1] + radius):
                for y in range(point[0] - radius, point[0] + radius):
                    img[x, y] = (0, 255, 0)

    cv2.imwrite('detector/' + filein + '_points.jpg', img)

if __name__ == '__main__':
    plot_points('male', radius=2)
    plot_points('female', radius=3)
    plot_points('multi', radius=1)
