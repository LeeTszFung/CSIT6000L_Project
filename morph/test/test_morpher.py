import setup
from morpher import FaceMorpher
import cv2

if __name__ == '__main__':
    size = (600, 500)
    morpher = FaceMorpher('face.jpg', 'face2.jpg', size)
    result = morpher.morph(0.5)
    cv2.imwrite('final.jpg', result)