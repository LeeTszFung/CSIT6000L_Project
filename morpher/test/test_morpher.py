import setup
from morpher import FaceMorpher
import cv2

if __name__ == '__main__':
    size = (600, 500)
    morpher = FaceMorpher('male.jpg', 'female.jpg', size)
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        weighted = morpher.morph(percent, blend='weighted')
        cv2.imwrite('morpher/weighted_' + str(percent) + '.jpg', weighted[0])
        alpha = morpher.morph(percent, blend='alpha')
        cv2.imwrite('morpher/alpha_' + str(percent) + '.jpg', alpha[0])
    overlay = morpher.morph(0.5, bg='overlay')
    cv2.imwrite('morpher/overlay_0.5.jpg', overlay[0])
    poisson = morpher.morph(0.5, bg='poisson')
    cv2.imwrite('morpher/poisson_0.5.jpg', poisson[0])

