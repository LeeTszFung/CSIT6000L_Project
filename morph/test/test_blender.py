import setup
from detector import FaceDetector
from aligner import FaceAligner
from warper import FaceWarper
from blender import FaceBlender
import cv2
import numpy as np
import scipy.spatial as spatial

def test_blender(filein1, filein2, percent=0.5):
    size = (600, 500)
    img1 = cv2.imread(filein1)
    img2 = cv2.imread(filein2)
    detector = FaceDetector()
    aligner = FaceAligner()
    warper = FaceWarper()
    blender = FaceBlender()

    points1 = detector.points(img1)
    points2 = detector.points(img2)
    img1, points1 = aligner.align(img1, points1, size)
    img2, points2 = aligner.align(img2, points2, size)
    weighted = detector.weighted(points1, points2)
    face1 = warper.warp(img1, points1, weighted, size)
    face2 = warper.warp(img2, points2, weighted, size)

    blended = blender.weighted(face1, face2, percent)
    cv2.imwrite('face_blended.jpg', blended)
    mask = blender.mask(weighted, blended.shape[:2])
    cv2.imwrite('face_blended_mask.jpg', mask)
    blended = np.dstack((blended, mask))
    cv2.imwrite('face_blended_trans.jpg', blended)

    blended_back = blender.weighted(img1, img2, percent)
    cv2.imwrite('face_blended_back.jpg', blended_back)
    overlay = blender.overlay(blended, blended_back, mask)
    cv2.imwrite('face_blended_overlay.jpg', overlay)

    poisson = blender.poisson(face2, face1, mask)
    cv2.imwrite('face_blended_poisson.jpg', poisson)

    alpha = blender.alpha_feathering(face2, face1, mask)
    cv2.imwrite('face_blended_alpha.jpg', alpha)

if __name__ == '__main__':
#    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#        fileout = 'blended_' + str(percent) + '.jpg'
#        test_blender('face.jpg', 'face2.jpg', fileout, percent)
    test_blender('face.jpg', 'face2.jpg', percent=0.5)

