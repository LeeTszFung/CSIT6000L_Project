import setup
from detector import FaceDetector
from aligner import FaceAligner
from warper import FaceWarper
from blender import FaceBlender
import cv2
import numpy as np
import scipy.spatial as spatial

def test_blender(filein1, filein2, size, percent=0.5):
    img1 = cv2.imread(filein1 + '.jpg')
    img2 = cv2.imread(filein2 + '.jpg')
    detector = FaceDetector()
    aligner = FaceAligner()
    warper = FaceWarper()
    blender = FaceBlender()

    points_set1 = detector.points(img1)
    points_set2 = detector.points(img2)
    if len(points_set2) != 1:
        raise RuntimeError("Dest image can only have one face")
    points2 = points_set2[0]
    img2, points2 = aligner.align(img2, points2, size)
    face_index = 0
    for points1 in points_set1:
        face_index = face_index + 1
        img1, points1 = aligner.align(img1, points1, size)
        weighted = detector.weighted(points1, points2)
        face1 = warper.warp(img1, points1, weighted, size)
        face2 = warper.warp(img2, points2, weighted, size)

        blended = blender.weighted(face1, face2, percent)
        cv2.imwrite('blender/face' + str(face_index) + '_weighted_' + str(percent) + '.jpg', blended)

        mask = blender.mask(weighted, blended.shape[:2])
        cv2.imwrite('blender/face' + str(face_index) + '_mask.jpg', mask)
        if percent <= 0.5:
            s = max(size)
            blur_radius = int(2 * s * percent) + 10
            blur = cv2.blur(mask, (blur_radius, blur_radius))
            cv2.imwrite('blender/face' + str(face_index) + '_blur.jpg', blur)
            alpha = blender.alpha_feathering(face2, face1, mask, blur_radius=blur_radius)
            cv2.imwrite('blender/face' + str(face_index) + '_alpha_' + str(percent) + '.jpg', alpha)
        else:
            s = max(size)
            percent = 1 - percent 
            blur_radius = int(2 * s * percent) + 10
            blur = cv2.blur(mask, (blur_radius, blur_radius))
            cv2.imwrite('blender/face' + str(face_index) + '_blur.jpg', blur)
            alpha = blender.alpha_feathering(face1, face1, mask, blur_radius=blur_radius)
            cv2.imwrite('blender/face' + str(face_index) + '_alpha_' + str(1 - percent) + '.jpg', alpha)

        trans = np.dstack((blended, mask))
        cv2.imwrite('blender/face' + str(face_index) + '_trans.jpg', trans)

        back = blender.weighted(img1, img2, percent)
        cv2.imwrite('blender/face' + str(face_index) + '_back.jpg', back)
        overlay = blender.overlay(trans, back, mask)
        cv2.imwrite('blender/face' + str(face_index) + '_overlay.jpg', overlay)

        poisson = blender.poisson(trans, img1, mask)
        cv2.imwrite('blender/face' + str(face_index) +'_poisson.jpg', poisson)

if __name__ == '__main__':
    test_blender('male', 'female', (600, 500), percent=0.5)

