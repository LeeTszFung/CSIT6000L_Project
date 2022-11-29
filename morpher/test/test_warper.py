import setup
from detector import FaceDetector
from aligner import FaceAligner
from warper import FaceWarper
import cv2
import numpy as np
import scipy.spatial as spatial

def plot_triangulation(filein, size):
    img = cv2.imread(filein + '.jpg')
    detector = FaceDetector()
    aligner = FaceAligner()
    points_set = detector.points(img)
    face_index = 1
    for points in points_set:
        aligned, points = aligner.align(img, points, size)

        aligned = aligned[:, :, :3]

        delaunay = spatial.Delaunay(points)
        for triangle in delaunay.simplices:
            v1 = points[triangle[0]]
            v2 = points[triangle[1]]
            v3 = points[triangle[2]]
            cv2.line(aligned, v1, v2, (0, 0, 255), 1)
            cv2.line(aligned, v2, v3, (0, 0, 255), 1)
            cv2.line(aligned, v1, v3, (0, 0, 255), 1)
        cv2.imwrite('warper/' + filein + '_face' +str(face_index) + '.jpg', aligned)
        face_index = face_index + 1

def test_warp(filein1, filein2, size):
    img1 = cv2.imread(filein1+'.jpg')
    img2 = cv2.imread(filein2+'.jpg')
    detector = FaceDetector()
    aligner = FaceAligner()
    warper = FaceWarper()

    points1 = detector.points(img1)[0]
    points2 = detector.points(img2)[0]
    img1, points1 = aligner.align(img1, points1, size)
    img2, points2 = aligner.align(img2, points2, size)
    weighted = detector.weighted(points1, points2)
    face1 = warper.warp(img1, points1, weighted, size)
    face2 = warper.warp(img2, points2, weighted, size)
    delaunay = spatial.Delaunay(weighted)
    for triangle in delaunay.simplices:
        v1 = weighted[triangle[0]]
        v2 = weighted[triangle[1]]
        v3 = weighted[triangle[2]]
        cv2.line(face1, v1, v2, (0, 0, 255), 1)
        cv2.line(face1, v2, v3, (0, 0, 255), 1)
        cv2.line(face1, v1, v3, (0, 0, 255), 1)
        cv2.line(face2, v1, v2, (0, 0, 255), 1)
        cv2.line(face2, v2, v3, (0, 0, 255), 1)
        cv2.line(face2, v1, v3, (0, 0, 255), 1)
   
    cv2.imwrite('warper/' + filein1 + '_warped.jpg', face1)
    cv2.imwrite('warper/' + filein2 + '_warped.jpg', face2)
    
def test_avg(filein, size):
    img = cv2.imread(filein+'.jpg')
    detector = FaceDetector()
    aligner = FaceAligner()
    warper = FaceWarper()

    points_set = detector.points(img)
    face_index = 1
    new_points_set = []
    new_img = []
    for points in points_set:
        img_temp, points_temp = aligner.align(img, points, size)
        new_img.append(img_temp)
        new_points_set.append(points_temp)
    avg = detector.average(new_points_set)
    delaunay = spatial.Delaunay(avg)
    for points in new_points_set:
        face = warper.warp(new_img[face_index-1], points, avg, size)
        for triangle in delaunay.simplices:
            v1 = avg[triangle[0]]
            v2 = avg[triangle[1]]
            v3 = avg[triangle[2]]
            cv2.line(face, v1, v2, (0, 0, 255), 1)
            cv2.line(face, v2, v3, (0, 0, 255), 1)
            cv2.line(face, v1, v3, (0, 0, 255), 1)
   
        cv2.imwrite('warper/' + filein + '_warped_face' + str(face_index) + '.jpg', face)
        face_index = face_index + 1
if __name__ == '__main__':
    plot_triangulation('female', (600, 500))
    plot_triangulation('male', (600, 500))
    plot_triangulation('multi', (150, 100))
    test_warp('female', 'male', (600, 500))
    test_avg('multi', (150, 100))
