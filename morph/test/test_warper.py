import setup
from detector import FaceDetector
from aligner import FaceAligner
from warper import FaceWarper
import cv2
import numpy as np
import scipy.spatial as spatial

def plot_triangulation(filein, fileout, size = (600, 500)):
    img = cv2.imread(filein)
    detector = FaceDetector()
    points = detector.points(img)
    aligner = FaceAligner()
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
    cv2.imwrite(fileout, aligned)

def test_warp(filein1, filein2, fileout1, fileout2):
    size = (600, 500)
    img1 = cv2.imread(filein1)
    img2 = cv2.imread(filein2)
    detector = FaceDetector()
    aligner = FaceAligner()
    warper = FaceWarper()

    points1 = detector.points(img1)
    points2 = detector.points(img2)
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
   
    cv2.imwrite(fileout1, face1)
    cv2.imwrite(fileout2, face2)
    

if __name__ == '__main__':
    plot_triangulation('face.jpg', 'face_triangulated.jpg')
    test_warp('face.jpg', 'face2.jpg', 'face_warped.jpg', 'face2_warped.jpg')
