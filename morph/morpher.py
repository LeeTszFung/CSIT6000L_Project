import cv2
from detector import FaceDetector
from aligner import FaceAligner
from warper import FaceWarper
from blender import FaceBlender

class FaceMorpher:
    def __init__(self, src, dest, size):
        self.size = size
        self.src = src
        self.dest = dest
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.warper = FaceWarper()
        self.blender = FaceBlender()

    def morph(self, percent, bg=True, blend='weighted'):
        img1 = cv2.imread(self.src)
        img2 = cv2.imread(self.dest)
        points1 = self.detector.points(img1)
        points2 = self.detector.points(img2)
        img1, points1 = self.aligner.align(img1, points1, self.size)
        img2, points2 = self.aligner.align(img2, points2, self.size)
        weighted = self.detector.weighted(points1, points2)
        face1 = self.warper.warp(img1, points1, weighted, self.size)
        face2 = self.warper.warp(img2, points2, weighted, self.size)

        if blend == 'weighted':
            blended = self.blender.weighted(face1, face2, percent)
            if bg:
                mask = self.blender.mask(weighted, blended.shape[:2])
                blended_back = self.blender.weighted(img1, img2, percent)
                overlay = self.blender.overlay(blended, blended_back, mask)
                return overlay
            else:
                return blended
