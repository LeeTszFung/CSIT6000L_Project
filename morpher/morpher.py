import cv2
import numpy as np
from detector import FaceDetector
from aligner import FaceAligner
from warper import FaceWarper
from blender import FaceBlender

class FaceMorpher:
    def __init__(self, src, dest, size):
        self.size = size
        self.src = cv2.imread(src)
        self.dest = cv2.imread(dest)
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.warper = FaceWarper()
        self.blender = FaceBlender()

    def morph(self, percent, bg=None, blend='weighted'):
        img1 = self.src
        img2 = self.dest
        points_set1 = self.detector.points(img1)
        points_set2 = self.detector.points(img2)
        if len(points_set2) != 1:
            raise RuntimeError("Dest image can only have one face")
        points2 = points_set2[0]
        img2, points2 = self.aligner.align(img2, points2, self.size)
        results = []
        for points1 in points_set1:
            img1, points1 = self.aligner.align(img1, points1, self.size)
            weighted = self.detector.weighted(points1, points2)
            face1 = self.warper.warp(img1, points1, weighted, self.size)
            face2 = self.warper.warp(img2, points2, weighted, self.size)
            mask = self.blender.mask(weighted, self.size)

            if blend == 'weighted':
                blended = self.blender.weighted(face1, face2, percent)
            elif blend == 'alpha':
                s = max(self.size)
                if percent <= 0.5:
                    # blur_radius should be greater than 0
                    blur_radius = int(2 * s * percent) + 10
                    blended = self.blender.alpha_feathering(face2, face1, mask, blur_radius)
                else:
                    percent = 1 - percent 
                    blur_radius = int(2 * s * percent) + 10
                    blended = self.blender.alpha_feathering(face1, face2, mask, blur_radius)
            else:
                raise RuntimeError("Invalid blend method")


            trans = np.dstack((blended, mask))
            if bg is None:
                results.append(trans)
            elif bg == 'overlay':
                back = self.blender.weighted(img1, img2, percent)
                overlay = self.blender.overlay(trans, back, mask)
                results.append(overlay)
            elif bg == 'poisson':
                poisson = self.blender.poisson(trans, img1, mask)
                results.append(poisson)
            else:
                raise RuntimeError("Invalid Background")
        return results
