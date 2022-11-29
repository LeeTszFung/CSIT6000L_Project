import cv2
import numpy as np
from detector import FaceDetector
from aligner import FaceAligner
from warper import FaceWarper
from blender import FaceBlender

class FaceAverager:
    def __init__(self, image_list, size):
        self.size = size
        self.imgs = image_list
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.warper = FaceWarper()
        self.blender = FaceBlender()

    def average(self, bg=None):
        new_points_set = []
        new_img = []
        for path in self.imgs:
            img = cv2.imread(path) 
            points_set = self.detector.points(img)
            for points in points_set:
                img_temp, points_temp = self.aligner.align(img, points, self.size)
                new_img.append(img_temp)
                new_points_set.append(points_temp)
        if len(new_img) == 0:
            raise RuntimeError("No face found")
        avg_points = self.detector.average(new_points_set)

        count = len(new_img)
        results = np.zeros(new_img[0].shape, np.float32)
        for i in range(count):
            results += self.warper.warp(new_img[i], new_points_set[i], avg_points, self.size, dtype=np.float32)
        result = np.uint8(results / count)
        mask = self.blender.mask(avg_points, self.size)
        trans = np.dstack((result, mask))
        if bg is None:
            return trans
        elif bg == 'overlay':
            avg_bg = self.detector.average(new_img)
            return self.blender.overlay(trans, avg_bg, mask)
        else:
            raise RuntimeError("Invalid bg")
