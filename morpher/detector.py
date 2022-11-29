import cv2
import dlib
import numpy as np
import os.path as path

DAT_DIR = path.join(path.dirname(path.realpath(__file__)), 'dat')
DAT_PATH = path.join(DAT_DIR, 'shape_predictor_68_face_landmarks.dat')

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(DAT_PATH)

    def points(self, img):
        # Get an array of fact points in the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # upsample the image 1 time.
        faces = self.detector(img, 1)

        if len(faces) == 0:
            raise RuntimeError('No face detected')
        
        points_set = []
        for face in faces:
            shapes = self.predictor(img, face)
            points = np.array([(shapes.part(i).x, shapes.part(i).y) for i in range(68)], np.int32)
            points = np.vstack([
                points,
                self.boundary(points, 0.1, -0.03),
                self.boundary(points, 0.13, -0.05),
                self.boundary(points, 0.15, -0.08),
                self.boundary(points, 0.33, -0.12)
            ])
            points_set.append(points)
        return points_set
    
    def boundary(self, points, width, height):
        # add two boundary points at top left and top right corners
        x, y, w, h = cv2.boundingRect(np.array([points], np.int32))
        delta_w = int(w * width)
        delta_h = int(h * height)
        return [[x + delta_w, y + delta_h], [x + w - delta_w, y + delta_h]]
    
    def weighted(self, src_points, dest_points, percent=0.5):
        # weighted average between 2 sets of face points
        if percent <= 0:
            return dest_points
        elif percent >= 1:
            return src_points
        else:
            return np.asarray(src_points*percent + dest_points*(1-percent), np.int32)
    
    def average(self, points_set):
        # average face points
        return np.mean(points_set, 0).astype(np.int32)