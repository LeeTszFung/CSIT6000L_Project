import cv2
import numpy as np

class FaceAligner:
    # resize image to designated size
    # align face in the center
    def __init__(self):
        pass

    def positive_cap(self, num):
        if num < 0:
            return 0, abs(num)
        else:
            return num, 0

    def align(self, img, points, size):
        height, width = size

        # resize image based on bounding rectangle
        rect = cv2.boundingRect(np.array([points], np.int32))
        rect_width, rect_height = rect[2:]
        height_ratio = rect_height / height
        width_ratio = rect_width / width
        if height_ratio > width_ratio:
            scale = 0.8 * height / rect_height
        else:
            scale = 0.8 * width / rect_width
        img_height, img_width = img.shape[:2]
        resized_height = int(scale * img_height)
        resized_width = int(scale * img_width)
        resized = cv2.resize(img, (resized_width, resized_height))

        # align the rectangle into the center
        x, y, w, h = rect
        mid_x = int((x + w / 2) * scale)
        mid_y = int((y + h / 2) * scale)

        delta_x = mid_x - int(width / 2)
        delta_y = mid_y - int(height / 2)

        delta_x, border_x = self.positive_cap(delta_x) 
        delta_y, border_y = self.positive_cap(delta_y)
        
        delta_h = np.min([height - border_y, resized_height - delta_y])
        delta_w = np.min([width - border_x, resized_width - delta_x])

        # crop to size
        crop = np.zeros((height, width, 3), resized.dtype)
        crop[border_y:border_y+delta_h, border_x:border_x+delta_w] = (
            resized[delta_y:delta_y+delta_h, delta_x:delta_x+delta_w]
        )

        # align face points
        points[:, 0] = (points[:, 0] * scale) + (border_x - delta_x)
        points[:, 1] = (points[:, 1] * scale) + (border_y - delta_y)

        return (crop, points)