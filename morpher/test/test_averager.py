import setup
from averager import FaceAverager
import cv2

image_list = ['multi.jpg']
averager = FaceAverager(image_list, (150, 100))
cv2.imwrite('averager/trans.jpg', averager.average())
cv2.imwrite('averager/overlay.jpg', averager.average(bg='overlay'))