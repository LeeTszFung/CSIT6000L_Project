import setup
from morpher import FaceMorpher
from mtcnnModel import mtcnnModel
from VGGFaceModel import VGGFaceModel
import cv2

if __name__ == '__main__':
    size = (600, 500)
	mtcnnModel.Get_input()
	mtcnnModel.Align_celebrity()
	mtcnnModel.generate_user_faces(mtcnnModel.input_image)
	mtcnnModel.image_mapping()
	for face in mtcnnModel.Possible_face: 
        morpher = FaceMorpher(user.face_image, user.matched, size)
        result = morpher.morph(0.5)
        cv2.imwrite('final.jpg', result)  
