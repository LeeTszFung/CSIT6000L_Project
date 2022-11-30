import setup
from morpher import FaceMorpher
from mtcnnModel import mtcnnModel
from VGGFaceModel import VGGFaceModel
import cv2

if __name__ == '__main__':
	size = (600, 500)
	mtcnnModel.Get_input()
	mtcnnModel.Align_celebrity()
	for input_image in mtcnnModel.input_file:
		mtcnnModel.generate_user_faces(mtcnnModel.input_image)
		mtcnnModel.image_mapping()
	for user in mtcnnModel.Possible_face: 
		Input_path = 'Input'
		input_filename = Input_path+'/'+user.name
		Image_path = 'Image'
		Celebrity_filename = Image_path+'/'+user.mateched
		morpher = FaceMorpher(input_filename,Celebrity_filename, size)
		for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
			weighted = morpher.morph(percent, blend='weighted')
			cv2.imwrite('morpher/weighted_' + str(percent) + '.jpg', weighted[user.id])
			alpha = morpher.morph(percent, blend='alpha')
			cv2.imwrite('morpher/alpha_' + str(percent) + '.jpg', alpha[user.id])
			overlay = morpher.morph(0.5, bg='overlay')
			cv2.imwrite('morpher/overlay_0.5.jpg', overlay[user.id])
			poisson = morpher.morph(0.5, bg='poisson')
			cv2.imwrite('morpher/poisson_0.5.jpg', poisson[user.id])