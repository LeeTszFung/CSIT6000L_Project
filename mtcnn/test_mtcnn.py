from mtcnnModel import mtcnnModel
import os
import shutil
from PIL import Image
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
	size = (600, 500)
	model = mtcnnModel()
	model.Get_input()
	#model.Align_celebrity()
	
	for input_image in model.input_file:
		model.generate_user_faces(input_image)
		model.image_mapping()
	for user in model.Possible_face: 
		Input_path = 'Input'
		print(user.name)
		input_filename = Input_path+'/'+user.name
		Image_path = 'Image'
		Celebrity_filename = Image_path+'/'+user.matched
		print(Celebrity_filename)
		# Save to most similar path
		src_path = Celebrity_filename
		dst_path = "../morpher/test/dst/"
		# print(user.face_image)
# 		im = Image.fromarray(user.face_image)
# 		im.save(dst_path+"dst.jpg")
		# shutil.copy(src_path, dst_path)