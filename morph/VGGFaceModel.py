from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input,decode_predictions
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine 

class VGGFaceModel:
    def __init__(self):
	             pass

    def extract_face(self,image,resize=(224,224)):
        faces = detector.detect_faces(image)
        x1,y1,width,height = faces[0]['box']
        x2,y2 = x1+width,y1+height
        face_boundary = image[y1:y2,x1:x2]
        face_image = cv2.resize(face_boundary,resize)
        return face_image
    
	def get_embeddings(self,faces):
        face  = np.asarray(faces,'float32')
        face =  preprocess_input(face,version=2)
        model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
        yhat = model.predict(face)
        return yhat
	
	def get_similarity(self,faces):
        embeddings = get_embeddings(faces)
        score = cosine(embeddings[0],embeddings[1])
        return score
    
	
	def search_most_similar_face(self,aline_img):
        images2 =[]
        user_face = extract_face(aline_img)
        min_score = 1 
        min_filename = ''
        for i in aline_filenames:
            images2.append(user_face)
            imagename = Aline_path + '/'+i
            compared_img = plt.imread(imagename)
            temp_face = extract_face(compared_img)
            images2.append(temp_face)
            score = get_similarity(images2)
            if score < min_score:
               min_score = score
               min_filename = i 
        images2.clear()
      

        return [min_score,min_filename]
    
 