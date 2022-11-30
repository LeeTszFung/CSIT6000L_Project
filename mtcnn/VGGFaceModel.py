from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
#from keras_vggface import preprocess_input,decode_predictions
from keras_vggface import utils
from keras_vggface import vggface
from scipy.spatial.distance import cosine 
#from mtcnnModel import mtcnnModel
import matplotlib as plt

class VGGFaceModel:
    def __init__(self,align_image):
        self.detector = MTCNN(steps_threshold=[0.0, 0.0, 0.0])
        self.align_image = align_image
        
    def extract_face(self,image,resize=(224,224)):
        faces = self.detector.detect_faces(image)
        x1,y1,width,height = faces[0]['box']
        x2,y2 = x1+width,y1+height
        face_boundary = image[y1:y2,x1:x2]
        face_image = cv2.resize(face_boundary,resize)
        return face_image
    def get_embeddings(self,faces):
        face  = np.asarray(faces,'float32')
        face =  utils.preprocess_input(face,version=2)
        model = vggface.VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
        yhat = model.predict(face)
        return yhat
    def get_similarity(self,faces):
        embeddings = self.get_embeddings(faces)
        score = cosine(embeddings[0],embeddings[1])
        return score
    def search_most_similar_face(self,align_img):
        images2 =[]
        Align_path = 'Align'
        user_face = self.extract_face(align_img)
        min_score = 1 
        min_filename = ''
        for i in self.align_image:
            images2.append(user_face)
            imagename = Align_path + '/'+i
            compared_img = plt.imread(imagename)
            temp_face = self.extract_face(compared_img)
            images2.append(temp_face)
            score = self.get_similarity(images2)
            if score < min_score:
               min_score = score
               min_filename = i 
        images2.clear()
      

        return [min_score,min_filename]
    
 