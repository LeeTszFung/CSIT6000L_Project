from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from os import walk
from userface import userface
from VGGFaceModel import VGGFaceModel


class mtcnnModel:
     def __init__(self,input_file,input_image,align_image,Possible_face,previous_name):
        self.input_file = input_file
        self.detector = MTCNN(steps_threshold=[0.0, 0.0, 0.0])
        self.align_image = align_image
        self.Possible_face = Possible_face
        self.previous_name = previous_name

     def Get_input(self): 
        Input_path = 'Input'
        input_file  = [] 
        input_filenames = next(walk(Input_path), (None, None, []))[2]
        for i in input_filenames:
            filename = Input_path + '/'+i
            input_file.append(filename)
     def landmarks(self,img):
        faces = self.detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        face = max(faces, key=lambda x: x['confidence'])  # The most clear one
        return face['keypoints']
     def affineMatrix(self,lmks, scale=2.5):
        nose = np.array(lmks['nose'], dtype=np.float32)
        left_eye = np.array(lmks['left_eye'], dtype=np.float32)
        right_eye = np.array(lmks['right_eye'], dtype=np.float32)
        eye_width = right_eye - left_eye
        angle = np.arctan2(eye_width[1], eye_width[0])
        center = nose
        alpha = np.cos(angle)
        beta = np.sin(angle)
        w = np.sqrt(np.sum(eye_width**2)) * scale
        m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
        [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]]
        return np.array(m), (int(w), int(w))
     def Align_celebrity(self):
        dataset_path = 'Image'
        Align_path = 'Align'
        celebrity_filenames = next(walk(dataset_path), (None, None, []))[2]   
        for j in celebrity_filenames:
            filename = dataset_path + '/'+j
            img = cv2.imread(filename)
            resize_img = cv2.resize(img,dsize=(500,500)) # After compress image,face may not clear to show
            mat, size =  self.affineMatrix(self.landmarks(resize_img ))
            align_img =  cv2.warpAffine(resize_img, mat, size)
            output_filename = Align_path+'/'+j
            cv2.imwrite(output_filename, align_img)
        align_filenames = next(walk(Align_path), (None, None, []))[2]   
        for i in align_filenames:
            filename = Align_path+'/'+i
            self.align_image.append(filename)
     def generate_user_faces(self,img):
        img = cv2.imread(img)
        resize_img = cv2.resize(img,dsize=(700,700))
        faces = self.detector.detect_faces(cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB))
        for face in faces:
            if face['confidence'] > 0.99:
                if(img != self.previous_name):
                       i = 0
                       self.previous_name = img
                mat,size = self.affineMatrix(face['keypoints'])
                align_img =  cv2.warpAffine(resize_img, mat, size)
                align_img =  cv2.cvtColor(align_img, cv2.COLOR_BGR2RGB)
                x = userface(img,i,"",0,align_img)
                self.Possible_face.append(x)
                i = i +1 
     def image_mapping(self):
          for user in self.Possible_face:
            result = VGGFaceModel.search_most_similar_face(user.face_image)
            user.score = result[0]
            user.matched = result[1]

