import setup
from morpher import FaceMorpher
import cv2
import read_parameter
import os
import shutil
from os import walk
import sys
import time

if __name__ == '__main__':
    start_time = time.time()
    parent_path = "./"
    from_ui_parent_path = "../morpher/test/"
    if "call from UI" in sys.argv:
        root_path = from_ui_parent_path
    else:
        root_path = parent_path
    
    my_percent = read_parameter.read_param("percent")
    my_dst_img = read_parameter.read_param("my_dst_img")
    # print(my_dst_img)

    # print("'test_morpher.py' started.", flush=True, end='')
    size = (600, 500)

    # Copy source imput image
    # print(os.getcwd())
    
    mtcnn_input_filenames = next(walk(root_path + "../../mtcnn/Input"), (None, None, []))[2]
    # print(mtcnn_input_filenames)
    shutil.copy(root_path + '../../mtcnn/Input/' + mtcnn_input_filenames[0], root_path + 'morpher_input/')
    morpher = FaceMorpher(root_path + 'dst/' + str(my_dst_img), root_path + 'morpher_input/'+mtcnn_input_filenames[0], size)

    # for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     weighted = morpher.morph(percent, blend='weighted')
    #     cv2.imwrite(parent_path+'morpher/test/morpher/weighted_' + str(percent) + '.jpg', weighted[0])
    #     alpha = morpher.morph(percent, blend='alpha')
    #     cv2.imwrite(parent_path+'morpher/test/morpher/alpha_' + str(percent) + '.jpg', alpha[0])
    
    # Generate all percent
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("Generating percent: " + str(percent))
        overlay = morpher.morph(percent, bg='overlay')
        # cv2.imwrite('morpher/overlay_0.5.jpg', overlay[0]) 
        cv2.imwrite(root_path+'morpher_output/overlay_' + str(my_dst_img.replace(".jpg", "")) + "_" +  str(percent) + '.jpg', overlay[0])
        poisson = morpher.morph(percent, bg='poisson')
        # cv2.imwrite('morpher/poisson_0.5.jpg', poisson[0])
        cv2.imwrite(root_path+'morpher_output/poisson_' + str(my_dst_img.replace(".jpg", "")) + "_" + str(percent) + '.jpg', poisson[0])
        alpha = morpher.morph(percent, bg='alpha')
        # cv2.imwrite('morpher/alphabg_0.5.jpg', alpha[0])
        cv2.imwrite(root_path+'morpher_output/alphabg_' + str(my_dst_img.replace(".jpg", "")) + "_" + str(percent) + '.jpg', alpha[0])
    
    # overlay = morpher.morph(my_percent, bg='overlay')
    # # cv2.imwrite('morpher/overlay_0.5.jpg', overlay[0]) 
    # cv2.imwrite(parent_path+'morpher/test/morpher_output/overlay_' + str(my_dst_img.replace(".jpg", "")) + "_" +  str(my_percent) + '.jpg', overlay[0])
    # poisson = morpher.morph(my_percent, bg='poisson')
    # # cv2.imwrite('morpher/poisson_0.5.jpg', poisson[0])
    # cv2.imwrite(parent_path+'morpher/test/morpher_output/poisson_' + str(my_dst_img.replace(".jpg", "")) + "_" + str(my_percent) + '.jpg', poisson[0])
    # alpha = morpher.morph(my_percent, bg='alpha')
    # # cv2.imwrite('morpher/alphabg_0.5.jpg', alpha[0])
    # cv2.imwrite(parent_path+'morpher/test/morpher_output/alphabg_' + str(my_dst_img.replace(".jpg", "")) + "_" + str(my_percent) + '.jpg', alpha[0])
    
    print("Writing done.", flush=True, end='')
    print("--- %s seconds ---" % (time.time() - start_time))