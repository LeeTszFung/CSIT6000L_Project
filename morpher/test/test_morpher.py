import setup
from morpher import FaceMorpher
import cv2
import read_parameter

if __name__ == '__main__':
    parent_path = "C:\\Users\\user\\Downloads\\Test\\CSIT6000L_Project-main\\"

    my_percent = read_parameter.read_param("percent")
    my_dst_img = read_parameter.read_param("my_dst_img")
    print(my_dst_img)

    print("'test_morpher.py' started.", flush=True, end='')
    size = (600, 500)
    morpher = FaceMorpher(parent_path+'morpher\\test\\female.jpg', parent_path+'morpher\\test\\dst\\' + str(my_dst_img), size)
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        weighted = morpher.morph(percent, blend='weighted')
        cv2.imwrite(parent_path+'morpher/test/morpher/weighted_' + str(percent) + '.jpg', weighted[0])
        alpha = morpher.morph(percent, blend='alpha')
        cv2.imwrite(parent_path+'morpher/test/morpher/alpha_' + str(percent) + '.jpg', alpha[0])
    overlay = morpher.morph(my_percent, bg='overlay')
    # cv2.imwrite('morpher/overlay_0.5.jpg', overlay[0]) 
    cv2.imwrite(parent_path+'morpher/test/morpher_output/overlay_' + str(my_dst_img.replace(".jpg", "")) + "_" +  str(my_percent) + '.jpg', overlay[0])
    poisson = morpher.morph(my_percent, bg='poisson')
    # cv2.imwrite('morpher/poisson_0.5.jpg', poisson[0])
    cv2.imwrite(parent_path+'morpher/test/morpher_output/poisson_' + str(my_dst_img.replace(".jpg", "")) + "_" + str(my_percent) + '.jpg', poisson[0])
    alpha = morpher.morph(my_percent, bg='alpha')
    print("Writing 'morpher/alphabg_0.5.jpg' ...", flush=True, end='')
    # cv2.imwrite('morpher/alphabg_0.5.jpg', alpha[0])
    cv2.imwrite(parent_path+'morpher/test/morpher_output/alphabg_' + str(my_dst_img.replace(".jpg", "")) + "_" + str(my_percent) + '.jpg', alpha[0])
    print("Writing done.", flush=True, end='')