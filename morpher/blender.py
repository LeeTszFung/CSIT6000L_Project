import cv2
import numpy as np
import scipy

class FaceBlender:
    def __init__(self):
        pass

    def weighted(self, face1, face2, percent=0.5):
        if percent <= 0:
            return face2
        elif percent >= 1:
            return face1
        else:
            return cv2.addWeighted(face1, percent, face2, 1-percent, 0)
    
    def mask(self, points, size):
        radius = 10  # kernel size
        kernel = np.ones((radius, radius), np.uint8)

        mask = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
        mask = cv2.erode(mask, kernel)

        return mask
    
    def overlay(self, face, back, mask):
        pixels = mask > 0
        back[...,:3][pixels] = face[...,:3][pixels]
        return back
    
    def alpha_feathering(self, src_img, dest_img, img_mask, blur_radius=500):
        mask = cv2.blur(img_mask, (blur_radius, blur_radius))
        mask = mask / 255.0

        result_img = np.empty(src_img.shape, np.uint8)
        for i in range(3):
            result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

        return result_img

    def poisson(self, img_source, dest_img, img_mask, offset=(0, 0)):
        img_target = np.copy(dest_img)
        import pyamg
        
        region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0] - offset[0], img_source.shape[0]),
            min(img_target.shape[1] - offset[1], img_source.shape[1]))
        region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0] + offset[0]),
            min(img_target.shape[1], img_source.shape[1] + offset[1]))
        region_size = (region_source[2] - region_source[0],
                 region_source[3] - region_source[1])

        # clip and normalize mask image
        img_mask = img_mask[region_source[0]:region_source[2],
                      region_source[1]:region_source[3]]

        # create coefficient matrix
        coff_mat = scipy.sparse.identity(np.prod(region_size), format='lil')
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if img_mask[y, x]:
                    index = x + y * region_size[1]
                    coff_mat[index, index] = 4
                    if index + 1 < np.prod(region_size):
                        coff_mat[index, index + 1] = -1
                    if index - 1 >= 0:
                        coff_mat[index, index - 1] = -1
                    if index + region_size[1] < np.prod(region_size):
                        coff_mat[index, index + region_size[1]] = -1
                    if index - region_size[1] >= 0:
                        coff_mat[index, index - region_size[1]] = -1
        coff_mat = coff_mat.tocsr()

        # create poisson matrix for b
        poisson_mat = pyamg.gallery.poisson(img_mask.shape)
        # for each layer (ex. RGB)
        for num_layer in range(img_target.shape[2]):
            # get subimages
            t = img_target[region_target[0]:region_target[2],
                    region_target[1]:region_target[3], num_layer]
            s = img_source[region_source[0]:region_source[2],
                    region_source[1]:region_source[3], num_layer]
            t = t.flatten()
            s = s.flatten()

            # create b
            b = poisson_mat * s
            for y in range(region_size[0]):
                for x in range(region_size[1]):
                    if not img_mask[y, x]:
                        index = x + y * region_size[1]
                        b[index] = t[index]

            # solve Ax = b
            x = pyamg.solve(coff_mat, b, verb=False, tol=1e-10)

            # assign x to target image
            x = np.reshape(x, region_size)
            x[x > 255] = 255
            x[x < 0] = 0
            x = np.array(x, img_target.dtype)
            img_target[region_target[0]:region_target[2],
               region_target[1]:region_target[3], num_layer] = x

        return img_target