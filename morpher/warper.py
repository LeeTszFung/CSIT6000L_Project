import cv2
import numpy as np
import scipy.spatial as spatial

class FaceWarper:
    def __init__(self):
        pass

    @staticmethod
    def affine(triangles, src_points, dest_points):
        ones = [1, 1, 1]
        for vertices in triangles:
            src_tri = np.vstack((src_points[vertices, :].T, ones))
            dst_tri = np.vstack((dest_points[vertices, :].T, ones))
            mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
            yield mat

    @staticmethod
    def grid(points):
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0]) + 1
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1]) + 1
        return np.asarray([(x, y) for y in range(ymin, ymax)
                     for x in range(xmin, xmax)], np.uint32)

    @staticmethod
    def bilinear_interpolate(img, coords):
        int_coords = np.int32(coords)
        x0, y0 = int_coords
        dx, dy = coords - int_coords

        # 4 Neighour pixels
        q11 = img[y0, x0]
        q21 = img[y0, x0+1]
        q12 = img[y0+1, x0]
        q22 = img[y0+1, x0+1]

        btm = q21.T * dx + q11.T * (1 - dx)
        top = q22.T * dx + q12.T * (1 - dx)
        inter_pixel = top * dy + btm * (1 - dy)

        return inter_pixel.T

    def warp(self, src_img, src_points, dest_points, size, dtype=np.uint8):
        src_img = src_img[:, :, :3]
        rows, cols = size[:2]
        dest_img = np.zeros((rows, cols, 3), dtype)
         
        delaunay = spatial.Delaunay(dest_points)

        affines = np.asarray(list(self.affine(delaunay.simplices, src_points, dest_points)))

        grid = self.grid(dest_points)
        indices = delaunay.find_simplex(grid)

        for simplex_index in range(len(delaunay.simplices)):
            coords = grid[indices == simplex_index]
            num_coords = len(coords)
            out_coords = np.dot(affines[simplex_index],
                        np.vstack((coords.T, np.ones(num_coords))))
            x, y = coords.T
            dest_img[y, x] = self.bilinear_interpolate(src_img, out_coords)

        return dest_img