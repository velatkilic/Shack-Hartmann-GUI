import unittest
import numpy as np

from utils import *

class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_centroid1D_wrong_array_shape(self):
        arr = np.zeros((10,10))
        cen = centroid1D(arr)
        self.assertIsNone(cen)

    def test_centroid1D_small_weight(self):
        arr = 1e-5*np.ones((10,))
        cen = centroid1D(arr, th_tot_weight=1e-1)
        self.assertEqual(cen, 0.)

    def test_centroid1D_index(self):
        arr = np.array([0.,1.,0.])
        cen = centroid1D(arr)
        self.assertEqual(cen, 1.)

    def test_centroid1D_index_sym(self):
        arr = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        cen = centroid1D(arr)
        self.assertEqual(cen, 2.)

    def test_centroid1D_scalar_index(self):
        arr = np.array([1.,])
        cen = centroid1D(arr)
        self.assertEqual(cen, 0.)

    def test_centroid2D_wrong_shape(self):
        arr = np.ones((1,2,2))
        row, col = centroid2D(arr)
        self.assertIsNone(row)
        self.assertIsNone(col)

    def test_centroid2D_simple(self):
        arr = np.array([[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]])
        row, col = centroid2D(arr)
        self.assertEqual(row, 1.)
        self.assertEqual(col, 1.)

    def test_centroid2D_sym(self):
        arr = np.array([[0., 0.5, 0.],
                        [0.5, 1., 0.5],
                        [0., 0.5, 0.]])
        row, col = centroid2D(arr)
        self.assertEqual(row, 1.)
        self.assertEqual(col, 1.)
    
    def test_centroid2D_single_row(self):
        arr = np.array([[0., 0.5, 0.],])
        row, col = centroid2D(arr)
        self.assertEqual(row, 0.)
        self.assertEqual(col, 1.)
    
    def test_centroid2D_single_col(self):
        arr = np.array([[0.0,],
                        [0.5,],
                        [0.0,]])
        row, col = centroid2D(arr)
        self.assertEqual(row, 1.)
        self.assertEqual(col, 0.)
    
    def test_centroid2D_negative(self):
        arr = -np.array([[0., 0.5, 0.],
                        [0.5, 1., 0.5],
                        [0., 0.5, 0.]])
        row, col = centroid2D(arr)
        self.assertIsNone(row)
        self.assertIsNone(col)

    def test_center_to_bbox_middle(self):
        s = 1.
        row, col = 10., 10.
        img_shape = 100, 100
        r0,c0,r1,c1 = center_to_bbox(row, col, s, img_shape)
        self.assertEqual([r0,c0,r1,c1], [9.,9.,12.,12.])

    def test_center_to_bbox_edge_max(self):
        s = 1.
        row, col = 10., 10.
        img_shape = 10, 10
        r0,c0,r1,c1 = center_to_bbox(row, col, s, img_shape)
        self.assertEqual([r0,c0,r1,c1], [9.,9.,11.,11.])
    
    def test_center_to_bbox_edge_min(self):
        s = 1.
        row, col = 0., 0.
        img_shape = 10, 10
        r0,c0,r1,c1 = center_to_bbox(row, col, s, img_shape)
        self.assertEqual([r0,c0,r1,c1], [0.,0.,2.,2.])

    def test_blobs_to_centroid_sym_exact_blob_estim(self):
        N = 512
        x = np.linspace(-10.,10.,N)
        y = np.linspace(-10.,10.,N)
        xx, yy = np.meshgrid(x, y)
        s = 10.
        gaus = np.exp(-0.5*(xx**2 + yy**2)/s**2)
        row, col = N//2, N//2
        cen = blobs_to_centroid(gaus, [(row, col, s)])
        self.assertAlmostEqual(cen, [[row, col]])
    
    def test_blobs_to_centroid_sym_good_blob_estim(self):
        N = 512
        x = np.linspace(-10.,10.,N)
        y = np.linspace(-10.,10.,N)
        xx, yy = np.meshgrid(x, y)
        s = 100.
        gaus = np.exp(-0.5*(xx**2 + yy**2)/s**2)
        row, col = N//2-1, N//2-1
        cen = blobs_to_centroid(gaus, [(row, col, s)])
        self.assertAlmostEqual(cen, [[row, col]])

    def test_bbox_to_centroid(self):
        pass

if __name__ == "__main__":
    unittest.main()