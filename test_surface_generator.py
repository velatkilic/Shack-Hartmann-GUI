import unittest
import os
import scipy.io as sio
from surface_generator import SurfaceGenerator

class TestSurfaceGenerator(unittest.TestCase):
    def setUp(self):
        super().__init__()
        self.surfaces = SurfaceGenerator()
    
    def test_surface_gen_iterator(self):
        # Test iterator is working
        mdict = {}
        for name, surf, gradx, grady in self.surfaces:
            mdict[name+"_gradx"] = gradx
            mdict[name+"_grady"] = grady
            mdict[name+"_surf"]  = surf
        
        self.assertGreater(len(mdict), 0)

        # save to file
        sio.savemat(os.path.join(os.getcwd(), "figures", "surf.mat"), mdict)



if __name__ == "__main__":
    unittest.main()