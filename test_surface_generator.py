import unittest
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from surface_generator import SurfaceGenerator, CrackGradientGenerator

def make_figure(grad_data, fig_name):
    plt.figure()
    plt.imshow(grad_data, cmap="jet")
    plt.colorbar()
    plt.savefig(os.path.join("figures", fig_name+".png"))
    plt.close()

class TestSurfaceGenerator(unittest.TestCase):
    def setUp(self):
        self.surfaces = SurfaceGenerator()
        self.crack = CrackGradientGenerator()

        fig_save_dir = os.path.join(os.getcwd(), "figures")
        if not(os.path.exists(fig_save_dir)):
            os.mkdir(fig_save_dir)
    
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

    def test_crack_grad_gen(self):
        gradx, grady = self.crack.calc_gradient()
        self.assertIsNotNone(gradx)
        self.assertIsNotNone(grady)
        make_figure(gradx, "crack_grad_x")
        make_figure(grady, "crack_grad_y")

if __name__ == "__main__":
    unittest.main()