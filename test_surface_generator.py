import unittest
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from surface_generator import SurfaceGenerator, calc_crack_grad_n1

def make_figure(grad_data, fig_name):
    plt.figure()
    plt.imshow(grad_data, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join("figures", fig_name+".png"))
    plt.close()

class TestSurfaceGenerator(unittest.TestCase):
    def setUp(self):
        self.surfaces = SurfaceGenerator()

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

    def test_calc_crack_grad_n1(self):
        N = 512
        ub = 30e-3
        x = np.linspace(ub, -ub, N)
        y = np.linspace(1.5*ub, -0.5*ub, N)
        yy, xx = np.meshgrid(x, y)
        gradx, grady, filt_final = calc_crack_grad_n1(xx, yy, # m
                                                      k_1 = 0.9, # MPa * m^-1/2
                                                      h_sample = 8.6e-3, # m
                                                      poissons_ratio = 0.34,
                                                      youngs_modulus = 3.3e3, # MPa
                                                      )
        gradx[filt_final] = 0.
        grady[filt_final] = 0.
        make_figure(gradx, "crack_grad_x_paper_coord")
        make_figure(grady, "crack_grad_y_paper_coord")
        make_figure(filt_final, "filt_final_paper_coord")

if __name__ == "__main__":
    unittest.main()