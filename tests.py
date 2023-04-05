import unittest

import numpy as np
import os
import matplotlib.pyplot as plt

from surface_reconstruction import (frankot_chellappa,
                                    poisson_solver_neumann, 
                                    der_y, der_x, 
                                    harker_oleary,
                                    harker_oleary_dirichlet,
                                    harker_oleary_spectral)

def gen_surface_grad(Nx=512, xb=10, Ny=256, yb=10, fx=1., fy=2.):
    x = np.linspace(-xb,xb,Nx)
    dx = abs(x[1] - x[0])

    y = np.linspace(-yb,yb,Ny)
    dy = abs(y[1] - y[0])

    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = 0.5*np.cos(2*np.pi*(fx*xx + fy*yy))
    z = z - z.mean()

    gx = der_x(z, dx)
    gy = der_y(z, dy) 

    return z, gx, gy, dx, dy

def imshow_surfs(gnd, rec, save_name):
    vmin = np.minimum(gnd.min(), rec.min())
    vmax = np.maximum(gnd.max(), rec.max())

    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].imshow(gnd, cmap="gray", vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth")
    im = axs[1].imshow(rec, cmap="gray", vmin=vmin, vmax=vmax)
    axs[1].set_title("Reconstruction")
    fig.colorbar(im, ax=axs[1])
    plt.savefig(os.path.join("figures",save_name+"_2D.png"))
    plt.close()

    rows, cols = gnd.shape
    N = cols // 2
    plt.figure()
    plt.plot(gnd[:,N], label="GND")
    plt.plot(rec[:,N], label="REC")
    plt.legend()
    plt.savefig(os.path.join("figures", save_name+"_1D.png"))
    plt.close()
class TestSurfaceReconstruction(unittest.TestCase):

    def setUp(self):
        # setup a gradient
        self.z, self.gx, self.gy, self.dx, self.dy = gen_surface_grad(Nx=1024, Ny=512)

        # Need to visually check surface reconstruction quality
        if not os.path.exists("figures"):
            os.mkdir("figures")

    def test_frankot_chellappa(self):
        # reconstruct surface
        s = frankot_chellappa(self.gx, self.gy, self.dx, self.dy)

        # save plots
        imshow_surfs(self.z, s, "frankot_chellappa")
    
    def test_poisson_solver_neumann(self):
        # reconstruct surface
        s = poisson_solver_neumann(self.gx, self.gy, self.dx, self.dy)

        # save plots
        imshow_surfs(self.z, s, "poisson_solver_neumann")

    def test_harker_oleary(self):
        # reconstruct surface
        s = harker_oleary(self.gx, self.gy, self.dx, self.dy)

        # save plots
        imshow_surfs(self.z, s, "Harker Oleary")

    def test_harker_oleary_dirichlet(self):
        # reconstruct surface
        s = harker_oleary_dirichlet(self.gx, self.gy, self.dx, self.dy)

        # save plots
        imshow_surfs(self.z, s, "Harker Oleary - Dirichlet")

    def test_harker_oleary_spectral(self):
        # reconstruct surface
        s = harker_oleary_spectral(self.gx, self.gy, self.dx, self.dy, mask=[150, 150])

        # save plots
        imshow_surfs(self.z, s, "Harker Oleary - Spectral")

if __name__ == "__main__":
    unittest.main()