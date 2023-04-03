import unittest

import numpy as np
import os
import matplotlib.pyplot as plt

from surface_reconstruction import frankot_chellappa

def gen_surface_grad(Nx=512, xb=10, Ny=256, yb=10, fx=1., fy=2.):
    x = np.linspace(-xb,xb,Nx)
    dx = abs(x[1] - x[0])

    y = np.linspace(-yb,yb,Ny)
    dy = abs(y[1] - y[0])

    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = 0.5*np.cos(2*np.pi*(fx*xx + fy*yy))
    z = z - z.min()

    gx = (z[1:,1:] - z[0:-1,1:]) / dx
    gy = (z[1:,1:] - z[1:,0:-1]) / dy

    return z, gx, gy, dx, dy

def imshow_surfs(gnd, rec, save_name):
    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].imshow(gnd, cmap="gray", vmin=gnd.min(), vmax=gnd.max())
    axs[0].set_title("Ground Truth")
    im = axs[1].imshow(rec, cmap="gray", vmin=gnd.min(), vmax=gnd.max())
    axs[1].set_title("Reconstruction")
    fig.colorbar(im, ax=axs[1])
    plt.savefig(os.path.join("figures",save_name+"_2D.png"))
    plt.close()

    rows, cols = gnd.shape
    N = cols // 2
    plt.figure()
    plt.plot(gnd[:,N], label="GND")
    plt.plot(rec[:,N-1], label="REC")
    plt.legend()
    plt.savefig(os.path.join("figures", save_name+"_1D.png"))
    plt.close()
class TestSurfaceReconstruction(unittest.TestCase):

    def setUp(self):
        # setup a gradient
        self.z, self.gx, self.gy, self.dx, self.dy = gen_surface_grad()

        # Need to visually check surface reconstruction quality
        if not os.path.exists("figures"):
            os.mkdir("figures")

    def test_frankot_chellappa(self):
        # reconstruct surface
        s = frankot_chellappa(self.gx, self.gy, self.dx, self.dy)

        # save plots
        imshow_surfs(self.z, s, "frankot_chellappa")
        



if __name__ == "__main__":
    unittest.main()