import unittest

import numpy as np
import os
import matplotlib.pyplot as plt

from surface_reconstruction import frankot_chellappa


class TestSurfaceReconstruction(unittest.TestCase):

    def setUp(self):
        Nx = 512
        x = np.linspace(-10,10,Nx)
        self.dx = abs(x[1] - x[0])

        Ny = 256
        y = np.linspace(-10,10,Ny)
        self.dy = abs(y[1] - y[0])

        xx, yy = np.meshgrid(x, y, indexing="ij")
        z = 0.5*np.cos(2*np.pi*(xx + 2*yy))
        self.z = z - z.min()

        self.gx = (z[1:,1:] - z[0:-1,1:]) / self.dx
        self.gy = (z[1:,1:] - z[1:,0:-1]) / self.dy

        # Need to visually check surface reconstruction quality
        if not os.path.exists("figures"):
            os.mkdir("figures")
        
        plt.figure()
        plt.imshow(self.z, cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join("figures","gnd_surface.png"))
        plt.close()

    def test_frankot_chellappa(self):
        # reconstruct surface
        s = frankot_chellappa(self.gx, self.gy, self.dx, self.dy)

        plt.figure()
        plt.imshow(s, cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join("figures","rec_surface_frankot_chellappa.png"))
        plt.close()

        plt.figure()
        plt.plot(self.z[:,100], label="GND")
        plt.plot(s[:,99], label="REC")
        plt.legend()
        plt.savefig(os.path.join("figures", "rec_vs_gnd_1d_frankot_chellappa.png"))
        plt.close()



if __name__ == "__main__":
    unittest.main()