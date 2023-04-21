import unittest
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

from surface_reconstruction import (frankot_chellappa,
                                    poisson_solver_neumann,
                                    harker_oleary,
                                    harker_oleary_dirichlet,
                                    harker_oleary_spectral,
                                    harker_oleary_tikhonov,
                                    harker_oleary_weighted)
from surface_generator import SurfaceGenerator

def imshow_surfs(gnd, rec, algo_name, save_name):
    save_dir = os.path.join(os.getcwd(), "figures", algo_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    gnd -= gnd.mean()
    rec -= rec.mean()

    vmin = np.minimum(gnd.min(), rec.min())
    vmax = np.maximum(gnd.max(), rec.max())

    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].imshow(gnd, cmap="gray", vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth")
    im = axs[1].imshow(rec, cmap="gray", vmin=vmin, vmax=vmax)
    axs[1].set_title("Reconstruction")
    fig.colorbar(im, ax=axs[1])
    plt.savefig(os.path.join(save_dir,save_name+"_2D.png"))
    plt.close()

    rows, cols = gnd.shape
    slices = [0.3, 0.5]
    fig, axs = plt.subplots(2,2,figsize=(12, 12), constrained_layout=True)

    for i in range(2):
        N = int(cols * slices[i])
        axs[0,i].plot(gnd[:,N], label="GND")
        axs[0,i].plot(rec[:,N], label="REC")
        if i == 0: axs[0,i].legend()

        N = int(rows * slices[i])
        axs[1,i].plot(gnd[N,:], label="GND")
        axs[1,i].plot(rec[N,:], label="REC")

    plt.savefig(os.path.join(save_dir, save_name+"_1D.png"))
    plt.close()

class TestSurfaceReconstruction(unittest.TestCase):
    def setUp(self):
        # test surface generator class
        self.surfaces = SurfaceGenerator()
        self.dx, self.dy = self.surfaces.get_dx_dy()

        # Need to visually check surface reconstruction quality
        if not os.path.exists("figures"):
            os.mkdir("figures")

    def test_frankot_chellappa(self):
        for name, surf, gradx, grady in self.surfaces:
            # reconstruct surface
            rec = frankot_chellappa(gradx, grady, self.dx, self.dy)
            # save plots
            imshow_surfs(surf, rec, "frankot_chellappa", "frankot_chellappa_"+name)
    
    def test_poisson_solver_neumann(self):
        for name, surf, gradx, grady in self.surfaces:
            # reconstruct surface
            rec = poisson_solver_neumann(gradx, grady, self.dx, self.dy)
            # save plots
            imshow_surfs(surf, rec, "poisson_solver_neumann", "poisson_solver_neumann"+name)

    def test_harker_oleary(self):
        for name, surf, gradx, grady in self.surfaces:
            # reconstruct surface
            rec = harker_oleary(gradx, grady, self.dx, self.dy)
            # save plots
            imshow_surfs(surf, rec, "Harker_Oleary", "Harker_Oleary"+name)

    def test_harker_oleary_dirichlet(self):
        for name, surf, gradx, grady in self.surfaces:
            # reconstruct surface
            rec = harker_oleary_dirichlet(gradx, grady, self.dx, self.dy)
            # save plots
            imshow_surfs(surf, rec, "Harker_Oleary_Dirichlet", "Harker_Oleary_Dirichlet"+name)

    def test_harker_oleary_spectral(self):
        for name, surf, gradx, grady in self.surfaces:
            # reconstruct surface
            rec = harker_oleary_spectral(gradx, grady, self.dx, self.dy, mask=[150, 150])
            # save plots
            imshow_surfs(surf, rec, "Harker_Oleary_Spectral", "Harker_Oleary_Spectral"+name)

    def test_harker_oleary_tikhonov(self):
        for name, surf, gradx, grady in self.surfaces:
            # reconstruct surface
            rec = harker_oleary_tikhonov(gradx, grady, 1., self.dx, self.dy, deg=0)
            # save plots
            imshow_surfs(surf, rec, "Harker_Oleary_TikhonovDeg0", "Harker_Oleary_TikhonovDeg0"+name)

            # reconstruct surface
            rec = harker_oleary_tikhonov(gradx, grady, 1., self.dx, self.dy, deg=1)
            # save plots
            imshow_surfs(surf, rec, "Harker_Oleary_TikhonovDeg1", "Harker_Oleary_TikhonovDeg1"+name)

            # reconstruct surface
            rec = harker_oleary_tikhonov(gradx, grady, 1., self.dx, self.dy, deg=2)
            # save plots
            imshow_surfs(surf, rec, "Harker_Oleary_TikhonovDeg2", "Harker_Oleary_TikhonovDeg2"+name)

    def test_harker_oleary_weighted(self):
        for name, surf, gradx, grady in self.surfaces:
            rows, cols = gradx.shape
            Lxx = Lyx = np.eye(cols)
            Lyy = Lxy = np.eye(rows)
            rec = harker_oleary_weighted(gradx, grady, Lxx, Lxy, Lyx, Lyy, dx=self.dx, dy=self.dy)
            # save plots
            imshow_surfs(surf, rec, "Harker_Oleary_Weighted", "Harker_Oleary_Weighted"+name)

if __name__ == "__main__":
    unittest.main()