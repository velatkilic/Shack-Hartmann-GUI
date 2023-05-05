import unittest
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

from surface_reconstruction import (frankot_chellappa,
                                    poisson_solver_neumann,
                                    harker_oleary,
                                    harker_oleary_dirichlet,
                                    harker_oleary_spectral,
                                    harker_oleary_tikhonov,
                                    harker_oleary_weighted,
                                    reconstruct_surface_from_sh)
from surface_generator import SurfaceGenerator, calc_crack_grad_n1

def imshow_surfs(gnd, rec, algo_name, save_name):
    save_dir = os.path.join(os.getcwd(), "figures", algo_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    gnd -= gnd.mean()
    rec -= rec.mean()

    vmin = np.minimum(gnd.min(), rec.min())
    vmax = np.maximum(gnd.max(), rec.max())

    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].imshow(gnd, cmap="jet", vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth")
    im = axs[1].imshow(rec, cmap="jet", vmin=vmin, vmax=vmax)
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

def imshow_surf_experimental_data(gradx, grady, rec, folder_name, save_name):
    save_dir = os.path.join(os.getcwd(), "figures", folder_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    vmin = np.minimum(gradx.min(), grady.min())
    vmax = np.maximum(gradx.max(), grady.max())
    
    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].imshow(gradx, cmap="jet", vmin=vmin, vmax=vmax)
    axs[0].set_title("Grad x")
    im = axs[1].imshow(grady, cmap="jet", vmin=vmin, vmax=vmax)
    axs[1].set_title("Grad y")
    fig.colorbar(im, ax=axs[1], shrink=0.5)
    plt.savefig(os.path.join(save_dir,save_name+"_grad.png"))
    plt.close()

    plt.figure()
    plt.imshow(rec, cmap="jet")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,save_name+"_rec.png"))
    plt.close()

class TestSurfaceReconstruction(unittest.TestCase):
    def setUp(self):
        # test surface generator class
        self.surfaces = SurfaceGenerator()
        self.dx, self.dy = self.surfaces.get_dx_dy()

        # Need to visually check surface reconstruction quality
        self.fig_save_dir = os.path.join(os.getcwd(), "figures")
        if not os.path.exists(self.fig_save_dir):
            os.mkdir(self.fig_save_dir)

    def test_harker_oleary_with_gradient_in_paper_coord(self):
        # generate gradients
        N = 512
        ub = 30e-3
        x = np.linspace(ub, -ub, N)
        y = np.linspace(1.5*ub, -0.5*ub, N)
        yy, xx = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        gradx, grady, filt_final = calc_crack_grad_n1(xx, yy, # m
                                                      k_1 = 0.9, # MPa * m^-1/2
                                                      h_sample = 8.6e-3, # m
                                                      poissons_ratio = 0.34,
                                                      youngs_modulus = 3.3e3, # MPa
                                                      )
        # convert to my coordinate system
        dx, dy = -dy, -dx
        gradx, grady = -grady, -gradx

        # reconstruct surface
        rec = harker_oleary(gradx, grady, dx, dy)
        rec[filt_final] = 0.

        # plot results
        plt.figure()
        plt.imshow(rec, cmap="jet")
        plt.colorbar()
        plt.savefig(os.path.join(self.fig_save_dir, "crack_grad_rec_paper_coord.png"))
        plt.close()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
        surf = ax.plot_surface(xx, yy, rec)
        # ax.dist = 10
        plt.savefig(os.path.join(self.fig_save_dir, "crack_grad_rec_surf_paper_coord.png"))
        plt.close()
    
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
    
    def test_reconstruct_surface_from_sh(self):
        data = sio.loadmat(os.path.join(self.fig_save_dir, "exp_data.mat"))
        center = data["center"]
        shifts = data["shifts"]
        for i in range(len(shifts)):
            rec, xq, yq, gx, gy = reconstruct_surface_from_sh(center, shifts[i,:,:], 256)
            imshow_surf_experimental_data(gx, gy, rec, "exp_data", "exp_data"+str(i))


if __name__ == "__main__":
    unittest.main()