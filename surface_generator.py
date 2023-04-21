import numpy as np
from sympy import symbols, lambdify, diff, sin, cos, exp

class TestSurfaceGenerator:
    """Generate test surfaces for the surface reconstruction algorithms.
    
    Example:
        
        surfaces = TestSurfaceGenerator()
        for name, surf, gradx, grady in surfaces:
            reconstruct(gradx, grady)

    """
    def __init__(self):
        # spatial domain
        xb, yb = 10, 10
        Nx, Ny = 512, 256
        x = np.linspace(-xb,xb,Nx)
        y = np.linspace(-yb,yb,Ny)

        # spatial sampling
        self.dx = abs(x[1] - x[0])
        self.dy = abs(y[1] - y[0])
        self.xx, self.yy = np.meshgrid(x, y, indexing="ij")

        # generate some test surfaces
        self.xs, self.ys = symbols('xs ys')
        self.test_names = ["cos_wave_1_2",
                           "sin_wave_2_2",
                           "Gaussian_0_5",
                           "Gaussian_0_1",
                           "Gaussian_0_20",
                           "Multi_Gaus_5_5",
                           "Multi_Gaus_10_10"]
        surfaces = [cos(2*np.pi*(self.xs + 2*self.ys)),
                    sin(2*np.pi*(2*self.xs + 2*self.ys)),
                    exp(-0.5*(self.xs**2 + self.ys**2)/5.),
                    exp(-0.5*(self.xs**2 + self.ys**2)),
                    exp(-0.5*(self.xs**2 + self.ys**2)/20),
                    exp(-0.5*(self.xs**2 + self.ys**2)/5.) + exp(-0.5*((self.xs - 5)**2 + (self.ys - 5)**2)/5.),
                    exp(-0.5*(self.xs**2 + self.ys**2)/10.) + exp(-0.5*((self.xs - 5)**2 + (self.ys - 5)**2)/10.),
                    exp(-0.5*(self.xs**2 + self.ys**2)/10.) + exp(-0.5*((self.xs - 10)**2 + (self.ys - 10)**2)/10.),
                    ]
        gx, gy = self._calc_grads(surfaces)

        # make lambda functions which can be evaluated on numpy arrays
        self.test_surfaces = [self._lambdify(func) for func in surfaces]
        self.gradx = [self._lambdify(func) for func in gx]
        self.grady = [self._lambdify(func) for func in gy]

    def _calc_grads(self, surfaces):
        """Calculate symbolic functions for the gradients
        """
        gradx = []
        grady = []
        for surf in surfaces:
            gradx.append(diff(surf, self.xs))
            grady.append(diff(surf, self.ys))
        return gradx, grady
    
    def _lambdify(self, func):
        """Generate lambda functions from symbolic ones that can be evaluated on numpy arrays
        """
        return lambdify([self.xs, self.ys], func, "numpy")

    def get_dx_dy(self):
        """Getter for spatial sampling
        """
        return self.dx, self.dy

    def __getitem__(self, idx):
        # evaluate at the spatial domain
        surf  = self.test_surfaces[idx](self.xx, self.yy)
        gradx = self.gradx[idx](self.xx, self.yy)
        grady = self.grady[idx](self.xx, self.yy)

        name = self.test_names[idx]

        return name, surf, gradx, grady

    def __len__(self):
        return len(self.test_surfaces)