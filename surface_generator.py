import numpy as np
from sympy import symbols, lambdify, diff, cos, exp

class SurfaceGenerator:
    """Generate test surfaces for the surface reconstruction algorithms.
    
    Example:
        
        surfaces = TestSurfaceGenerator()
        for name, surf, gradx, grady in surfaces:
            reconstruct(gradx, grady)

    """
    def __init__(self):
        """
        x is along the column direction
        y is along the row direction
        """

        # spatial domain
        xb, yb = 10, 10
        Nx, Ny = 256, 512
        x = np.linspace(-xb,xb,Nx)
        y = np.linspace(-yb,yb,Ny)

        # spatial sampling
        self.dx = abs(x[1] - x[0])
        self.dy = abs(y[1] - y[0])
        self.xx, self.yy = np.meshgrid(x, y)

        # generate some test surfaces
        self.xs, self.ys = symbols('xs ys')
        self.test_names = ["cos_wave_1_2",
                           "cos_wave_5_5",
                           "Gaussian_0_5",
                           "Gaussian_0_1",
                           "Gaussian_0_20",
                           "Multi_Gaus_5_5",
                           "Multi_Gaus_10_10"]
        surfaces = [cos(2*np.pi*(self.xs + 2*self.ys)),
                    cos(2*np.pi*(5*self.xs + 5*self.ys)),
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

class CrackGradientGenerator:
    def __init__(self, ) -> None:
        xb, yb = -10, 10
        Nx, Ny = 512, 512
        x = np.linspace(-xb,xb,Nx)
        y = np.linspace(-yb,yb,Ny)

        # spatial sampling
        self.dx = abs(x[1] - x[0])
        self.dy = abs(y[1] - y[0])
        self.xx, self.yy = np.meshgrid(x, y)

    def calc_gradient(self, xc=0., yc=0., c=10.):
        """Calculate gradients near a crack tip, from the following paper:

        Miao, C. and Tippur, H.V., 2018. Higher sensitivity Digital Gradient
        Sensing configurations for quantitative visualization of stress gradients
        in transparent solids. Optics and Lasers in Engineering, 108, pp.54-67.

        Coordinate system for the equations:
            From Fig 2, x in the negative row direction (my x is along positive column)
                        y in the negative column direction (my y is along positve row)
        
        Gradient equations from equation 16 (only N=1)

        Args:
            xc (float): Crack tip position. Defaults to 0..
            yc (float): _description_. Defaults to 0..
            c (float): Constant which corresponds to nu*B/E where
                    nu -> Poisson's ratio
                    B  -> Undeformed thickness
                    E  -> Elastic modulus

        Returns:
            gradx : Gradient along x direction
            grady : Gradient along y direction
        """
        # convert from my coordinate system to that of the paper
        xx, yy = -self.yy, -self.xx
        xc, yc = -yc, -xc

        # calculate gradients
        r   = np.sqrt((xx - xc)**2 + (yy - yc)**2)
        phi = np.arctan2(yy - yc, xx - xc)
        gradx = 0.5* c * r**(-1.5) * np.cos(-1.5*phi)
        grady = 0.5* c * r**(-1.5) * np.sin(-1.5*phi)

        # filter out small radii
        ind = r < 1.
        grady[ind] = 0.
        gradx[ind] = 0.

        # convert gradients back to my coordinate system
        gradx, grady = -grady, -gradx
        
        return gradx, grady