import numpy as np
from sympy import symbols, lambdify, diff, cos, exp
from scipy.signal import convolve2d

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

def calc_crack_grad_n1(xx, yy, k_1, h_sample, poissons_ratio, youngs_modulus, filter_output=True):
    """Calculate gradients near a crack tip, from the following paper:

        Miao, C. and Tippur, H.V., 2018. Higher sensitivity Digital Gradient
        Sensing configurations for quantitative visualization of stress gradients
        in transparent solids. Optics and Lasers in Engineering, 108, pp.54-67.

        Coordinate system for the equations:
            From Fig 2, x in the negative row direction
                        y in the negative column direction
        
        Gradient equations from equation 16 (only N=1), divide by 2 to get gradient

    Args:
        xx (numpy array): (m, n) position array, likely calculated from meshgrid. x direction along the crack tip
        yy (numpy array): (m, n) position array, likely calculated from meshgrid. y direction normal to the crack tip
        k_1 (float): Mode I stress intensity factor
        h_sample (float): Sample thickness
        poissons_ratio (float): Poisson's ratio
        youngs_modulus (float): Young's modulus

    Returns:
        gradx (numpy array) : gradient along x direction
        grady (numpy array) : gradient along y direction
        filt_final (numpy array): boolean filter. True corresponds to excluded regions
    """
    # polar coordinates
    r     = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # gradient from first order expansion
    a_1 = k_1 * np.sqrt(2./np.pi)
    c_1 = 0.25 * a_1* h_sample * poissons_ratio / youngs_modulus
    gradx = c_1 * r**(-1.5) * np.cos(-1.5*theta)
    grady = c_1 * r**(-1.5) * np.sin(-1.5*theta)

    # mismatch between equation and experiment results! TODO: check with paper authors
    grady = -grady

    # filter 0.5 <= r/thickness <= 1.5 and -135 deg <= theta <= 135deg
    # discussed right after equation 16
    if filter_output:
        radius_ratio = r / h_sample
        filt_radius = (radius_ratio >= 0.5) & (radius_ratio <= 1.5)
        angle_rads = np.pi*135./180.
        filt_angle = (theta >= -angle_rads) & (theta <= angle_rads)
        filt_final = np.bitwise_not(filt_radius & filt_angle)
    else:
        filt_final = np.zeros(gradx.shape, dtype=np.bool_)

    return gradx, grady, filt_final

class ShackHartmannMeasGenerator:
    def __init__(self, surf_gen, micro_lens_count=64):
        self.surf_gen = surf_gen

        # Shack Hartmann will essentially average
        self.kernel_size_x =  self.surf_gen.Nx // micro_lens_count
        self.kernel_size_y =  self.surf_gen.Ny // micro_lens_count
        self.kernel = np.ones((self.kernel_size_y, self.kernel_size_x))

        # center
        xpos = np.arange(0, self.surf_gen.Nx, self.kernel_size_x)
        ypos = np.arange(0, self.surf_gen.Ny, self.kernel_size_y)
        self.center = np.meshgrid(xpos, ypos)
    
    def generate_measurement(self):
        # gradient field
        gradx, grady = self.surf_gen.calc_gradient()

        # Shack Hartmann will essentially average
        gradx_downsampled = convolve2d(gradx, self.kernel, mode="same")
        grady_downsampled = convolve2d(grady, self.kernel, mode="same")

        # decimate
        gradx_meas = gradx_downsampled[::self.kernel_size_y, ::self.kernel_size_x]
        grady_meas = grady_downsampled[::self.kernel_size_y, ::self.kernel_size_x]
        return gradx_meas, grady_meas
    
