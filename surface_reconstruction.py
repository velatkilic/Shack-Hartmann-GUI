import numpy as np
from scipy.fftpack import dct, idct

def dct2(x):
    return dct( dct( x, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(x):
    return idct( idct( x, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def der_x(z, dx):
    dzdx = (z[1:,1:] - z[0:-1,1:]) / dx
    return dzdx

def der_y(z, dy):
    dzdy = (z[1:,1:] - z[1:,0:-1]) / dy
    return dzdy

def calc_laplacian(gx, gy, dx, dy):
    # zero pad
    gx = np.pad(gx, pad_width=1)
    gy = np.pad(gy, pad_width=1)

    # divergence of gradients
    rho_x = der_x(gx, dx) 
    rho_y = der_y(gy, dy)  
    rho = rho_x + rho_y

    return rho[:-1,:-1]

def frankot_chellappa(gx, gy, dx=None, dy=None):
    # Frankot-Chellappa algorithm
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")

    # fill nan with zeros
    gx, gy = map(np.nan_to_num, (gx, gy))

    rows, cols = gx.shape
    # top left corner is (0,0)
    # rows correspond to the x coord
    # columns correspond to the y coord
    wx = np.fft.fftfreq(rows, d=dx) * 2*np.pi
    wy = np.fft.fftfreq(cols, d=dy) * 2*np.pi
    wx, wy = np.meshgrid(wx, wy, indexing="ij") # ij indexing is needed so that x corresponds to rows
    d = wx**2 + wy**2

    # 2D ffts
    GX = np.fft.fft2(gx)
    GY = np.fft.fft2(gy)
    Z  = np.zeros((rows, cols), dtype=np.complex128)
    Z[1:,1:]  = - (1j*wx[1:,1:]*GX[1:,1:] + 1j*wy[1:,1:]*GY[1:,1:]) / (d[1:,1:] + 1e-16) # 1e-16 to avoid dvision by zero

    z = np.fft.ifft2(Z)
    z = z.real
    z = z - z.min() 

    return z


def poisson_solver_neumann(gx, gy, dx=None, dy=None):
    # see: Agrawal et al., "What is the range of surface reconstructions from a gradient field?" and
    # https://elonen.iki.fi/code/misc-notes/neumann-cosine/
    
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")
    if dx is None: dx = 1.
    if dy is None: dy = 1.
    # fill nan with zeros
    gx, gy = map(np.nan_to_num, (gx, gy))

    rows, cols = gx.shape

    # calculate laplacian from gradients (apply divergence)
    rho = calc_laplacian(gx, gy, dx, dy)
    RHO = dct2(rho)

    x, y = np.mgrid[0:rows, 0:cols]
    d = 2.*(np.cos(np.pi*x/rows) + np.cos(np.pi*y/cols) - 2)
    RHO[1:,1:] = RHO[1:,1:] / (d[1:,1:] + 1e-16) # add 1e-16 to avoid division by zero

    z = dx*dy*idct2(RHO)
    z = z - z.min()
    return z

