import numpy as np
from scipy.fftpack import dct, idct
from scipy import linalg

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

def frankot_chellappa(gx, gy, dx=1., dy=1.):
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
    z = z - z.mean() 

    return z


def poisson_solver_neumann(gx, gy, dx=1., dy=1.):
    # see: Agrawal et al., "What is the range of surface reconstructions from a gradient field?" and
    # https://elonen.iki.fi/code/misc-notes/neumann-cosine/
    
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")

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
    z = z - z.mean()
    return z

# Harker, M., O’Leary, P., Regularized Reconstruction of a Surface from its Measured Gradient Field
def D_central(N, dx):
    D = np.diag(-np.ones(N-1,),1)+np.diag(np.ones(N-1,),-1)
    D[0,0:3] = [-3., 4., -1.]
    D[-1,-3:] = [1., -4., 3]
    D = D / (2.*dx)
    return D

def harker_oleary_dirichlet(gx, gy, dx=1., dy=1., ZB=None):
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")
    rows, cols = gx.shape

    if ZB is None:
        ZB = np.zeros((rows, cols))

    P  = np.diag(np.ones(rows-1,),-1)[:,:-2]
    Q  = np.diag(np.ones(cols-1,),-1)[:,:-2]
    Dy = D_central(rows, dx)
    Dx = D_central(cols, dy)

    A = Dy @ P
    B = Dx @ Q

    F = ( gy - Dy @ ZB ) @ Q 
    G = P.T @ ( gx - ZB @ Dx.T ) 

    ZB[1:-1,1:-1] += linalg.solve_sylvester( A.T @ A, B.T @ B, A.T @ F + G @ B )
    return ZB


def harker_oleary(gx, gy, dx=1., dy=1.):
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")
    rows, cols = gx.shape
    u = np.ones((rows, 1))
    v = np.ones((cols, 1))

    Dy = D_central(rows, dx)
    Dx = D_central(cols, dy)

    return sylvester_solver(Dy, Dx, gy, gx, u, v)

def sylvester_solver(A, B, F, G, u, v):
    '''
    Solution to:
    A'A Phi + Phi B'B - A'F - GB = 0

    u and v are the null vectors of A and B respectively
    '''

    # Householder vectors
    m = len(u)
    n = len(v)
    u[0] += np.linalg.norm(u,2)
    u = np.sqrt(2) * u / np.linalg.norm(u, 2)

    v[0] += np.linalg.norm(v, 2)
    v = np.sqrt(2) * v / np.linalg.norm(v)

    # Householder updates
    A = A - ( A @ u ) @ u.T
    B = B - ( B @ v ) @ v.T 
    F = F - ( F @ v ) @ v.T 
    G = G - u @ ( u.T @ G )

    # solve
    Phi = np.zeros((m, n))
    Phi[0,1:] = np.linalg.lstsq(B[:,1:], G[0,:].reshape(-1,1), rcond=None)[0].T
    Phi[1:,0] = np.linalg.lstsq(A[:,1:], F[:,0], rcond=None)[0]
    Phi[1:,1:] = linalg.solve_sylvester(A[:,1:].T @ A[:,1:], B[:,1:].T @ B[:,1:], A[:,1:].T @ F[:,1:] + G[1:,:] @ B[:,1:])

    # Invert
    Phi = Phi - u @ (u.T @ Phi)
    Phi = Phi - (Phi @ v) @ v.T

    return Phi
