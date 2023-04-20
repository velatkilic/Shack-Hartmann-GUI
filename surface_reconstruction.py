import numpy as np
import numpy.typing as npt
from scipy.fftpack import dct, idct
from scipy import linalg

def dct2(x: npt.NDArray) -> npt.NDArray:
    """2D discrete cosine transform (DCT)

    Args:
        x (npt.NDArray): 2D signal (m,n)

    Returns:
        npt.NDArray: 2D DCT transform of the input
    """
    return dct( dct( x, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(x: npt.NDArray) -> None:
    """2D inverse discrete cosine transform

    Args:
        x (npt.NDArray): 2D signal (m, n)

    Returns:
        npt.NDArray: 2D inverse DCT transform of the input
    """
    return idct( idct( x, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def der_x(z: npt.NDArray, dx: float) -> None:
    """Discrete first order derivate in x direction (row difference)

    Args:
        z (npt.NDArray): 2D signal (m, n)
        dx (float): Sampling distance along the x direction (row direction)

    Returns:
        npt.NDArray: dz/dx partial derivative along x
    """
    dzdx = (z[1:,1:] - z[0:-1,1:]) / dx
    return dzdx

def der_y(z: npt.NDArray, dy: float) -> None:
    """Discrete first order derivate in x direction (column difference)

    Args:
        z (npt.NDArray): 2D signal (m, n)
        dy (float): Sampling distance along the y direction (column direction)

    Returns:
        npt.NDArray: dz/dy partial derivative along y
    """
    dzdy = (z[1:,1:] - z[1:,0:-1]) / dy
    return dzdy

def calc_laplacian(gx: npt.NDArray, gy: npt.NDArray, dx: float, dy: float) -> None:
    """Calculate Laplacian from gradient data

    Args:
        gx (npt.NDArray): surface gradient in x direction
        gy (npt.NDArray): surface gradient in y direction
        dx (float): sampling space in x
        dy (float): sampling space in y

    Returns:
        npt.NDArray: Discrete Laplacian calcualted from the gradients
    """
    # zero pad
    gx = np.pad(gx, pad_width=1)
    gy = np.pad(gy, pad_width=1)

    # divergence of gradients
    rho_x = der_x(gx, dx) 
    rho_y = der_y(gy, dy)  
    rho = rho_x + rho_y

    return rho[:-1,:-1]

def frankot_chellappa(gx: npt.NDArray, gy: npt.NDArray, dx: float = 1., dy: float = 1.) -> None:
    """Surface reconstruction using the Frankot Chellappa algorithm:
    
    Frankot, Robert T., and Rama Chellappa. "A method for enforcing 
    integrability in shape from shading algorithms." IEEE Transactions 
    on pattern analysis and machine intelligence 10.4 (1988): 439-451.
    
    Args:
        gx (npt.NDArray): Gradient in x direction (m, n)
        gy (npt.NDArray): Gradient in y direction (m, n)
        dx (float, optional): Sampling space in x. Defaults to 1.
        dy (float, optional): Sampling space in y. Defaults to 1.

    Returns:
        npt.NDArray: Reconstructed surface (m, n)
    """
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


def poisson_solver_neumann(gx: npt.NDArray, gy: npt.NDArray, dx: float = 1., dy: float = 1.) -> None:
    """Surface reconstruction from gradient measurements using Poisson Solver

    1. Agrawal, Amit, Ramesh Raskar, and Rama Chellappa. "What is the range of surface 
    reconstructions from a gradient field?." Computer Vision ECCV 2006: 9th European 
    Conference on Computer Vision, Graz, Austria, May 7-13, 2006.

    2. https://elonen.iki.fi/code/misc-notes/neumann-cosine/

    Args:
        gx (npt.NDArray): Gradient in x direction (m, n)
        gy (npt.NDArray): Gradient in y direction (m, n)
        dx (float, optional): Sampling space in x. Defaults to 1.
        dy (float, optional): Sampling space in y. Defaults to 1.

    Returns:
        npt.NDArray: Reconstructed surface (m, n)
    """
    
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

def D_central(N: int, dx: float) -> npt.NDArray:
    """Calculate first derivative matrix corresponding to central difference

    Harker, Matthew, and Paul Oleary. "Regularized reconstruction of a 
    surfacefrom its measured gradient field: algorithms for spectral, Tikhonov, 
    constrained, and weighted regularization." Journal of Mathematical Imaging 
    and Vision 51 (2015): 46-70.

    Args:
        N (int): Size of the output matrix
        dx (float): Spatial sampling

    Returns:
        npt.NDArray: (N, N) Central difference operator
    """
    D = np.diag(-np.ones(N-1,),1)+np.diag(np.ones(N-1,),-1)
    D[0,0:3] = [-3., 4., -1.]
    D[-1,-3:] = [1., -4., 3]
    D = D / (2.*dx)
    return D

def harker_oleary(gx, gy, dx=1., dy=1.):
    """Surface reconstruction using the Harker-Oleary algorithm

    Harker, Matthew, and Paul Oleary. "Regularized reconstruction of a 
    surfacefrom its measured gradient field: algorithms for spectral, Tikhonov, 
    constrained, and weighted regularization." Journal of Mathematical Imaging 
    and Vision 51 (2015): 46-70.

    Args:
        gx (npt.NDArray): Gradient in x direction (m, n)
        gy (npt.NDArray): Gradient in y direction (m, n)
        dx (float, optional): Sampling space in x. Defaults to 1.
        dy (float, optional): Sampling space in y. Defaults to 1.

    Returns:
        npt.NDArray: Reconstructed surface (m, n)
    """
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")
    rows, cols = gx.shape
    u = np.ones((rows, 1))
    v = np.ones((cols, 1))

    Dy = D_central(rows, dx)
    Dx = D_central(cols, dy)

    return sylvester_solver(Dy, Dx, gy, gx, u, v)

def sylvester_solver(A: npt.NDArray, B: npt.NDArray, F: npt.NDArray , G: npt.NDArray, u: npt.NDArray, v: npt.NDArray) -> None:
    """Solution to:
    
    A'A Phi + Phi B'B - A'F - GB = 0

    u and v are the null vectors of A and B respectively

    Harker, Matthew, and Paul Oleary. "Regularized reconstruction of a 
    surfacefrom its measured gradient field: algorithms for spectral, Tikhonov, 
    constrained, and weighted regularization." Journal of Mathematical Imaging 
    and Vision 51 (2015): 46-70.

    Args:
        A (npt.NDArray): (m, m) matrix
        B (npt.NDArray): (n, n) matrix
        F (npt.NDArray): (m, n) matrix
        G (npt.NDArray): (m, n) matrix
        u (npt.NDArray): (m, 1) vector
        v (npt.NDArray): (n, 1) vector

    Returns:
        npt.NDArray: Solution to the Sylvecter equation (Phi)
    """
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

def harker_oleary_dirichlet(gx: npt.NDArray, gy: npt.NDArray, dx: float = 1., dy: float = 1., ZB: npt.NDArray = None) -> npt.NDArray:
    """Surface reconstruction using the Harker-Oleary algorithm using Dirichlet boundary conditions

    Harker, Matthew, and Paul Oleary. "Regularized reconstruction of a 
    surfacefrom its measured gradient field: algorithms for spectral, Tikhonov, 
    constrained, and weighted regularization." Journal of Mathematical Imaging 
    and Vision 51 (2015): 46-70.

    Args:
        gx (npt.NDArray): Gradient in x direction (m, n)
        gy (npt.NDArray): Gradient in y direction (m, n)
        dx (float, optional): Sampling space in x. Defaults to 1.
        dy (float, optional): Sampling space in y. Defaults to 1.
        ZB (npt.NDArray): Dirichlet boundary conditions (m, n)

    Returns:
        npt.NDArray: Reconstructed surface (m, n)
    """
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

def harker_oleary_spectral(gx: npt.NDArray, gy: npt.NDArray, dx: float = 1., dy: float = 1., mask: list = None, Bx: npt.NDArray = None, By: npt.NDArray = None) -> npt.NDArray:
    """Surface reconstruction using the Harker-Oleary algorithm using basis functions

    Harker, Matthew, and Paul Oleary. "Regularized reconstruction of a 
    surfacefrom its measured gradient field: algorithms for spectral, Tikhonov, 
    constrained, and weighted regularization." Journal of Mathematical Imaging 
    and Vision 51 (2015): 46-70.

    Args:
        gx (npt.NDArray): Gradient in x direction (m, n)
        gy (npt.NDArray): Gradient in y direction (m, n)
        dx (float, optional): Sampling space in x. Defaults to 1.
        dy (float, optional): Sampling space in y. Defaults to 1.
        mask (list, optional): List of filtering rows and columns. e.g. [50, 10] uses 
        50 basis functions from By and 10 from Bx Defaults to None.
        Bx (npt.NDArray, optional): Basis functions for the x direction. Defaults to DCT.
        By (npt.NDArray, optional): Basis functions for the y direction. Defaults to DCT.

    Returns:
        npt.NDArray: Reconstructed surface (m, n)
    """
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")
    rows, cols = gx.shape

    if mask is None:
        p = rows
        q = cols
    else:
        p, q = mask

    if Bx is None:
        Bx = dct(np.eye(cols), norm="ortho")
        Bx = Bx[:,0:q]
    if By is None:
        By = dct(np.eye(rows), norm="ortho")
        By = By[:,0:p]

    Dy = D_central(rows, dx)
    Dx = D_central(cols, dy)

    A = Dy @ By
    B = Dx @ Bx

    F = gy @ Bx
    G = By.T @ gx

    C = np.zeros((p,q))

    C[0,1:q] = np.linalg.lstsq(B[:,1:q], G[0,:].reshape(-1,1), rcond=None)[0].T
    C[1:p,0] = np.linalg.lstsq(A[:,1:p], F[:,0], rcond=None)[0]
    C[1:p,1:q] = linalg.solve_sylvester( A[:,1:p].T @ A[:,1:p], B[:,1:q].T @ B[:,1:q], A[:,1:p].T @ F[:,1:] + G[1:,:] @ B[:,1:q] )

    Z = By @ C @ Bx.T

    return Z.real # real in case FFT was used for B

def harker_oleary_tikhonov(gx: npt.NDArray, gy: npt.NDArray, lam: float, dx: float = 1., dy: float = 1., deg: int = 0, Z0: npt.NDArray = None) -> npt.NDArray:
    """Surface reconstruction using the Harker-Oleary algorithm with Tikhonov regularization

    Harker, Matthew, and Paul Oleary. "Regularized reconstruction of a 
    surfacefrom its measured gradient field: algorithms for spectral, Tikhonov, 
    constrained, and weighted regularization." Journal of Mathematical Imaging 
    and Vision 51 (2015): 46-70.

    Args:
        gx (npt.NDArray): Gradient in x direction (m, n)
        gy (npt.NDArray): Gradient in y direction (m, n)
        lam (float): Regularization parameter
        dx (float, optional): Sampling space in x. Defaults to 1.
        dy (float, optional): Sampling space in y. Defaults to 1.
        deg (int, optional): Degree. Defaults to 0.
        Z0 (npt.NDArray, optional): Surface estimate. Defaults to unregularized solution (m, n).

    Returns:
        npt.NDArray: Reconstructed surface (m, n)
    """
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")
    rows, cols = gx.shape

    if isinstance(lam, tuple | list) and len(lam) == 2:
        mu=lam[1]
        lam = lam[0]
    else:
        mu = lam

    if Z0 is None: # shoudl this be zeros by default?
        Z0 = harker_oleary(gx, gy, dx, dy)

    Dy = D_central(rows, dx)
    Dx = D_central(cols, dy)

    if deg==0:
        A = np.vstack((Dy, mu*np.eye(rows)))
        B = np.vstack((Dx, lam*np.eye(cols)))
        F = np.vstack((gy, mu*Z0))
        G = np.hstack((gx, lam*Z0))
    
        Z = linalg.solve_sylvester(A.T @ A, B.T @ B, A.T @ F + G @ B)
    
    else:
        Dyk = Dy**deg
        Dxk = Dx**deg

        A = np.vstack((Dy, Dyk))
        B = np.vstack((Dx, Dxk))
        F = np.vstack((gy, mu* Dyk @ Z0))
        G = np.hstack((gx, lam*Z0 @ Dxk.T))
        u = np.ones((rows, 1))
        v = np.ones((cols, 1))

        Z = sylvester_solver(A, B, F, G, u, v)
    
    return Z

def harker_oleary_weighted(gx: npt.NDArray, gy: npt.NDArray, Lxx: npt.NDArray, Lxy: npt.NDArray, Lyx: npt.NDArray, Lyy: npt.NDArray, dx: float = 1.0, dy: float = 1.) -> npt.NDArray:
    """Surface reconstruction using the Harker-Oleary algorithm with weights

    Harker, Matthew, and Paul Oleary. "Regularized reconstruction of a 
    surfacefrom its measured gradient field: algorithms for spectral, Tikhonov, 
    constrained, and weighted regularization." Journal of Mathematical Imaging 
    and Vision 51 (2015): 46-70.

    Args:
        gx (npt.NDArray): Gradient in x direction (m, n)
        gy (npt.NDArray): Gradient in y direction (m, n)        
        Lxx (npt.NDArray): (n, n) Covariance matrix x component along x direction
        Lxy (npt.NDArray): (m, m) Covariance matrix x component along y direction
        Lyx (npt.NDArray): (n, n) Covariance matrix y component along x direction
        Lyy (npt.NDArray): (m, m) Covariance matrix y component along y direction
        dx (float, optional): Sampling space in x. Defaults to 1.
        dy (float, optional): Sampling space in y. Defaults to 1.

    Returns:
        npt.NDArray: Reconstructed surface (m, n)
    """
    # check if shape is correct
    if gx.shape != gy.shape:
        raise ValueError("Gradient shapes must match")
    rows, cols = gx.shape

    Dy = D_central(rows, dx)
    Dx = D_central(cols, dy)

    Wxx = linalg.sqrtm( Lxx )
    Wxy = linalg.sqrtm( Lxy )
    Wyx = linalg.sqrtm( Lyx )
    Wyy = linalg.sqrtm( Lyy )
    u = np.linalg.lstsq(Wxy, np.ones((rows, 1)), rcond=None)[0]
    v = np.linalg.lstsq(Wyx, np.ones((cols, 1)), rcond=None)[0]
    
    A = np.linalg.lstsq(Wyy, Dy @ Wxy, rcond=None)[0]
    B = np.linalg.lstsq(Wxx, Dx @ Wyx, rcond=None)[0]

    t = np.linalg.lstsq(Wyx.T, gy.T, rcond=None)[0].T
    F = np.linalg.lstsq(Wyy, t, rcond=None)[0]

    t = np.linalg.lstsq(Wxx.T, gx.T, rcond=None)[0].T
    G = np.linalg.lstsq(Wxy, t, rcond=None)[0]

    Z = sylvester_solver(A, B, F, G, u, v)
    Z = Wxy @ Z @ Wyx 
    return Z
