import numpy as np

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