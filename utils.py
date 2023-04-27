import numpy as np

def centroid1D(arr, th_tot_weight=1e-1):
    # check if 1D
    if len(arr.shape) > 1: return None
    # check if positive
    if np.any(arr < 0): return None
    # calculate centroid
    inds = np.arange(0., len(arr), 1.)
    arr = arr.astype(np.float32)
    tot_weight = sum(arr)
    if tot_weight < th_tot_weight:
        return 0
    else:
        return sum(arr*inds)/tot_weight

def centroid2D(crop):
    if len(crop.shape) != 2: return None, None
    col = centroid1D(crop.sum(0))
    row = centroid1D(crop.sum(1))
    return row, col

def center_to_bbox(row, col, s, shape):
    row0 = max(row - s, 0)
    col0 = max(col - s, 0)
    # These are used for 2D image array indexing
    # therefore, last index non-inclusive, hence +1
    row1 = min(row + s, shape[0]) + 1
    col1 = min(col + s, shape[1]) + 1
    return row0,col0,row1,col1

def blobs_to_centroid(img, blobs, con=3.):
    out = []
    for i, (row,col,s) in enumerate(blobs):
        s *= con # make side length larger
        bbox = center_to_bbox(row, col, s, img.shape)
        row0,col0,row1,col1 = map(int, bbox)
        crop = img[row0:row1,col0:col1]
        drow, dcol = centroid2D(crop)
        out.append([row0+drow, col0+dcol])
    return out

def roi_to_centroid(img, roi):
    # aspect ratio is locked
    w = roi.size().x()
    row0 = int(roi.pos().y() - w)
    col0 = int(roi.pos().x() - w)
    row1 = int(roi.pos().y())
    col1 = int(roi.pos().x())

    crop = img[row0:row1,col0:col1]
    drow, dcol = centroid2D(crop)
    return [drow+row0, dcol+col0] 