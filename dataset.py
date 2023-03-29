from skimage import io
import glob
import os
import numpy as np

def normalize(img):
    img = img.astype(np.float64)
    mx = img.max()
    mn = img.min()
    df = mx - mn
    img -= mn
    if df != 0:
        img /= df
    return img


class TiffFolder:
    def __init__(self, image_folder, normalize_img=True):
        self.file_names = glob.glob(os.path.join(image_folder, "*.tiff"))
        self.normalize_img = normalize_img
        
        # buffer images in memory (faster)
        self.data = [None, ] * len(self.file_names)

    def __getitem__(self, idx):
        if len(self) > 0:
            if self.data[idx] is None:
                img =  np.array(io.imread(self.file_names[idx]))
                if self.normalize_img:
                    img = normalize(img)
                self.data[idx] = img
            else:
                img = self.data[idx]
            return img
        else:
            return None

    def __len__(self):
        return len(self.file_names)