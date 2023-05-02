import numpy as np
import os
import imageio
from skimage.io import imread_collection
import glob
from utils import normalize

IMG_TYPES = set(["rgb", "gif", "pbm","pgm","ppm","tif","tiff","rast","xbm","jpeg","jpg","bmp","png","webp","exr"])

def check_if_image(ext):
    ext = ext.split(".")[-1]
    return ext in IMG_TYPES

class Dataset:
    def __init__(self, video_name=None, image_folder=None, normalize_img=True):
        self.imgs = []
        self.length = 0
        self.is_video = False
        self.reader = None
        self.normalize_img = normalize_img
        
        if video_name is not None:
            self.set_video_name(video_name)
        elif image_folder is not None:
            self.set_image_folder(image_folder)
    
    def set_video_name(self, video_name):
        self.reader = imageio.get_reader(video_name)
        self.length = self.reader.count_frames()
        self.imgs = [None, ] * self.length
        self.is_video = True

    def set_image_folder(self, image_folder):
        img_files = glob.glob(os.path.join(image_folder, "*"))
        img_files = list(filter(check_if_image, img_files))
        self.reader = imread_collection(img_files, conserve_memory=True)
        self.is_video = False
        self.length = len(self.reader)
        self.imgs = [None, ] * self.length

    def __getitem__(self, idx):
        if self.length == 0 or idx >= self.length:
            return None

        # return cached image if exists
        if self.imgs[idx] is not None:
            return self.imgs[idx]
        
        # otherwise read to memory
        if self.is_video:
            img = np.array(self.reader.get_data(idx))
        else:
            img = np.array(self.reader[idx])
        
        if self.normalize_img:
            img = 255. * normalize(img)
            img = img.astype(np.uint8)

        if len(img.shape) < 3:
            img = np.repeat(img[:,:, None], 3, axis=2)

        # cache the result
        self.imgs[idx] = img
        return img

    def __len__(self):
        return self.length