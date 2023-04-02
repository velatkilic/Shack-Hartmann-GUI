from pathlib import Path
from skimage.feature import blob_log
import numpy as np
from scipy.interpolate import griddata

from dataset import TiffFolder

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QProgressDialog

from utils import blobs_to_centroid, bbox_to_centroid

class ViewBox(pg.ViewBox):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.init_vars()

    def init_vars(self):
        self.clear()

        # image and annot index
        self.idx = 0
        self.length = 0

        # image data
        self.imgs = {}

        # bounding box data
        self.pen = pg.mkPen(width=1, color='r')
        self.drawing = False
        self.start = None
        self.end = None
        self.prev_roi = None
        self.rois = None

        # centroid data
        self.centroids = None
        self.reference_idx = None
        self.surface_reconstructions = None

    def load_images(self, fname: Path) -> None:
        self.init_vars()
        self.tiff_folder = TiffFolder(fname)
        self.length = len(self.tiff_folder)

    def convert_rois_to_numpy(self):
        rois = np.zeros((len(self.rois), 4))
        for i, roi in enumerate(self.rois):
            rois[i,:] = roi.pos().x(), roi.pos().y(), roi.size().x(), roi.size().y()
        return rois

    def set_rois(self, rois):
        # progress bar
        pb = QProgressDialog("Rendering ROIs", "Cancel", 0, len(rois))
        pb.setWindowModality(Qt.WindowModal)
        
        pg_rois = []
        for i in range(len(rois)):
            roi = pg.RectROI(rois[i, 0:2], rois[i, 2:], pen=self.pen, removable=True, rotatable=False)
            roi.sigRemoveRequested.connect(self.remove_roi)
            self.addItem(roi)

            pg_rois.append(roi) 
            
            # update progress bar
            if pb.wasCanceled():
                break
            pb.setValue(i)
        self.rois = pg_rois

    def prev(self) -> None:
        self.clear_image()
        self.idx = (self.idx - 1) % self.length # prev index
        self.set_image()

    def next(self) -> None:
        self.clear_image()
        self.idx = (self.idx + 1) % self.length # next index
        self.set_image()
    
    def get_image(self) -> pg.ImageItem:
        # get current image if not already buffered
        if not(self.idx in self.imgs):
            img = self.tiff_folder[self.idx] 
            img = pg.ImageItem(img)
            self.imgs[self.idx] = img
        return self.imgs[self.idx]

    def clear_image(self):
        img = self.get_image()
        self.removeItem(img)

    def set_image(self) -> None:
        img = self.get_image()
        self.addItem(img)

    def make_roi(self, start: QPoint, end: QPoint) -> pg.RectROI:
        # width and height
        w = end.x() - start.x()
        #h = end.y() - start.y()

        # make roi 
        # roi = pg.RectROI(start, [w, h], pen=self.pen,
        #             removable=True, rotatable=False) 

        # lock aspect ratio 
        roi = pg.RectROI(start, [w, w], pen=self.pen,
                    removable=True, rotatable=False)   
        self.addItem(roi)

        # removing roi from context menu
        roi.sigRemoveRequested.connect(self.remove_roi)
        
        return roi
    
    def add_roi_to_list(self, roi: pg.RectROI) -> None:
        # if dict entry exists, append to the list
        if self.idx in self.rois:
            self.rois.append(roi)
        # if not, make a new list
        else:
            self.rois = [roi]
    
    def remove_roi(self, roi: pg.RectROI) -> None:
        # find roi in the list
        idx = self.rois.index(roi)

        # remove from the list
        del self.rois[idx]

        # remove from the viewbox
        self.removeItem(roi)

    def mouseClickEvent(self, event: QMouseEvent) -> None:
        # left button for drawing bounding boxes
        if event.button() == Qt.LeftButton:

            # map coordinates
            pos = self.mapSceneToView(event.pos())
            
            # draw roi
            if self.drawing:
                self.drawing = False
                self.end = pos
                roi = self.make_roi(self.start, self.end)
                self.add_roi_to_list(roi)
                self.removeItem(self.prev_roi)
                self.prev_roi = None
            else:
                self.drawing = True
                self.start = pos

    def hoverEvent(self, event):
        # show bounding box while drawing is on
        if self.drawing:
            # delete previous bounding box
            if self.prev_roi is not None:
                self.removeItem(self.prev_roi)
            
            # get current mouse position
            pos = self.mapSceneToView(event.pos())

            # visual feedback
            # set current as end and draw a bounding box
            roi = self.make_roi(self.start, pos)
            self.prev_roi = roi
    
    def run_blob(self, blob_log_params, box_size):
        # delete prev rois
        if self.rois is not None:
            self.clear()
            self.set_image()

        # detect blobs
        img = self.tiff_folder[self.idx] 
        blobs = blob_log(img, **blob_log_params)

        # refine blob predictions
        centroids = blobs_to_centroid(img, blobs)

        # progress bar
        pb = QProgressDialog("Calculating ROIs", "Cancel", 0, len(centroids))
        pb.setWindowModality(Qt.WindowModal)

        # convert bboxes to rois
        rois = []
        r = box_size/2.
        for i, (row, col) in enumerate(centroids):
            # roi pos is the lower left corner:
            roi = pg.RectROI([col-r, row-r], [box_size, box_size], pen=self.pen, removable=True, rotatable=False)
            roi.sigRemoveRequested.connect(self.remove_roi)
            self.addItem(roi)

            rois.append(roi) 
            
            # update progress bar
            if pb.wasCanceled():
                break
            pb.setValue(i)

        self.rois = rois
        self.reference_idx = self.idx # record reference frame

    def calc_centroids(self):
        # progress bar
        pb = QProgressDialog("Calculating Centroids ...", "Cancel", 0, self.length)
        pb.setWindowModality(Qt.WindowModal)

        self.centroids = np.zeros((self.length, len(self.rois), 2)) 
        for i in range(self.length):
            img = self.tiff_folder[i]
            for j, roi in enumerate(self.rois):
                self.centroids[i,j,:] = bbox_to_centroid(img, roi)
            
            # update progress bar
            if pb.wasCanceled():
                break
            pb.setValue(i)
    
    def surf_rec(self):
        if self.reference_idx is None:
            raise NotImplementedError("Reference index must be set with blob detection")
        
        # calculate shifts with respect to the reference frame
        center = self.centroids[self.reference_idx, :, :]
        shifts = self.centroids - center
        
        dx = abs(center[0,0] - center[1,0])
        dy = abs(center[0,1] - center[1,1])

        xmin = center[:,0].astype(np.intc).min()
        xmax = center[:,0].astype(np.intc).max()
        ymin = center[:,1].astype(np.intc).min()
        ymax = center[:,1].astype(np.intc).max()

        xq, yq = np.mgrid[xmin:xmax, ymin:ymax]

        self.surface_reconstructions = np.zeros((len(self.centroids), xmax-xmin, ymax-ymin))
        for i in range(len(shifts)):
            gx = griddata(center, self.centroids[i, :, 0], (xq,yq), method='linear')
            gy = griddata(center, self.centroids[i, :, 1], (xq,yq), method='linear')

            self.surface_reconstructions[i,:,:] = self.frankot_chellappa(gx, gy, dx, dy)

