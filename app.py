import sys
import os
from pathlib import Path
import scipy.io as sio

from gui import Ui_MainWindow
from viewbox import ViewBox
from roi import ROI

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication,
                            QMainWindow,
                            QFileDialog,
                            )
import pyqtgraph as pg
import numpy as np

pg.setConfigOption('imageAxisOrder', 'row-major') # best performance

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        # set-up the GUI
        self.setupUi(self)

        # main window title
        self.setWindowTitle("Shack - Hartmann")

        # view_box holds images and ROIs
        self.view_box = ViewBox(self, lockAspect=True, invertY=True)
        self.hist = pg.HistogramLUTItem()

        self.img_view.addItem(self.view_box, row=0, col=0, rowspan=1, colspan=1)
        self.img_view.addItem(self.hist, row=0, col=1, rowspan=1, colspan=1)

        # action menu items: File
        self.action_load_images.triggered.connect(self.load_images)
        self.action_load_annot.triggered.connect(self.load_annot)
        self.action_save.triggered.connect(self.save_annot)

        # buttons: blob detection, surface reconstruction
        self.roi_dialog = ROI()
        self.button_calc_roi.clicked.connect(self.run_blob)
        self.button_calc_centroids.clicked.connect(self.calc_centroids)
        self.button_surf_rec.clicked.connect(self.surf_rec)

        # prev/next buttons
        self.spin_frame_id.valueChanged.connect(self.navigate_to_idx)
        self.button_prev.clicked.connect(self.prev)
        self.button_next.clicked.connect(self.next)

        # remember last position
        self.last_folder = os.getcwd()

    def navigate_to_idx(self, idx):
        self.view_box.navigate_to_idx(idx)

    def prev(self):
        idx = self.view_box.prev()
        self.spin_frame_id.setValue(idx)

    def next(self):
        idx = self.view_box.next()
        self.spin_frame_id.setValue(idx)
    
    def update_frame_id(self) -> None:
        frame_count = len(self.view_box.dset)
        self.label_frame_count.setText("/" + str(frame_count - 1))
        self.spin_frame_id.setMaximum(frame_count-1)
    
    def update_hist(self):
        # keep old histogram levels
        hist_levels = self.hist.getLevels()

        # tie ImageItem to hist
        self.set_hist()

        # set old histogram levels
        self.hist.setLevels(*hist_levels)

    def set_hist(self):
        img = self.view_box.img
        if img is not None:
            self.hist.setImageItem(img)
            # set auto range on
            self.hist.autoHistogramRange()

    def save(self) -> Path:
        # directory and filename for saving annotation data in json format
        fname = QFileDialog.getSaveFileName(self, "Save file", str(self.last_folder), "Matlab files (*.mat)")
        if len(fname[0]) == 0: return None
        fname = Path(fname[0])
        self.last_folder = os.path.dirname(fname)
        return fname 

    def load_images(self) -> None:
        # get directory 
        fname = QFileDialog.getExistingDirectory(self, 'Select Folder', str(self.last_folder))
        if fname is not None and len(fname) > 0:
            fname = Path(fname)
            self.last_folder = os.path.dirname(fname)

            # set image from view_box
            self.view_box.load_images(fname)

            # Update frame id and show the first frame
            self.update_frame_id()
            self.navigate_to_idx(0)
    
    def load_annot(self) -> None:
        fname = QFileDialog.getOpenFileName(self, "Open file", str(self.last_folder), "Matlab files (*.mat)")
        if len(fname[0]) == 0:
            return None
        
        fname = Path(fname[0])
        data = sio.loadmat(fname)

        self.view_box.centroids = data["centroids"]
        
        params = data["params"]
        self.roi_dialog.set_params(params)

        self.view_box.set_rois(data["rois"])

    def save_annot(self) -> None:
        # ask user for filename
        fname = self.save()
        if fname is None:
            return None

        # gather data
        blob_log_params, box_size = self.roi_dialog.get_params()
        params = blob_log_params | {"box_size": box_size}
        mdict = {"params":params}

        centroids = self.view_box.centroids
        if centroids is not None:
            mdict["centroids"] = centroids

        surfaces = self.view_box.surface_reconstructions
        if surfaces is not None:
            mdict["surfaces"] = surfaces
        
        rois = self.view_box.convert_rois_to_numpy()
        if rois is not None:
            mdict["rois"] = rois

        # save data
        sio.savemat(fname, mdict)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()

    def run_blob(self):
        if self.roi_dialog.exec():
            blob_log_params, box_size = self.roi_dialog.get_params()
            self.view_box.run_blob(blob_log_params, box_size)
    
    def calc_centroids(self):
        # if no blobs were calculated, do that first
        if self.view_box.rois is None:
            self.run_blob()

        self.view_box.calc_centroids()

    def surf_rec(self):
        if self.view_box.centroids is None:
            self.calc_centroids()

        self.view_box.surf_rec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())