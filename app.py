import sys
import os
from pathlib import Path
import scipy.io as sio

from gui import Ui_MainWindow
from viewbox import ViewBox

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
        self.view_box = ViewBox(lockAspect=True, invertY=True)
        self.hist = pg.HistogramLUTItem()

        self.img_view.addItem(self.view_box, row=0, col=0, rowspan=1, colspan=1)
        self.img_view.addItem(self.hist, row=0, col=1, rowspan=1, colspan=1)

        # action menu items: File
        self.action_load_images.triggered.connect(self.load_images)
        self.action_load_annot.triggered.connect(self.load_annot)
        self.action_save.triggered.connect(self.save_annot)

        # blob detection
        self.button_calc_roi.clicked.connect(self.run_blob)
        self.button_calc_centroids.clicked.connect(self.calc_centroids)

        # prev/next buttons
        self.button_prev.clicked.connect(self.prev)
        self.button_next.clicked.connect(self.next)

        # remember last position
        self.last_folder = os.getcwd()

    def prev(self):
        # update image and annotation data
        self.view_box.prev()

        # update histogram
        self.update_hist()

    def next(self):
        # update image and annotation data
        self.view_box.next()

        # update histogram
        self.update_hist()
    
    def update_hist(self):
        # keep old histogram levels
        hist_levels = self.hist.getLevels()

        # tie ImageItem to hist
        self.set_hist()

        # set old histogram levels
        self.hist.setLevels(*hist_levels)

    def set_hist(self):
        # tie new ImateItem
        img = self.view_box.get_image()
        self.hist.setImageItem(img)

        # set auto range on
        self.hist.autoHistogramRange()

    def load(self) -> Path:
        # get directory for loading content from a folder
        fname = QFileDialog.getExistingDirectory(self, 'Select Folder', str(self.last_folder))
        fname = Path(fname)
        self.last_folder = fname
        return fname

    def save(self) -> Path:
        # directory and filename for saving annotation data in json format
        fname = QFileDialog.getSaveFileName(self, "Save file", str(self.last_folder), "Matlab files (*.mat)")
        fname = Path(fname[0])
        self.last_folder = os.path.dirname(fname)
        return fname 

    def load_images(self) -> None:
        # get file directory
        fname = self.load()

        if fname is not None:
            # set image from view_box
            self.view_box.load_images(fname)

            # init histogram
            self.set_hist()
    
    def load_annot(self) -> None:
        fname = QFileDialog.getOpenFileName(self, "Open file", str(self.last_folder), "Matlab files (*.mat)")
        fname = Path(fname[0])
        data = sio.loadmat(fname)

        self.view_box.centroids = data["centroids"]
        
        params = data["params"]
        self.set_params(params)

        self.view_box.set_rois(data["rois"])

    def set_params(self, params):
        self.spinBox_threshold.setValue(float(params["threshold"]))
        self.spinBox_min_sigma.setValue(float(params["min_sigma"]))
        self.spinBox_max_sigma.setValue(float(params["max_sigma"]))
        self.spinBox_num_sigma.setValue(int(params["num_sigma"]))
        self.spinBox_overlap.setValue(float(params["overlap"]))
        self.spinBox_exclude_border.setValue(int(params["exclude_border"]))
        self.spinBox_box_size.setValue(float(params["box_size"]))

    def save_annot(self) -> None:
        # ask user for filename
        fname = self.save()

        # gather data
        blob_log_params, box_size = self.get_params()
        params = blob_log_params | {"box_size": box_size}
        centroids = self.view_box.centroids
        rois = self.view_box.convert_rois_to_numpy()

        # save data
        sio.savemat(fname, {"centroids":centroids, "rois":rois, "params":params})

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()
    
    def get_params(self):
        blob_log_params = {
            "threshold": self.spinBox_threshold.value(),
            "min_sigma": self.spinBox_min_sigma.value(),
            "max_sigma": self.spinBox_max_sigma.value(),
            "num_sigma": self.spinBox_num_sigma.value(),
            "overlap": self.spinBox_overlap.value(),
            "exclude_border": self.spinBox_exclude_border.value()
        }
        box_size = self.spinBox_box_size.value()

        return blob_log_params, box_size

    def run_blob(self):
        blob_log_params, box_size = self.get_params()
        self.view_box.run_blob(blob_log_params, box_size)
    
    def calc_centroids(self):
        # if no blobs were calculated, do that first
        if self.view_box.rois is None:
            self.run_blob()

        self.view_box.calc_centroids()        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())