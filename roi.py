from roi_gui import Ui_Dialog

from PyQt5.QtWidgets import QDialog

class ROI(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)

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

    def set_params(self, params):
            self.spinBox_threshold.setValue(float(params["threshold"]))
            self.spinBox_min_sigma.setValue(float(params["min_sigma"]))
            self.spinBox_max_sigma.setValue(float(params["max_sigma"]))
            self.spinBox_num_sigma.setValue(int(params["num_sigma"]))
            self.spinBox_overlap.setValue(float(params["overlap"]))
            self.spinBox_exclude_border.setValue(int(params["exclude_border"]))
            self.spinBox_box_size.setValue(float(params["box_size"]))
