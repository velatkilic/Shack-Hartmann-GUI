from PyQt5.QtWidgets import QDialog, QHBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SurfaceFigure(FigureCanvas): # Class for 3D window
    def __init__(self, figsize=(10,10)):
        self.fig = plt.figure(figsize=figsize)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(projection='3d')

    def plot_surface(self, x, y, z):
        self.axes.clear()
        self.axes.plot_surface(x, y, z)
        self.draw()

class SurfaceFigureDiag(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig = SurfaceFigure()

        layout = QHBoxLayout()
        layout.addWidget(self.fig)
        self.setLayout(layout)