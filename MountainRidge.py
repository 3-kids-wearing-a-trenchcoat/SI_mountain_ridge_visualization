from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from generate_height_map import combine_landscape

class MountainRidge:
    def __init__(self, w, h, seed, neighborhood_radius: int = 1):
        self.mat = combine_landscape((w, h), seed)
        self.radius = neighborhood_radius
        self.w, self.h = w, h

    def get_height_map(self) -> NDArray:
        return self.mat

    def get_neighborhood(self, i, j):
        return self.mat[max(0, i - self.radius): min(self.w, i + self.radius + 1),
                        max(0, j - self.radius): min(self.h, j + self.radius + 1)]

    def get_height(self, i, j):
        return self.mat[i,j]

    def shape(self):
        return self.w, self.h


# Simple graphic test to make sure the generated map makes sense
if __name__ == '__main__':
    import PyQt6
    import pyqtgraph as pg
    from PyQt6 import QtWidgets
    import sys

    h, w, seed = 512, 512, 445
    space = MountainRidge(w, h, seed)

    def show_heatmap(matrix):
        app = QtWidgets.QApplication(sys.argv)

        win = pg.GraphicsLayoutWidget(title="2D Heatmap")
        win.resize(800, 800)
        win.show()

        view = win.addViewBox()
        view.setAspectLocked(True)

        img = pg.ImageItem(matrix)
        view.addItem(img)

        # Flip vertically so (0,0) is bottom-left like math coords
        img.setTransform(pg.QtGui.QTransform().scale(1, -1))
        img.setPos(0, matrix.shape[1])

        # Apply colormap
        colormap = pg.colormap.get("viridis")
        img.setColorMap(colormap)

        # Auto-range view
        view.autoRange()

        sys.exit(app.exec())

    matrix = space.get_height_map()
    # matrix = np.full((h, w), 0)
    # matrix[0:10] = np.full_like(matrix[0:10], 100)

    show_heatmap(matrix)