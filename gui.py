# gui.py
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel, QSpinBox,
    QHBoxLayout, QVBoxLayout, QSizePolicy
)
from PyQt6.QtCore import QTimer, QPointF
from PyQt6.QtGui import QFont

# for typing only (import local file)
from MountainRidgeOptimizer import MountainRidgeOptimizer  # type: ignore

# Moore neighborhood directions (not directly used here, but kept for reference)
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


class MainWindow(QMainWindow):
    """
    Main window that displays:
      - heightmap (numpy 2D array) as an image (colormap spans min->max)
      - agent positions overlayed as scatter points
      - controls for pause/resume and speed (ms per iteration)
      - labels showing current best among agents and the global best
      - hover readout (coords + height) when the mouse is over the map
      - user-adjustable, fixed-width sidebar (so label changes don't resize main area)
      - reset view button so user can re-fit after zoom/pan
    """

    def __init__(self, sim: MountainRidgeOptimizer, greed: float, social: float, chaos: float,
                 update_interval_ms: int = 20, sidebar_width: int = 325):
        super().__init__()
        self.greed, self.social, self.chaos = greed, social, chaos
        self.sim = sim
        self.setWindowTitle("Swarm Intelligence — Mountain Ridge")

        # Main container widget and layout (image left, controls right)
        container = QWidget()
        main_layout = QHBoxLayout()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Left: pyqtgraph GraphicsLayoutWidget containing the image
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win, stretch=1)

        # add a single ViewBox for the image and overlays
        self.view = self.win.addViewBox()
        self.view.setAspectLocked(True)  # lock aspect so pixels are square
        self.view.setMouseEnabled(x=True, y=True)

        # ImageItem to display the height map
        self.img_item = pg.ImageItem()
        # Force row-major axis order: index [row, col] -> x=col, y=row.
        # This prevents the transposed display you observed (width/height swapping).
        try:
            self.img_item.setOpts(axisOrder='row-major')
        except Exception:
            # older pyqtgraph versions may not support setOpts(axisOrder=...), ignore safely
            pass
        self.view.addItem(self.img_item)

        # ScatterPlotItem to display agents on top
        self.scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(width=1, color='k'),
                                          brush=pg.mkBrush(255, 0, 0, 200))
        self.view.addItem(self.scatter)

        # Right: controls panel
        controls = QWidget()
        controls_layout = QVBoxLayout()
        controls.setLayout(controls_layout)
        # Start with a fixed width that the user can change through the spinbox.
        self._sidebar_width_default = int(sidebar_width)
        controls.setFixedWidth(self._sidebar_width_default)
        # Prevent layout from letting this widget shrink/grow due to label changes
        controls.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(controls, stretch=0)

        # Pause/Resume button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.clicked.connect(self._toggle_pause)
        controls_layout.addWidget(self.pause_button)

        # Speed control (ms per iteration)
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed (ms/iter):")
        speed_layout.addWidget(speed_label)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 2000)
        self.speed_spin.setValue(update_interval_ms)
        self.speed_spin.setSingleStep(1)
        self.speed_spin.valueChanged.connect(self._change_speed)
        speed_layout.addWidget(self.speed_spin)
        controls_layout.addLayout(speed_layout)

        # Sidebar width control — lets user choose how wide the fixed sidebar is
        width_layout = QHBoxLayout()
        width_label = QLabel("Sidebar width (px):")
        width_layout.addWidget(width_label)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(120, 1000)
        self.width_spin.setValue(self._sidebar_width_default)
        self.width_spin.setSingleStep(10)
        self.width_spin.valueChanged.connect(lambda v: controls.setFixedWidth(int(v)))
        width_layout.addWidget(self.width_spin)
        controls_layout.addLayout(width_layout)

        # Reset view button (allow user to re-fit the whole map after zooming/panning)
        self.reset_view_btn = QPushButton("Reset view")
        self.reset_view_btn.clicked.connect(self._reset_view)
        controls_layout.addWidget(self.reset_view_btn)

        # Current best among agents
        mono = QFont("Courier New")
        self.current_best_label = QLabel("Current best: (—, —)  height=—")
        self.current_best_label.setFont(mono)
        controls_layout.addWidget(self.current_best_label)

        # Global best (min of the whole search space)
        self.global_best_label = QLabel("Global best: (—, —)  height=—")
        self.global_best_label.setFont(mono)
        controls_layout.addWidget(self.global_best_label)

        # Hover readout label
        self.hover_label = QLabel("Hover: (—, —)  height=—")
        self.hover_label.setFont(mono)
        controls_layout.addWidget(self.hover_label)

        # Stretch to push widgets to the top
        controls_layout.addStretch(1)

        # initial draw
        self._view_initialized = False  # do not auto-reset view after first initialization
        self._refresh_image()
        self._refresh_agents()
        self._refresh_best_labels()

        # Connect mouse move signal (scene coordinates -> mapSceneToView)
        self.view.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Timer to step simulation and update visualization
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(update_interval_ms)

        # show the window
        self.resize(1000, 700)
        self.show()

    # -------------------------
    # UI callbacks
    # -------------------------
    def _change_speed(self, val: int) -> None:
        """Adjust timer interval (ms per iteration)."""
        # Always set the timer interval even if not active, so it will use new speed when resumed
        self.timer.setInterval(val)

    def _toggle_pause(self, checked: bool) -> None:
        """Pause/resume the simulation timer."""
        if checked:
            self.pause_button.setText("Resume")
            self.timer.stop()
        else:
            self.pause_button.setText("Pause")
            self.timer.start(self.speed_spin.value())

    def _reset_view(self) -> None:
        """Re-fit the view to the full image (user-requested)."""
        img = np.array(self.sim.get_search_space(), copy=False)
        if img.ndim == 2:
            rows, cols = img.shape
            # set range and allow some padding
            self.view.setRange(QtCore.QRectF(0, 0, cols, rows), padding=0)
            self._view_initialized = True

    # -------------------------
    # Rendering & updates
    # -------------------------
    def _refresh_image(self) -> None:
        """Load the height map from sim and display it, normalizing min->max.

        IMPORTANT: this function does NOT force the view range after the first initialization.
        That allows the user to pan/zoom while the simulation runs without being reset.
        """
        img = np.array(self.sim.get_search_space(), copy=False)

        # if image is scalar or degenerate, handle safely
        if img.ndim != 2:
            raise ValueError("search space must be a 2D numpy array")

        # convert to float32 for stable min/max calculations
        imgf = img.astype(np.float32)

        rows, cols = imgf.shape

        # compute min / max and normalize to [0, 255] so colormap spans min->max
        vmin = float(np.nanmin(imgf))
        vmax = float(np.nanmax(imgf))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            norm = (imgf - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(imgf, dtype=np.float32)

        img8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)

        # IMPORTANT: do NOT transpose img8. We explicitly set the ImageItem to use
        # row-major axis order above so that img8[row, col] corresponds to x=col, y=row.
        self.img_item.resetTransform()
        self.img_item.setImage(img8, autoLevels=False)

        # set a colormap lookup table if available
        try:
            cmap = pg.colormap.get("viridis")
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            try:
                self.img_item.setLookupTable(lut)
            except Exception:
                pass
        except Exception:
            pass

        # set the view range only once (initial auto-fit) unless the user resets it manually
        if not getattr(self, "_view_initialized", False):
            self.view.setRange(QtCore.QRectF(0, 0, cols, rows), padding=0)
            self._view_initialized = True

        # store the latest global best info for labels (computed from original un-normalized image)
        self._last_global_vmin = vmin
        self._last_global_vmax = vmax
        # compute global minimum location and value (row, col)
        try:
            flat_idx = int(np.argmin(imgf))
            r, c = np.unravel_index(flat_idx, imgf.shape)
            self._global_best_coord = (int(r), int(c))
            self._global_best_value = float(imgf[r, c])
        except Exception:
            self._global_best_coord = None
            self._global_best_value = None

    def _refresh_agents(self) -> None:
        """Query sim for agent positions and update scatter points."""
        positions = list(self.sim.get_swarm_positions())
        if not positions:
            self.scatter.setData([], [])
            return

        spots_x = []
        spots_y = []
        for pos in positions:
            # defensive: pos might be sequences of floats or numpy scalars
            try:
                r = float(pos[0])
                c = float(pos[1])
            except Exception:
                r = float(pos[1])
                c = float(pos[0])
            x = c
            y = r
            spots_x.append(x)
            spots_y.append(y)

        # set scatter data — allow per-point symbol/brush if desired
        self.scatter.setData(spots_x, spots_y)

    def _refresh_best_labels(self) -> None:
        """Update labels for current best (from sim.get_best()) and the global best (from image)."""
        # current best among agents
        try:
            best = self.sim.get_best()
            if best is None:
                self.current_best_label.setText("Current best: (—, —)  height=—")
            else:
                coord, val = best
                # make sure coord is shown as ints
                try:
                    coord_t = (int(coord[0]), int(coord[1]))
                except Exception:
                    coord_t = tuple(int(x) for x in coord)
                # round height to fixed decimals so label width is stable; format to 6 decimals
                self.current_best_label.setText(
                    f"Current best: {coord_t}  height={float(val):.6f}"
                )
        except Exception:
            self.current_best_label.setText("Current best: (error)")

        # global best (computed in _refresh_image)
        if getattr(self, "_global_best_coord", None) is not None:
            coord = self._global_best_coord
            val = self._global_best_value
            self.global_best_label.setText(f"Global best: {coord}  height={float(val):.6f}")
        else:
            self.global_best_label.setText("Global best: (—, —)  height=—")

    def _on_mouse_moved(self, scene_pos: QPointF) -> None:
        """
        Mouse moved handler (receives a QPointF in scene coordinates).
        Map to view coordinates, then to image indices (row, col).
        Display the coordinates and height under the cursor in hover_label.
        """
        try:
            # map scene point to view coordinates (image coordinate system)
            view_pt = self.view.mapSceneToView(scene_pos)
            x = float(view_pt.x())
            y = float(view_pt.y())

            # image indices: row = int(y), col = int(x)
            col = int(np.floor(x))
            row = int(np.floor(y))

            # get current image shape
            img = np.array(self.sim.get_search_space(), copy=False)
            if img.ndim != 2:
                self.hover_label.setText("Hover: (—, —)  height=—")
                return
            rows, cols = img.shape

            if 0 <= row < rows and 0 <= col < cols:
                height = float(img[row, col])
                self.hover_label.setText(f"Hover: ({row}, {col})  height={height:.6f}")
            else:
                self.hover_label.setText("Hover: (out of bounds)")
        except Exception:
            # ignore mapping errors
            self.hover_label.setText("Hover: (—, —)  height=—")

    def _on_timer(self) -> None:
        """Called periodically: step the sim and update overlays & labels."""
        try:
            self.sim.step(self.greed, self.social, self.chaos)
        except Exception as e:
            # don't crash the GUI — stop timer and print error
            self.timer.stop()
            print("Simulation step failed:", e)
            return

        # update agent positions (fast)
        self._refresh_agents()

        # update current best and global best display
        self._refresh_best_labels()

        # If the search space itself is dynamic, refresh image too.
        # For static heightmaps this is mostly a no-op / cheap check.
        # We'll refresh it occasionally (every 20 steps) to avoid needless redraws.
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        self._step_counter += 1
        if self._step_counter % 20 == 0:
            # only update the image data (colors) — this will not change the view if the user zoomed/panned
            self._refresh_image()


def run(sim: MountainRidgeOptimizer, greed: float = 1.0, social: float = 1.0, chaos: float = 0.25,
        update_in_ms: int = 20) -> None:
    """
    Entrypoint for main.py:
      mkQApp("SI simple demo")
      main_window = MainWindow(sim, greed, social, chaos, update_in_ms)
      pg.exec()
    """
    pg.mkQApp("SI simple demo")
    main_window = MainWindow(sim, greed, social, chaos, update_in_ms)
    pg.exec()