import pickle
import time
from dataclasses import dataclass, field
from math import tau
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from policy_vis_model import *

batch_size = 20
row_count = 2


def main():
    app = QtGui.QApplication([])

    class CloseOnKeyPressWindow(pg.GraphicsLayoutWidget):
        def keyPressEvent(self, ev):
            self.scene().keyPressEvent(ev)
            if ev.key() == 16777216:  # ESC
                self.close()

    # win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
    win = CloseOnKeyPressWindow(show=True, title="Basic plotting examples")
    win.resize(1200, 2 * 600 / 5)
    win.setWindowTitle("pyqtgraph example: Plotting")

    pg.setConfigOptions(antialias=True)

    image_items = []

    for i in range(batch_size):
        plot = win.addPlot(row=i % row_count, col=i // row_count)
        plot.hideAxis("left")
        plot.hideAxis("bottom")

        ii = pg.ImageItem()
        plot.addItem(ii)
        image_items.append(ii)

    # with open(model_filepath, "wb") as f:
    #     vis_model = VisDataModel(
    #         [
    #             VisPolicy(pixels=np.random.normal(scale=i, size=(200, 200)))
    #             for i in range(batch_size)
    #         ]
    #     )
    #     pickle.dump(vis_model, f)

    last_mtime = 0

    def update():
        nonlocal last_mtime
        try:
            mtime = model_filepath.stat().st_mtime
            if mtime == last_mtime:
                return

            with open(model_filepath, "rb") as f:
                vis_model = pickle.load(f)
            vis_model.policies.sort(key=lambda policy: -policy.reward)
            for ii, vis_policy in zip(image_items, vis_model.policies):
                ii.setImage(vis_policy.pixels)
        except Exception as err:
            print("oh no", err)
        else:
            last_mtime = mtime

    update()
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(60)

    QtGui.QApplication.instance().exec_()


if __name__ == "__main__":
    main()
