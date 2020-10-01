# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with
the left/right mouse buttons. Right click on any plot to show a context menu.
"""



from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

def plot_xy(datapoints):
    #QtGui.QApplication.setGraphicsSystem('raster')
    app = QtGui.QApplication([])
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)

    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle('pyqtgraph example: Plotting')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    datapoints = np.array(datapoints)
    w1 = pg.PlotWidget(title="accuracy over tolerance")
    p1 = w1.plot(x=datapoints[:,0], y=datapoints[:,1])
    proxy1 = QtGui.QGraphicsProxyWidget()
    proxy1.setWidget(w1)
    win.addItem(proxy1, row=0, col=0)

    w2 = pg.PlotWidget(title="control_points over tolerance")
    p2 = w2.plot(x=datapoints[:,0], y=datapoints[:,2])
    proxy2 = QtGui.QGraphicsProxyWidget()
    proxy2.setWidget(w2)
    win.nextRow()
    win.addItem(proxy2, row=1, col=0)


    bang_buck = datapoints[:,1] * datapoints[:,2]
    w3 = pg.PlotWidget(title="control_points over tolerance")
    p3 = w3.plot(x=datapoints[:,0], y=bang_buck)
    proxy3 = QtGui.QGraphicsProxyWidget()
    proxy3.setWidget(w3)
    win.nextRow()
    win.addItem(proxy3, row=2, col=0)


    syncedPlots = [w1, w2, w3] # put as many plots as you wish
    def onSigRangeChanged(r):
        for g in syncedPlots:
            if g != r:
                g.blockSignals(True)
                g.setRange(xRange=r.getAxis('bottom').range)
                g.blockSignals(False)
    for g in syncedPlots:
         g.sigRangeChanged.connect(onSigRangeChanged)

    QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    plot_xy([
        (1,1,1),
        (2,2,2),
        (3,3,3),
        (4,4,4),
    ])
