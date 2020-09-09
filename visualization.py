import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import itertools

from cspline import CSpline, Vec2

class DraggableNodes(pg.GraphItem):
    def __init__(self, update_callback):
        self.update_callback = update_callback
        self.dragPoint = None
        self.dragOffset = None
        pg.GraphItem.__init__(self)

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            self.data['pos'] = np.array(self.data['pos'])
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        if 'pos' not in self.data:
            return
        control_points = self.data['pos']
        self.update_callback(control_points)


    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            index = pts[0].data()[0]
            self.dragOffset = self.data['pos'][index] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        index = self.dragPoint.data()[0]

        dx = (ev.pos() + self.dragOffset) - self.data['pos'][index]
        self.data['pos'][index] += dx
        for i in children(index):
            self.data['pos'][i] += dx

        self.updateGraph()
        ev.accept()

# Some functions to calculate the control point nodes.
def parent(i):
    return (i//3) * 3
def children(i):
    if parent(i) != i:
        return []
    return [i+1, i+2]

def main():

    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="spline_regression")
    win.resize(800,800)
    pg.setConfigOptions(antialias=True)
    view_box = win.addPlot()

    # Define the spline control points
    center_0 = Vec2(1, 2)
    center_1 = Vec2(6, 3)
    center_2 = Vec2(8, 3)
    control_points = [
        center_0,
        center_0 - Vec2(1,1),
        center_0 + Vec2(1,1),
        center_1,
        center_1 - Vec2(2,4),
        center_1 + Vec2(1,2),
        center_2,
        center_2 - Vec2(1,2),
        center_2 + Vec2(1,2),
    ]

    def make_control_point_line_data(control_points):
        return np.array(list(itertools.chain(*(
            [
                control_points[i],
                control_points[i+1],

                control_points[i],
                control_points[i+2],
            ] for i in range(0, len(control_points), 3)
        ))))
    control_point_lines = pg.PlotDataItem(make_control_point_line_data(control_points), connect='pairs')

    def make_curve_data(control_points):
        """
        Samples a spline defined by the control_points and turns it into
        many small line segments.
        Could optimize the number of lines by taking curvature and view_box ranges into account.
        """
        spline = CSpline(control_points)
        t_extrapolation = 2.5  # how far to draw the straight lines past the
        t_min = 0 - t_extrapolation
        t_max = len(control_points)//3 - 1  + t_extrapolation
        return np.array([
            spline.get_pos(t)
            for t in np.linspace(
                t_min,
                t_max,
                num=int(50*(t_max-t_min) + 1)
            )
        ])
    spline_curve = pg.PlotDataItem(pen=pg.mkPen(width=2, color='#ff0000ee'))

    intersect = pg.GraphItem()

    def on_update(control_points):
        control_point_lines.setData(make_control_point_line_data(control_points))
        spline_curve.setData(make_curve_data(control_points))

        intersect_x = 4
        spline = CSpline(control_points)
        # intersect_pos = spline.get_pos(spline.fast_intersect(intersect_x))
        pos = [
            spline.get_pos(spline.fast_intersect(intersect_x, approximation_iterations=i))
            for i in range(10)
        ]
        intersect.setData(
            pos=pos,
            symbol=['o'] * len(pos),
            size=10,
            symbolBrush=[pg.mkBrush(f'#FF0000{9-i}0') for i in range(len(pos))]
        )

    draggables = DraggableNodes(on_update)
    draggables.setData(
        pos=control_points,
        symbol=['o' if parent(i)==i else 's' for i in range(len(control_points)) ],
        size=30,
        symbolBrush=pg.mkBrush('#FFFFFF40')
    )



    view_box.addItem(draggables)
    view_box.addItem(control_point_lines)
    view_box.addItem(spline_curve)
    view_box.addItem(intersect)
    # view_box.setAutoPan(x=False, y=False)
    view_box.setRange(xRange=[-1,10], yRange=[-1,10])
    view_box.showGrid(x=True, y=True)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    main()
