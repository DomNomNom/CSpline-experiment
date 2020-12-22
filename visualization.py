import itertools

import numpy as np
import pyqtgraph as pg
from cma import CMAEvolutionStrategy
from pyqtgraph.Qt import QtCore, QtGui

from cspline import CSpline, Vec2
from ml_optimizer import CrossEntopyMethodStrategy
from ml_policies import ParameterCreatingPolicy


class TangentPolicy(ParameterCreatingPolicy):
    def act(self, control_points):
        """
        This function mutates given control_points
        and returns the same list.
        """
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.
        for i in range(0, len(control_points), 3):
            center = control_points[i]
            prev_handle = control_points[i + 1]
            next_handle = control_points[i + 2]
            # Next and prev handles are parameters
            for j, x in enumerate(prev_handle):
                prev_handle[j] = p(x, guess_std=abs(x - center[j]) / 3)
            for j, x in enumerate(next_handle):
                next_handle[j] = p(x, guess_std=abs(x - center[j]) / 3)

            # current handles should be on the appropriate side of the center
            epsilon = 0.0001
            prev_handle[0] = min(prev_handle[0], center[0] - epsilon)
            next_handle[0] = max(next_handle[0], center[0] + epsilon)
            # prev handle should be after last center.
            if i > 0:
                prev_handle[0] = max(prev_handle[0], control_points[i - 3][0] + epsilon)
            # next handle should be before next center.
            if i + 3 < len(control_points):
                next_handle[0] = min(next_handle[0], control_points[i + 3][0] - epsilon)

        return control_points


class DraggableNodes(pg.GraphItem):
    def __init__(self, update_callback, get_children=lambda i: []):
        self.update_callback = update_callback
        self.dragPoint = None
        self.dragOffset = None
        self.get_children = get_children
        pg.GraphItem.__init__(self)

    def setData(self, **kwds):
        self.text = kwds.pop("text", [])
        self.data = kwds
        if "pos" in self.data:
            self.data["pos"] = np.array(self.data["pos"])
            npts = self.data["pos"].shape[0]
            self.data["data"] = np.empty(npts, dtype=[("index", int)])
            self.data["data"]["index"] = np.arange(npts)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        if "pos" not in self.data:
            return
        control_points = self.data["pos"]
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
            self.dragOffset = self.data["pos"][index] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        index = self.dragPoint.data()[0]

        dx = (ev.pos() + self.dragOffset) - self.data["pos"][index]
        self.data["pos"][index] += dx
        for i in self.get_children(index):
            self.data["pos"][i] += dx

        self.updateGraph()
        ev.accept()


# Some functions to calculate the control point nodes.
def parent(i):
    return (i // 3) * 3


def is_parent(i):
    return parent(i) == i


def children(i):
    if parent(i) != i:
        return []
    return [i + 1, i + 2]


def main():

    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="spline_regression")
    win.resize(800, 800)
    pg.setConfigOptions(antialias=True)
    view_box = win.addPlot()

    # # Define the spline control points
    # center_0 = Vec2(1, 2)
    # center_1 = Vec2(6, 3)
    # center_2 = Vec2(8, 3)
    # control_points = [
    #     center_0,
    #     center_0 - Vec2(1,1),
    #     center_0 + Vec2(1,1),
    #     center_1,
    #     center_1 - Vec2(2,4),
    #     center_1 + Vec2(1,2),
    #     center_2,
    #     center_2 - Vec2(1,2),
    #     center_2 + Vec2(1,2),
    # ]

    import random

    samples = [
        # 0.1 * i*i
        20.0 * np.sin(i / 5.0)
        for i in range(20)
    ]
    samples += samples[:-1][::-1]
    random.seed(0)
    # samples = [ x + 1.1 * random.random() for x in samples]
    polyline = list(enumerate(samples))
    control_points = CSpline.fit_to_line(polyline, 0.1, corner_angle=0.0014)

    def make_control_point_line_data(control_points):
        return np.array(
            list(
                itertools.chain(
                    *(
                        [
                            control_points[i],
                            control_points[i + 1],
                            control_points[i],
                            control_points[i + 2],
                        ]
                        for i in range(0, len(control_points), 3)
                    )
                )
            )
        )

    control_point_lines = pg.PlotDataItem(
        make_control_point_line_data(control_points), connect="pairs"
    )

    def make_curve_data(control_points):
        """
        Samples a spline defined by the control_points and turns it into
        many small line segments.
        Could optimize the number of lines by taking curvature and view_box ranges into account.
        """
        spline = CSpline(control_points)
        t_extrapolation = 1  # how far to draw the straight lines past the
        t_min = 0 - t_extrapolation
        t_max = len(control_points) // 3 - 1 + t_extrapolation
        return np.array(
            [
                spline.get_pos(t)
                for t in np.linspace(t_min, t_max, num=int(50 * (t_max - t_min) + 1))
            ]
        )

    spline_curve = pg.PlotDataItem(pen=pg.mkPen(width=2, color="#ff0000ee"))

    intersect_x = None
    if intersect_x is not None:
        intersect = pg.GraphItem()

    def on_change_control_points(control_points):
        control_point_lines.setData(make_control_point_line_data(control_points))
        spline_curve.setData(make_curve_data(control_points))

        if intersect_x is not None:
            spline = CSpline(control_points)
            pos = [
                spline.get_pos(
                    spline.fast_intersect(intersect_x, approximation_iterations=i)
                )
                for i in range(5)
            ]
            intersect.setData(
                pos=pos,
                symbol=["o"] * len(pos),
                size=10,
                symbolBrush=[pg.mkBrush(f"#FF0000{9-i}0") for i in range(len(pos))],
            )

    control_point_draggables = DraggableNodes(
        on_change_control_points, get_children=children
    )
    control_point_draggables.setData(
        pos=control_points,
        symbol=["o" if parent(i) == i else "s" for i in range(len(control_points))],
        size=30,
        symbolBrush=pg.mkBrush("#FFFFFF40"),
    )

    def on_change_polyline(polyline):
        nonlocal control_points
        polyline = sorted(polyline, key=lambda pos: pos[0])
        control_points = CSpline.fit_to_line(polyline, 2, corner_angle=0.0001)
        control_points = np.array(control_points)
        control_point_draggables.setData(pos=control_points)
        on_change_control_points(control_points)

    on_change_polyline(polyline)
    polyline_draggables = DraggableNodes(on_change_polyline)
    polyline_draggables.setData(
        pos=polyline,
        symbol=["o" for i in range(len(polyline))],
        size=10,
        symbolBrush=pg.mkBrush("#AFFFAF80"),
    )

    # Use an evolutionary strategy to optimize stuff.
    bootstrap = TangentPolicy(parameters=None)
    bootstrap.act(control_points[:])
    es = CMAEvolutionStrategy(
        bootstrap.parameters, np.array(bootstrap.parameters_std).mean(), {}
    )
    # es = CrossEntopyMethodStrategy(
    #     bootstrap.parameters,
    #     bootstrap.parameters_std,
    #     elite_frac=0.1,
    # )

    def loss_function(control_points):
        spline = CSpline(control_points)
        loss = 0
        for (x, wantY) in polyline:
            t = spline.fast_intersect(x)
            gotY = spline.get_pos(t)[1]
            loss += (gotY - wantY) ** 2
        return loss

    iteration = 0
    toleranace = 100
    datapoints = []

    def step():
        nonlocal control_points
        nonlocal iteration
        nonlocal toleranace
        iteration += 1
        # Runs a step of the optimization algorithm
        solutions = es.ask(number=12)
        toleranace *= 0.98
        if toleranace < 0.0001:
            timer.stop()
        control_points = CSpline.fit_to_line(polyline, toleranace, corner_angle=0.0014)
        datapoints.append(
            (toleranace, loss_function(control_points), len(control_points) / 3.0)
        )
        control_point_draggables.setData(pos=control_points)
        on_change_control_points(control_points)

    step_button = QtGui.QPushButton("step_button")
    step_proxy = QtGui.QGraphicsProxyWidget()
    step_proxy.setWidget(step_button)
    view_box.addItem(step_proxy)
    p3 = win.addLayout(row=2, col=0)
    p3.addItem(step_proxy, row=1, col=1)
    step_button.clicked.connect(step)

    timer = QtCore.QTimer()
    timer.timeout.connect(step)
    # timer.start(10)

    view_box.addItem(control_point_draggables)
    view_box.addItem(polyline_draggables)
    view_box.addItem(control_point_lines)
    view_box.addItem(spline_curve)
    if intersect_x is not None:
        view_box.addItem(intersect)
    # view_box.setAutoPan(x=False, y=False)
    bot = np.min(control_points, axis=0) - 1
    top = np.max(control_points, axis=0) + 1
    view_box.setRange(xRange=[bot[0], top[0]], yRange=[bot[1], top[1]])
    view_box.showGrid(x=True, y=True)

    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()

    from xy_plot import plot_xy

    plot_xy(datapoints)


if __name__ == "__main__":
    main()
