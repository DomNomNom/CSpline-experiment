import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import itertools

Vec = np.array
def Vec2(x, y):
    return np.array([x,y], dtype=float)
def Vec3(x, y, z):
    return np.array([x,y,z], dtype=float)

def lerp(a,b,t):
    """ Linear interpolation between from @a to @b as @t goes between 0 an 1. """
    return (1-t)*a + t*b

class CSpline:
    """
    A Cubic Bézier Spline where defined by control points which define
       "center"s - positions that the curve passes through as well as
       "handle"s - positions relative to a center which defines the tangent in that direction.
    """
    def __init__(self, control_points: np.array):
        '''
        control_points has this structure:
        [
            center_0,
            handle_before_center_0,
            handle_after_center_0,
            center_1,
            handle_before_center_1,
            handle_after_center_1,
            center_2,
            handle_before_center_2,
            handle_after_center_2,
            ...
        ]
        '''
        self.control_points = control_points

    # These methods return indecies into control_points.
    def i_center_before(self, t:float):
        return int(t) * 3
    def i_center_after(self, t:float):
        return self.i_center_before(t) + 3
    def i_handle_before(self, t:float):
        return self.i_center_before(t) + 2
    def i_handle_after(self, t:float):
        return self.i_center_before(t) + 3+1

    def get_pos(self, t: float) -> Vec:
        # Handle things outside the normal range by extrapolating.
        max_interp_t = len(self.control_points)//3 - 1
        if t <= 0: # extrapolate at start
            A = self.control_points[0]
            B = self.control_points[1]
            return lerp(A, B, -t)
        elif t >= max_interp_t: # Extrapolate at end
            A = self.control_points[-3]
            B = self.control_points[-1]
            return lerp(A, B, t - max_interp_t)  # linear extrapolation


        # Variable names according to this:
        # https://thumbs.gfycat.com/HarmoniousHarshAdmiralbutterfly-size_restricted.gif
        A = self.control_points[self.i_center_before(t)]
        B = self.control_points[self.i_handle_before(t)]
        C = self.control_points[self.i_handle_after(t)]
        D = self.control_points[self.i_center_after(t)]
        t = t % 1.0  # note, we have both `t` and `T` in this scope.
        # P = lerp(A, B, t)
        # Q = lerp(B, C, t)
        # R = lerp(C, D, t)
        # S = lerp(P, Q, t)
        # T = lerp(Q, R, t)  # `T` is a position, `t` is the interpolant
        # O = lerp(S, T, t)
        # return O

        # Reducing the number of `lerp`s by using basis polynomials.
        # https://en.wikipedia.org/wiki/Bézier_curve
        T = 1 - t  # note: using this variable name differently to above.
        t2 = t * t
        T2 = T * T
        return (
            (    T * T2) * A +
            (3 * t * T2) * B +
            (3 * T * t2) * C +
            (    t * t2) * D
        )

    def fast_intersect(self, x: float, axis: int=0) -> Vec:
        """
        Returns a t such that self.get_pos(t)[axis] == x
        This requires a couple of assumtions:
            all centers are sorted along the given axis.
            for all handles, previous_center <= handle <= next_center along the given axis.
        """
        pass # TODO

if __name__ == '__main__':
    import visualization
    visualization.main()
