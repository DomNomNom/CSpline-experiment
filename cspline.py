import bisect
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
        assert len(control_points) > 0
        assert len(control_points) % 3 == 0
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

    def fast_intersect(self, x: float, axis: int=0, approximation_iterations=5) -> float:
        """
        Returns a `t` such that `self.get_pos(t)[axis]` is very close to `x`.
        Increase approximation_iterations to get closer.

        This requires a couple of assumtions:
            All centers are sorted along the given axis.
            For all handles, previous_center < handle < next_center along the given axis.
            (fewer constraints for the first and last handle)

        Proof that this should return a unique solution:
        (RLBot discord) https://discordapp.com/channels/348658686962696195/535605770436345857/752888844814123048
        """
        projected = self.control_points[:,axis]
        i_hi_center = 3 * bisect.bisect_left(projected[::3], x)
        i_lo_center = i_hi_center - 3

        if i_lo_center < 0:
            # extrapolate on the lower end
            C = projected[i_hi_center+1]
            D = projected[i_hi_center]
            return (D - x) / (C - D)
        max_interp_t = len(self.control_points)//3 - 1
        if i_hi_center >= len(projected):
            # extrapolate on the lower end
            A = projected[i_lo_center]
            B = projected[i_lo_center+2]
            return max_interp_t + (x - A) / (B - A)
        assert projected[i_lo_center] <= x <= projected[i_hi_center]

        # Variable names according to this:
        # https://thumbs.gfycat.com/HarmoniousHarshAdmiralbutterfly-size_restricted.gif
        A = projected[i_lo_center]
        B = projected[i_lo_center+2]
        C = projected[i_hi_center+1]
        D = projected[i_hi_center]

        # Shift/scale everything such that A=0 and D=1.
        # This is going to make our equations simpler.
        B -= A
        C -= A
        D -= A
        x -= A
        B /= D
        C /= D
        x /= D
        # Remove them from the scope so we don't accidentally refer to them.
        del A
        del D

        # Find t via the bisection method.
        lo = 0.
        hi = 1.
        t = 0.5  # this t is purely between A and D, not the full spline.
        for _ in range(approximation_iterations//2):
            T = 1 - t
            t2 = t * t
            T2 = T * T
            got_x = (
                # fast 1D spline evaluation. (we dropped the A term because it's zero)
                (3 * t * T2) * B +
                (3 * T * t2) * C +
                (    t * t2)
            )
            if got_x < x:
                lo = t
            else:
                hi = t
            t = (lo+hi)/2

        # Refine t using Newton's method on the 1D spline (points are no longer vectors but numbers)
        # We also keep using the bounds found in bisection as tricky derivatives can lead us astray.
        for _ in range((approximation_iterations+1)//2):
            # f(t) / f'(t) was generated via sympy:
            # f = (3*t*(1-t)**2*B + 3*(1-t)*t**2*C + t**3) - x
            # print(simplify(f / diff(f, t)))
            T = t-1
            tt = t*t
            TT = T*T
            t -= (C*tt*T - B*t*TT - tt*t/3 + x/3)/(C*tt - 2*B*t*T - B*TT + 2*C*t*T - tt)
            t = max(min(t, hi), lo) # Sanity-check with the bounds from bisection.

        return i_lo_center / 3 + t

if __name__ == '__main__':
    import visualization
    visualization.main()
