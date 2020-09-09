from sympy import *

A,B,C,D,t,x = symbols('A B C D t x')

f = ((1-t)**3*A + 3*t*(1-t)**2*B + 3*(1-t)*t**2*C + t**3*D) - x
print(simplify(f / diff(f, t)))

# f = (3*t*(1-t)**2*B + 3*(1-t)*t**2*C + t**3) - x
# print(simplify(f / diff(f, t)))

# print('T=t-1; ' + str(simplify(f / diff(f, t)))
#     .replace('(t - 1)', 'T')
#     .replace('t**2', 't*t')
#     .replace('T**2', 'T*T')
# )

# print('T=1-t; ' + str((f / diff(f, t)))
#     .replace('(1 - t)', 'T')
#     .replace('t**2', 't*t')
#     .replace('T**2', 'T*T')
# )

# print(solve(Eq(x,  3*t*(1-t)**2*b + 3*(1-t)*t**2*c + t**3), t))
# print(simplify((3*t*(1-t)**2*b + 3*(1-t)*t**2*c + t**3) / diff(3*t*(1-t)**2*b + 3*(1-t)*t**2*c + t**3, t)))

