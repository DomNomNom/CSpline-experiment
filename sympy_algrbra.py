from sympy import *

A,B,C,D,t,x = symbols('A B C D t x')

print(simplify(
    (sin(t)/(1+cos(t)))
    - ((1-cos(t))/sin(t))
))

# f = ((1-t)**3*A + 3*t*(1-t)**2*B + 3*(1-t)*t**2*C + t**3*D) - x
# f = 3*t*(1-t)**2*B + 3*(1-t)*t**2*C + t**3 - x
# print(solve(f, t)[1])
# print(simplify(solve(f, t)[1]))
# print(simplify(f / diff(f, t)))

# f = x - (3*t*(1-t)**2*B + 3*(1-t)*t**2*C + t**3)
# f = ((1-t)**3*A + 3*t*(1-t)**2*B + 3*(1-t)*t**2*C + t**3*D) - x
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

# print(solve(Eq(x,  3*t*(1-t)**2*b + 3*(1-t)*t**2*c + t**3), t))


# [-(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)/(3*(27*(-A + x)/(2*(A - 3*B + 3*C - D)) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(2*(A - 3*B + 3*C - D)**2) + sqrt(-4*(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)**3 + (27*(-A + x)/(A - 3*B + 3*C - D) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(A - 3*B + 3*C - D)**2 + 2*(-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**2)/2 + (-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**(1/3)) - (-3*A + 6*B - 3*C)/(3*(A - 3*B + 3*C - D)) - (27*(-A + x)/(2*(A - 3*B + 3*C - D)) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(2*(A - 3*B + 3*C - D)**2) + sqrt(-4*(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)**3 + (27*(-A + x)/(A - 3*B + 3*C - D) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(A - 3*B + 3*C - D)**2 + 2*(-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**2)/2 + (-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**(1/3)/3,
#  -(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)/(3*(-1/2 - sqrt(3)*I/2)*(27*(-A + x)/(2*(A - 3*B + 3*C - D)) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(2*(A - 3*B + 3*C - D)**2) + sqrt(-4*(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)**3 + (27*(-A + x)/(A - 3*B + 3*C - D) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(A - 3*B + 3*C - D)**2 + 2*(-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**2)/2 + (-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**(1/3)) - (-3*A + 6*B - 3*C)/(3*(A - 3*B + 3*C - D)) - (-1/2 - sqrt(3)*I/2)*(27*(-A + x)/(2*(A - 3*B + 3*C - D)) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(2*(A - 3*B + 3*C - D)**2) + sqrt(-4*(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)**3 + (27*(-A + x)/(A - 3*B + 3*C - D) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(A - 3*B + 3*C - D)**2 + 2*(-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**2)/2 + (-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**(1/3)/3,
#  -(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)/(3*(-1/2 + sqrt(3)*I/2)*(27*(-A + x)/(2*(A - 3*B + 3*C - D)) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(2*(A - 3*B + 3*C - D)**2) + sqrt(-4*(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)**3 + (27*(-A + x)/(A - 3*B + 3*C - D) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(A - 3*B + 3*C - D)**2 + 2*(-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**2)/2 + (-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**(1/3)) - (-3*A + 6*B - 3*C)/(3*(A - 3*B + 3*C - D)) - (-1/2 + sqrt(3)*I/2)*(27*(-A + x)/(2*(A - 3*B + 3*C - D)) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(2*(A - 3*B + 3*C - D)**2) + sqrt(-4*(-3*(3*A - 3*B)/(A - 3*B + 3*C - D) + (-3*A + 6*B - 3*C)**2/(A - 3*B + 3*C - D)**2)**3 + (27*(-A + x)/(A - 3*B + 3*C - D) - 9*(3*A - 3*B)*(-3*A + 6*B - 3*C)/(A - 3*B + 3*C - D)**2 + 2*(-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**2)/2 + (-3*A + 6*B - 3*C)**3/(A - 3*B + 3*C - D)**3)**(1/3)/3
#  ]
