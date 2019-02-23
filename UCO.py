import sympy
from sympy.parsing.sympy_parser import parse_expr

import numpy

# input : function variables from user (as space separated string)
# function variables 'f_vars'
f_vars_string = 'x1 x2'
# input : function input from user (as normal python expression using the above variables)
f_string = 'x1**3 + 5*x1**2*x2 + 7*x1*x2**2 + 2*x2**3'


var_str_list = f_vars_string.split()
# print(type(var_str_list))
# var_list is a list of strings above

# No. of dimensions in function 'n'
n = len(var_str_list)

# Dynamic creation of SymPy Variables
ab = {}
k = 0
while k < n:
    key = var_str_list[k]
    value = sympy.var(var_str_list[k])
    ab[key] = value
    k += 1

# Parse the input function string as SymPy expression
f = parse_expr(f_string)


def gradient(function):
    print(n)
    g = [None]*n
    it = 0
    for x in ab.keys():
        g[it] = sympy.diff(function, x)
        it += 1
    return numpy.matrix(list(mat(g)))

def hessian(function):
    h = [None]*(n**2)
    it = 0
    for x in ab.keys():
        for y in ab.keys():
            h[it] = sympy.diff(function, x, y)
            it += 1
    return numpy.matrix(list(mat(h)))

def jacobian(function):
    return numpy.transpose(hessian(function))

def mat(list_object):
    if len(list_object) == n:
        li = zip(*[iter(list_object)] * 1)
    elif len(list_object) == n**2:
        li = zip(*[iter(list_object)] * n)
    return li

def evaluate(grid, x):
    #replacements = {key: value for (key, value) in zip}
    replacements = dict(zip(var_str_list, list(x)))
    evaluator = lambda t: t.evalf(subs=replacements)
    vfunc = numpy.vectorize(evaluator)
    return vfunc(grid)

def improve(old_x, grad, jaco):
    g_x0 = evaluate(grad, old_x)
    j_x0 = evaluate(jaco, old_x)
    new_x = old_x - numpy.dot(numpy.linalg.inv(j_x0), g_x0)
    return new_x

def minimize_NR(function, x0):
    x = []
    x.append(numpy.matrix(zip(*[iter(x0)]*1)))
    grad = gradient(f)
    jaco = jacobian(f)
    i = 0
    gap = 1
    new_x = 'inf'
    while(gap > 0.01):
        old_x = x[i]
        new_x = improve(old_x, grad, jaco)
        gap = new_x - old_x
        x.append(new_x)
        i += 1
    return new_x



grad = gradient(f)
hess = hessian(f)
print(f)
print(grad)
print(hess)
