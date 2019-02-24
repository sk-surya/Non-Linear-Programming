import sympy
import itertools
from sympy.parsing.sympy_parser import parse_expr
import numpy
from scipy.sparse.linalg import eigs
from sympy.abc import theta

# input : function variables from user (as space separated string)
# function variables 'f_vars'
f_vars_string = 'x1 x2'
# input : function input from user (as normal python expression using the above variables)
#f_string = '(x-3)*((x-5)**2)*(3*x-16)'
f_string = 'x1**3 + 5*x1**2*x2 + 7*x1*x2**2 + 2*x2**3'
# f_string = 'x1**2 + 2*x2**2 - 3*x1 - 4*x2'
# Parse the input function string as SymPy expression
f = parse_expr(f_string)
# input : intial point x0
x0 = [2,1]
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


def gradient(function):
    # print(n)
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
    hess = numpy.matrix(list(mat(h)))
    return hess

def jacobian(function):
    return numpy.transpose(hessian(function))

def mat(list_object):
    if len(list_object) == n:
        li = zip(*[iter(list_object)] * 1)
    elif len(list_object) == n**2:
        li = zip(*[iter(list_object)] * n)
    return li


def flatten(matrix):
    x = list(itertools.chain.from_iterable(matrix.tolist()))
    return x

def evaluate(grid, x):
    #replacements = {key: value for (key, value) in zip}
    xz = flatten(x)
    replacements = dict(zip(var_str_list, xz))
    evaluator = lambda t: float(t.evalf(subs=replacements))
    vfunc = numpy.vectorize(evaluator)
    out_grid = vfunc(grid)
    return out_grid

# We know that hessian is real symmetric, hence not checking it before doing Cholesky decomposition
def is_pos_def(A):
    try:
        numpy.linalg.cholesky(A)
        return True
    except numpy.linalg.LinAlgError:
        return False


def modifier(A):
    y = eigs(A,1, which='SR', return_eigenvectors=False)
    mu = abs(y)+1
    return mu


def find_ab(fn):
    a = 0
    i = 1
    if fn.evalf(subs={theta: a}) > 0.001:
        while True:
            if fn.evalf(subs={theta: i}) < 0.001:
                b = i
                break
            elif fn.evalf(subs={theta: -i}) < 0.001:
                b = -i
                break
            #print(i)
            i += 1
    elif fn.evalf(subs={theta: a}) < 0.001:
        while True:
            if fn.evalf(subs={theta: i}) > -0.009:
                b = i
                break
            elif fn.evalf(subs={theta: -i}) > -0.009:
                b = -i
                break
            #print(i)
            i += 1
    else:
        b = 0
    return a, b


def Bisect(fn, a,b):
    while abs(a-b) >= 0.0001:
        f_a = fn.evalf(subs={theta: a})
        m = (a+b)/2
        f_m = fn.evalf(subs={theta: m})
        if f_a*f_m < 0:
            a, b = Bisect(fn, a, m)
        else:
            a, b = Bisect(fn, m, b)
    return a, b

def LineSearch(x_k, p_k, method):
    z_m = x_k + theta*(p_k)
    # print(z_m)
    z = flatten(z_m)
    replacements = dict(zip(var_str_list, z))
    f_theta = f.subs(replacements)
    # print(sympy.expand(f_theta))
    f_theta_1 = sympy.diff(f_theta)
    # print(sympy.expand(f_theta_1))
    a, b = find_ab(f_theta_1)
    # print('Initial Uncertainty interval : ', a, b)
    a, b = Bisect(f_theta_1, a, b)
    return (a+b)/2


# test
# print(LineSearch(numpy.matrix([[0], [3]]), numpy.matrix([[1], [-1]]), method='Bisection'))


def improve(old_x, grad, jaco, lm_mod):
    g_x0 = evaluate(grad, old_x)
    j_x0 = evaluate(jaco, old_x)
    alpha = 1
    if n == 1:
        j_inv = 1/j_x0.item(0)
        g = g_x0.item(0)
        p_k = -1 * (numpy.dot(j_inv, g))
    else:
        g = g_x0
        j_inv = numpy.linalg.inv(j_x0)
        p_k = -1 * (numpy.dot(j_inv, g))
        if lm_mod == True:
            if is_pos_def(j_x0) == False:
                mu = modifier(j_x0)
                j_x0 = j_x0 + numpy.dot(mu, numpy.identity(n))
                alpha = LineSearch(old_x, p_k, method='Bisection')
    new_x = old_x + alpha*p_k
    return new_x


x = []


def minimize_NR(function, x0, lm_mod=False):
    x.append(numpy.matrix(list(zip(*[iter(x0)]*1))))
    grad = gradient(f)
    jaco = jacobian(f)
    i = 0
    gap = 1
    new_x = 'inf'
    while(gap > 0.00001):
        old_x = x[i]
        new_x = improve(old_x, grad, jaco, lm_mod)
        gap = numpy.linalg.norm(new_x - old_x)
        x.append(new_x)
        i += 1
    return new_x


# Uncomment the following line if you want to find the roots of the given function (for 1D functions)
#f = sympy.integrate(f)
print(sympy.expand(f))

ans = minimize_NR(f, x0, lm_mod=True)
print(len(x), 'Iterations.')
print('ans is: ', ans)
print('gradient at ', ans, 'is ', evaluate(gradient(f), ans))
