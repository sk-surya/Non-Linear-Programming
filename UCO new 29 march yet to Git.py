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
# f_string = '(x-3)*((x-5)**2)*s(3*x-16)'
#f_string = ".0107*y**.62"
# function below is complicated, line search needs decimal numbers
#f_string = 'x1**3 + 5*x1**2*x2 + 7*x1*x2**2 + 2*x2**3'
#f _string = '2*sin(x) - (x**2)/10'
#f_string = 'x1**5 + 2*x2**2 - 3*x1 - 4*x2'

#f_string = '100*(x2-x1**2)**2 + (1-x1)**2'
# SDM at 7447 iterations from [-1.2, 1] with accuracy 10e-8
# Iteration  7447 [[1.0072049 ]
#  [1.01447784]] 0.008544791650008493

f_string = '3*x1**2 + 5*x2**2 -3*x1*x2 -5*x1 - 8*x2 + 8'

# Parse the input function string as SymPy expression
f = parse_expr(f_string)
# input : intial point x0
x0 = [1.33807846, 1.79111291]
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
    #print(x)
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
    mu = abs(y) + 0.1
    return mu


def find_ab(fn):
    a = 0
    b = 0
    i = 0
    #print(fn)
    if fn.evalf(subs={theta: a}) > 0.00000001:
        while True:
            if fn.evalf(subs={theta: i}) < 0.00000001:
                b = i
                break
            #print(i)
            i += 1
    elif fn.evalf(subs={theta: a}) < 0.00000001:
        while True:
            if fn.evalf(subs={theta: i}) > -0.00000009:
                b = i
                break
            #print(i)
            i += 1
    else:
        b = 0
    return a, b


def Bisect(fn, a,b):
    #print('Interval : [', a, ',', b, ']')
    while abs(a-b) >= 0.00000001:
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
    #print(sympy.expand(f_theta_1))
    a, b = find_ab(f_theta_1)
    # print('Initial Uncertainty interval : ', a, b)
    a, b = Bisect(f_theta_1, a, b)
    return (a+b)/2


# test
# print(LineSearch(numpy.matrix([[0], [3]]), numpy.matrix([[1], [-1]]), method='Bisection'))


def improve(old_x, grad, jaco, method, line_search):
    terminate = False
    g_x0 = evaluate(grad, old_x)

    if method != 'SDM':
        j_x0 = evaluate(jaco, old_x)
    else:   # only for SDM (Note: jaco will have Identity in this case)
        j_x0 = jaco

    # Handling 1-D and n-D cases
    if n == 1:
        j_inv = 1/j_x0.item(0)
        g = g_x0.item(0)
        p_k = -1 * (numpy.dot(j_inv, g))
    else:
        g = g_x0
        j_inv = numpy.linalg.inv(j_x0)
        p_k = -1 * (numpy.dot(j_inv, g))

        # LM Modification
        if method == 'lm_mod':
            if is_pos_def(j_x0) == False:
                mu = modifier(j_x0)
                j_x0 = j_x0 + numpy.dot(mu, numpy.identity(n))

    if line_search == False:
        alpha = 1
    else:
        alpha = LineSearch(old_x, p_k, method='Bisection')

    if alpha < 0.00000001:  # if alpha becomes zero, you don't move anymore
        print('alpha is almost zero, implying we are at optimum. True alpha is ', alpha)

        # Wait!!!! alpha cannot be zero (Check your concepts, find_ab, Bisect)
        # alpha will be zero, if you are at optimum as f'(alpha) = 0 (But it doesn't look like we are at optimum)
        # Wait, are you sure you are not at optimum? (Check again)
        #alpha = ??  # what to do now? (Look for Practical Line Search algorithms?)
        # Inference : This route of termination might result in lesser accuracy. (Check why)

        terminate = True

    #print('step size: ', alpha)
    new_x = old_x + alpha*p_k
    return new_x, terminate


x = []


# methods : null/CN - Classical Newton, lm_mod - CN with LM mod, SDM - Steepest Descent Method
def minimize_NR(function, x0, method='Newton', line_search=True):
    x.append(numpy.matrix(list(zip(*[iter(x0)]*1))))
    grad = gradient(f)
    if method == 'lm_mod' or method == 'Newton':
        jaco = jacobian(f)
    elif method == 'SDM':
        jaco = numpy.identity(n)
    i = 0
    gap = 1
    new_x = 'inf'
    terminate = False
    while(gap > 0.00000001 and i < 10000 and terminate == False):
        old_x = x[i]
        new_x, terminate = improve(old_x, grad, jaco, method, line_search)
        #gap = numpy.linalg.norm(new_x - old_x)
        grad_x = evaluate(grad, new_x)
        gap = numpy.linalg.norm(grad_x)
        #print('gap : ', gap)
        x.append(new_x)
        i += 1
        print('Iteration ', i-1, new_x, gap)
    return new_x, gap


# Uncomment the following line if you want to find the roots of the given function (for 1D functions)
#f = sympy.integrate(f)
print(sympy.expand(f))

ans, gap = minimize_NR(f, x0, method='Newton', line_search=False)

print(len(x), 'Iterations.')
print('ans is: ', ans)
print('gradient at ', ans, 'is ', evaluate(gradient(f), ans))
print('gradient gap is ', gap)
