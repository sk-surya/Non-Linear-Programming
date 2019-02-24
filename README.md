# Non-Linear-Programming
Numerical methods in NLP

-------------------------------------------------------------------
Python Implementations of Algorithms in Unconstrained Optimization
-------------------------------------------------------------------
through Symbolic expressions
1. Newton-Raphson root finding algorithm
2. Classical Newton's method with stepsize = 1
3. Newton's method with stepsize <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{k}&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{k}&space;=&space;1" title="\alpha_{k} = 1" /></a>

  3.1 Line Search Methods
  
    3.1.1 Exact Method
    
    3.1.2 Bisection Method
    
    3.1.3 Golden Section Method
    
    3.1.4 Fibonacci Method
    
Work in Progress:
-----------------
Update:
-------
2/23/2019
---------
1. Included Levenberg-Marquardt modification (guaranteed descent) with Bisection Line Search.
2. To use LM modification, give lm_mod=True in function call minimize_NR(f, x0, lm_mod=True) in line 196. 
3. lm_mod=False does Classical Newton's method with stepsize=1.
4. Input : line 10, 13 and 18

2/22/2019
---------
1. Classical Newton's method is implemented and works for higher dimensional functions.
2. Needs exception handling when Jacobian becomes Singular matrix.
3. To test it, change the input on lines 9, 12 and 14
    3.1) line 9 : input function variables
    3.2) line 12 : input function
    3.3) line 14 : input starting point x0
    
 
