import numpy as np
from practica2 import predict,sigmoid

#   computeCostLogReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for logistic regression using regularization.
def computeCostLogReg(theta, X, y,lambda1):
    # Initialize some useful values
    m,p = X.shape
    # You need to return the following variable correctly 
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost. You may find useful numpy.log
#               and the sigmoid function.
#


# =============================================================

    return J

#   gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,p = X.shape

    # ====================== YOUR CODE HERE ======================

  
    # ============================================================

    return theta

#   computeCostLinReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for linear regression using regularization.
def computeCostLinReg(theta, X, y,lambda1):
    # Initialize some useful values
    m,p = X.shape
    # You need to return the following variable correctly 
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost. 
#

# =============================================================

    return J

#   gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,p = X.shape

    # ====================== YOUR CODE HERE ======================

  
    # ============================================================

    return theta


#   normalEqn(X,y) computes the closed-form solution to linear 
#   regression using the normal equations with regularization.
def normalEqnReg(X, y,lambda1):
    # Initialize some useful values
    m,p = X.shape
    # You need to return the following variable correctly 
    theta = np.zeros((p,1))
    

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression with regularization and put the result in theta.
#


# ============================================================

    return theta


