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
    h = sigmoid(X@theta)
    dota = -y.T @ np.log(h)
    dotb = (1 - y).T @ np.log(1 - h)
    sinreg = (dota - dotb) / m
    J = sinreg + lambda1/(2*m) * (theta.T @ theta - theta[0]**2)




# =============================================================

    return J

#   gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,p = X.shape

    # ====================== YOUR CODE HERE ======================

    for i in range(iterations):

        h = sigmoid(X @ theta)
        theta = theta - alpha * 1/m * (X.T @ (h-y) + lambda1 * theta)

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
    J = (1/(2*m))* ((X @ theta - y).T @ (X @ theta - y) + lambda1 * (theta.T @ theta - theta[0]**2))

# =============================================================

    return J

#   gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,p = X.shape

    # ====================== YOUR CODE HERE ======================

    #theta = (1/(m)) * (X.T @ (X @ theta - y)  + lambda1 * theta)
    #theta = theta - (aux * lambda1 / m)

    for i in range(iterations):
        aux = theta[0]
        H =(X @ theta) - y
        theta = theta - alpha / m * (X.T @ H + theta * lambda1)
        theta[0] += alpha / m * lambda1 * aux
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
    I = np.eye(p+1)
    I[0,0] = 0

    theta = (X.T @ X + lambda1 * I).I @ X.T @ y

# ============================================================

    return theta


