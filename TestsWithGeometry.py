from scipy.spatial.transform import Rotation as R

import numpy as np


def homogenize(X):
    xl = np.zeros(np.shape(X))
    i = 0
    for x1 in X:
        xl[i] = x1/x1[-1]
        i = i + 1
    return xl


def project_points(P, X):
    xc = np.zeros([2, 3])
    xc[0, 0] = (P[0, 0] * X[0, 0] + P[0, 1] * X[0, 1] + P[0, 2] * X[0, 2] + P[0, 3]) / \
            (P[2, 0] * X[0, 0] + P[2, 1] * X[0, 1] + P[2, 2] * X[0, 2] + P[2, 3])

    xc[0, 1] = (P[1, 0] * X[0, 0] + P[1, 1] * X[0, 1] + P[1, 2] * X[0, 2] + P[1, 3]) / \
            (P[2, 0] * X[0, 0] + P[2, 1] * X[0, 1] + P[2, 2] * X[0, 2] + P[2, 3])

    xc[0, 2] = 1

    return xc


def linear_system(P, X, x):
    A = np.zeros([7, 6])
    b = np.zeros([7, 1])
    A[0, 0] = P[0, 0] - x[0, 0]*P[2, 0]
    A[0, 1] = P[0, 1] - x[0, 0]*P[2, 1]
    A[0, 2] = P[0, 2] - x[0, 0]*P[2, 2]
    b[0] = x[0, 0]*P[2, 3] - P[0, 3]

    A[1, 0] = P[1, 0] - x[0, 1]*P[2, 0]
    A[1, 1] = P[1, 1] - x[0, 1]*P[2, 1]
    A[1, 2] = P[1, 2] - x[0, 1]*P[2, 2]
    b[1] = x[0, 1]*P[2, 3] - P[1, 3]

    A[2, 3] = P[0, 0] - x[1, 0]*P[2, 0]
    A[2, 4] = P[0, 1] - x[1, 0]*P[2, 1]
    A[2, 5] = P[0, 2] - x[1, 0]*P[2, 2]
    b[2] = x[1, 0]*P[2, 3] - P[0, 3]

    A[3, 3] = P[1, 0] - x[1, 1]*P[2, 0]
    A[3, 4] = P[1, 1] - x[1, 1]*P[2, 1]
    A[3, 5] = P[1, 2] - x[1, 1]*P[2, 2]
    b[3] = x[1, 1]*P[2, 3] - P[1, 3]

    A[4, 0] = 1
    A[4, 3] = -1
    b[4, 0] = X[0][0] - X[0][1]
    A[5, 2] = 1
    A[5, 5] = -1
    A[6, 1] = 1
    A[6, 4] = -1
    return A, b


r = R.from_euler('xyz', [0, 0, 0], degrees=True)
rM = r.as_matrix()
rMT = -rM.transpose()
t = np.array([[0], [0], [0]])
c = np.dot(rMT, t).transpose()
rt = np.concatenate((rM, c.transpose()), axis=1)
row, column = np.shape(rt)
mu = [[2500, 0, 1920/2], [0, 2500, 1080/2], [0, 0, 1]]
P = np.dot(mu, rt)
X = np.array([[-0.5, 0.5, 0.5, 1], [0.5, 0.5, 0.5, 1]])
xc = project_points(P, X)
x = np.dot(P, X.T).transpose()
x = homogenize(x)
P = np.dot(mu, rt)
row, column = np.shape(X)
Xnoise = X + np.random.randn(row, column)/10
A, b = linear_system(P, Xnoise, x)
Xs = np.concatenate([X[0, 0:3].T, X[1, 0:3].T])
e = np.dot(A[0:6], Xs) - b[0:6].T
print(e)
tmp = np.linalg.lstsq(A, b, rcond=None)
print(tmp)
