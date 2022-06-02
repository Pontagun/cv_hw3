import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, make_classification, make_blobs
import pandas as pd
import matplotlib.pyplot as plt

iteration = 5
val_x = []
val_y = []

a = np.array([[-1, 0, 1], [0, 1, 1]])  # t[0:1]
b = np.array([[1, 0, 1], [1, 1, 1]])  # t[2:3]
t = np.array([1, 1, -1, -1])
w = np.array([[0, 0, 0]])

X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1
)


#
#
# def recursive(r, n):
#     recursive(r - (r - 1), n)
#
#     # if
#     # for i in range(1, n):
#     #     for j in range(i, n):
#     #         for k in range(j, n):
#     #             print("{} {} {}".format(i, j, k))
#
#
# def face(r):
#     if r > 1:
#         return r * face(r - 1)
#     else:
#         return 1

def perceptron(w, a):
    for i in range(iteration):
        for eid in range(len(t)):  # use t because we do elementwise.
            net_input = np.dot(w, a[eid, :].transpose())

            error = net_input - t[eid]

            if t[eid] > 0 and net_input <= 0:
                w = w + a[eid, :]
            elif t[eid] < 0 and net_input >= 0:
                w = w - a[eid, :]
            else:
                w = w

    for j in range(-2, 3):
        if w[0][1] != 0:
            y = (-1 * w[0][0] * j) - w[0][2]
            y = y / w[0][1]
        else:
            y = j
        val_y.append(y)

        if w[0][0] != 0:
            x = (-1 * w[0][1] * j) - w[0][2]
            x = x / w[0][0]
        else:
            x = j
        val_x.append(x)

    return val_x, val_y


if __name__ == '__main__':
    val_x, val_y = perceptron(w, X1)

    plt.plot(val_x, val_y)
    plt.plot(a[0:2, 0], a[0:2, 1], 'bo')
    plt.plot(a[2:, 0], a[2:, 1], 'rx')
    plt.show()
