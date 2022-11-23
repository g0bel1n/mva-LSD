import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sns.set()


# It's a class that represents a quadratic program with inequality constraints
class QP:
    def __init__(self, Q, p, A, b, t):
        self.Q = Q
        self.p = p
        self.A = A
        self.b = b
        self.t = t

    def __call__(self, x):
        return (
            (
                0.5 * (x.T @ self.Q @ x)
                + self.p.T @ x
                - self.t * np.sum(np.log(self.b - self.A @ x))
            )
            if np.all(self.b - self.A @ x > 0)
            else np.inf
        )

    def gradient(self, x):
        return (
            self.Q @ x
            + self.p
            - self.t
            * np.sum(
                [
                    self.A[i, :] / (self.A @ x - self.b)[i]
                    for i in range(self.A.shape[0])
                ],
                axis=0,
            )
        )


def back_tracking_line_search(f, x, d, alpha, beta, c):
    """
    Backtracking line search
    :param f: function
    :param x: current point
    :param d: search direction
    :param alpha: step size
    :param beta: step size reduction
    :param c: sufficient decrease
    :return: step size
    """
    f_x = f(x)
    i = 0
    while f(x + alpha * d) > f_x + c * alpha * np.dot(f.gradient(x), d):

        if i < 5:
            print("f(x + alpha * d): ", f(x + alpha * d))
            print("f_x +  ", f_x + c * alpha * np.dot(f.gradient(x), d))
        alpha *= beta
        i += 1
    print("#################")
    return alpha


def centering_step(Q, p, A, b, t, v0, eps):
    """
    Centering step
    :param Q: quadratic term
    :param p: linear term
    :param A: inequality constraint matrix
    :param b: inequality constraint vector
    :param t: penalty parameter
    :param v0: initial point
    :param eps: stopping criterion
    :return: solution
    """
    v = v0
    f = QP(Q, p, A, b, t)
    while True:
        d = -f.gradient(v)
        alpha = back_tracking_line_search(f, v, d, 1, 0.5, 0.5)
        v += alpha * d

        if (a := np.linalg.norm(f.gradient(v))) < eps:
            break
        # print(d)
    return v


def barr_method(Q, p, A, b, v0, eps, mu=10):
    """
    Barrier method
    :param Q: quadratic term
    :param p: linear term
    :param A: inequality constraint matrix
    :param b: inequality constraint vector
    :param v0: initial point
    :param eps: stopping criterion
    :return: solution
    """
    t = 1
    v = v0
    vs = []
    m = A.shape[0]
    with tqdm() as pbar:
        while True:
            v = centering_step(Q, p, A, b, t, v, eps * 100)
            vs.append(v)
            t *= mu
            if m / t < eps:
                break
            pbar.update(1)
    return vs


if __name__ == "__main__":

    N: int = 5
    lambda_ = 10
    Q = np.eye(N)
    p = np.random.randn(N)
    X = np.random.randn(N, N)
    p = -(X @ np.arange(N) + np.random.normal(0, 1, N))
    A = np.concatenate([X.T, -X.T], axis=0)

    b = lambda_ * np.ones(2 * N)
    v0 = np.random.randn(N)

    f = QP(Q, p, A, b, 0)
    eps = 1e-5

    vs = barr_method(Q, p, A, b, v0, eps)
    print(vs[-1])
    f_v = [f(v) - f(vs[-1]) for v in vs]

    plt.plot(f_v)
    plt.xlabel("iteration")
    plt.ylabel("f(v) - f(v*)")
    plt.show()
