#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Any, Optional, List
from math import log, exp
from multiprocessing import cpu_count, Pool


sns.set()


def sample_data(
    data_filepath: str = "signal.txt",
) -> Tuple[Any, np.ndarray, Any, Any, Any]:
    """
    It takes a signal, samples it at exponentially spaced points, and then reconstructs it at regularly
    spaced points

    :param data_filepath: the path to the file containing the signal, defaults to signal.txt
    :type data_filepath: str (optional)
    :return: a tuple of four elements. The first element is a vector of length N, the second element is
    a vector of length N, the third element is a vector of length M, and the fourth element is a matrix
    of size MxN.
    """
    with open(data_filepath, "r") as f:
        x = np.array([float(line) for line in f.readlines()])

    Tmin, Tmax, N = 1, 1000, len(x)
    exp_samp = lambda i: Tmin * np.exp(-(i - 1) * (np.log(Tmin / Tmax)) / (N - 1))
    T = exp_samp(np.arange(1, N + 1))

    tmin, tmax, M = 0, 1.5, 50
    regular_samp = lambda m: tmin + (tmax - tmin) * (m - 1) / (M - 1)
    t = regular_samp(np.arange(1, M + 1))

    K = np.exp(-t.reshape((M, 1)) @ T.reshape((1, N)))

    z = K @ x
    sigma = 1e-2 * z[0]
    w = np.random.normal(scale=sigma, size=M)
    y = z + w

    return (T, x, t, y, K)


class Optimizer:
    def __init__(self, K: np.ndarray, y: np.ndarray, x: np.ndarray) -> None:
        self.K = K
        self.y = y
        self.x = x
        self.N = len(x)

    def optimize(self, beta: float):
        pass

    def grid_search(self, parameters: np.ndarray) -> List:
        """
        > `grid_search` takes a list of betas and returns a list of the corresponding values of the
        objective function

        :param betas: a numpy array of floats
        :type betas: np.ndarray
        :return: A list of the grid search results for each beta.
        """

        return list(map(self.optimize, tqdm(parameters, total=len(parameters))))


class SmoothOptimize(Optimizer):
    def __init__(self, K: np.ndarray, y: np.ndarray, x: np.ndarray) -> None:
        super().__init__(K, y, x)
        self.D = np.eye(self.N) - np.roll(np.eye(self.N), -1, 1)

    def optimize(self, beta: float) -> Tuple:
        """
        We're trying to find the value of $\beta$ that minimizes the error between the true signal $ and
        the estimated signal $\hat{x}$ with an l2 penalization

        :param beta: the regularization parameter
        :type beta: float
        :return: The tuple contains the following:
            1. The optimal solution to the problem
            2. The relative error between the optimal solution and the true solution
            3. The value of beta
        """
        return (
            x_hat := (
                np.linalg.inv(self.K.T @ self.K + beta * self.D.T @ self.D)
                @ self.K.T
                @ self.y
            ),
            np.linalg.norm(x_hat - self.x) / np.linalg.norm(self.x),
            beta,
        )


def display_result(T, grid, x, alg: str):
    if type(grid[0][1]) is list:
        x_hat, err, beta = grid[min(enumerate(grid), key=lambda x: x[1][1][-1])[0]]
        plt.plot(err)
        plt.title(f"Errors for {beta=:.2f}")
        plt.show()
        err = err[-1]

        end_errors = [np.min(el[1]) for el in grid]

    else:
        x_hat, err, beta = grid[min(enumerate(grid), key=lambda x: x[1][1])[0]]
        end_errors = [el[1] for el in grid]

    plt.plot(T, x_hat, label="Reconstructed Signal")
    plt.plot(T, x, label="Original signal")
    plt.legend()
    plt.xscale("log")
    plt.title(f"Reconstructed signal, {beta=:.2f}, {err=:.2f} ")
    plt.show()

    plt.savefig(f"{alg}_signal.png")

    betas = [el[2] for el in grid]
    plt.scatter(betas, end_errors, marker="x")
    plt.title("Errors depending on beta")
    plt.xlabel("beta")
    plt.ylabel("Relatve error")
    plt.yscale("log")
    plt.show()
    plt.savefig(f"{alg}_beta.png")


#%%


def projection(v, x):
    return np.clip(v, 0, max(x))


class SmoothProject(Optimizer):
    def __init__(self, K: np.ndarray, y: np.ndarray, x: np.ndarray, eps: float) -> None:
        super().__init__(K, y, x)
        self.D = np.eye(self.N) - np.roll(np.eye(self.N), -1, 1)
        self.projection = projection
        self.eps = eps

    def optimize(
        self,
        beta: float,
        _lambda: Optional[float] = None,
        gamma: Optional[float] = None,
        n_iter: int = 100000,
    ):

        if gamma is None or _lambda is None:
            gamma = (
                0.9
                * 2.0
                / max(np.linalg.eigvalsh(self.K.T @ self.K + beta * self.D.T @ self.D))
            )
            _lambda = 1 / 2.0

        xs = []
        x_k = np.ones_like(self.x) * (max(self.x) + min(self.x)) / 2.0
        errors = []
        f_grad = self.K.T @ self.K + beta * self.D.T @ self.D
        s_grad = self.K.T @ self.y
        _grad_x = f_grad @ x_k - s_grad
        i = 0
        while i < n_iter and np.linalg.norm(_grad_x) > self.eps:
            _grad_x = f_grad @ x_k - s_grad
            y_k = self.projection(x_k - gamma * _grad_x, self.x)
            x_k += _lambda * (y_k - x_k)
            errors.append(np.linalg.norm(x_k - self.x) / np.linalg.norm(self.x))
            i += 1
            xs.append(x_k)

        return (xs[np.argmin(errors)], errors, beta)


#%%


# %%


def lambert_o_exp(z):
    if z > 100:
        return z - log(z)
    if z < -20:
        return 0
    w = 1
    v = float("inf")
    x = exp(z)
    while abs(w - v) / abs(w) > 1e-8:
        v = w
        e = exp(w)
        f = w * e - x
        w -= f / ((e * (w + 1) - (w + 2) * f / (2 * w + 2)))
    return w


lambert_vec = np.vectorize(lambert_o_exp)


def l1_prox(x, beta, gamma):
    m = np.zeros((len(x), 2))
    m[:, 0] = np.abs(x) - beta * gamma

    return np.sign(x) * np.max(m, axis=1)


def ent_prox(x, beta, gamma):
    gammabeta = gamma * beta
    return gammabeta * lambert_vec(x / (gammabeta) - 1 - log(gammabeta))


class FB_Optmizer(Optimizer):
    def __init__(
        self, K: np.ndarray, y: np.ndarray, x: np.ndarray, eps: float, penal: str = "l1"
    ) -> None:
        super().__init__(K, y, x)
        self.eps = eps
        self.penal = penal

    def optimize(
        self,
        beta: float,
        _lambda: Optional[float] = None,
        gamma: Optional[float] = None,
        n_iter: int = 25000,
    ):

        if gamma is None or _lambda is None:
            gamma = 0.9 * 2.0 / max(np.linalg.eigvalsh(self.K.T @ self.K))
            _lambda = 1 / 2.0
        xs = []
        _prox = (
            lambda x: l1_prox(x, beta, gamma)
            if self.penal == "l1"
            else ent_prox(x, beta, gamma)
        )

        x_k = np.ones_like(self.x) * (max(self.x) + min(self.x)) / 2.0
        errors = []
        i = 0
        y_k = x_k - gamma * self.K.T @ (self.K @ x_k - self.y)
        while i < n_iter and np.linalg.norm(y_k) > self.eps:

            y_k = x_k - gamma * self.K.T @ (self.K @ x_k - self.y)
            x_k += _lambda * (_prox(y_k) - x_k)
            errors.append(np.linalg.norm(x_k - self.x) / np.linalg.norm(self.x))
            i += 1
            xs.append(x_k)

        return (xs[np.argmin(errors)], errors, beta)


class DR(Optimizer):
    def __init__(
        self, K: np.ndarray, y: np.ndarray, x: np.ndarray, eps: float, penal: str = "l1"
    ) -> None:
        super().__init__(K, y, x)
        self.eps = eps

    def optimize(
        self,
        beta: float,
        _lambda: Optional[float] = None,
        gamma: Optional[float] = None,
        n_iter: int = 25000,
    ):

        if gamma is None or _lambda is None:
            gamma = 0.9 * 2.0 / max(np.linalg.eigvalsh(self.K.T @ self.K))
            _lambda = 1 / 2.0
        xs = []
        _prox_ent = lambda x: ent_prox(x, beta, gamma)
        F1_prox = lambda x: np.linalg.inv(
            gamma * self.K.T @ self.K + np.eye(self.N)
        ) @ (gamma * self.K.T @ self.y + x)

        x_k = np.ones_like(self.x) * (max(self.x) + min(self.x)) / 2.0
        errors = []
        i = 0
        y_k = x_k - gamma * self.K.T @ (self.K @ x_k - self.y)
        while i < n_iter and np.linalg.norm(y_k) > self.eps:

            y_k = _prox_ent(x_k)
            z_k = F1_prox(2 * y_k - x_k)
            x_k += _lambda * (z_k - y_k)
            errors.append(np.linalg.norm(x_k - self.x) / np.linalg.norm(self.x))
            i += 1
            xs.append(x_k)

        return (xs[np.argmin(errors)], errors, beta)


#%%
# betas = np.linspace(0, 1e-1, 10)

# l1_opt = FB_Optmizer(K=K, y=y, x=x, eps=0.015)

# grid = l1_opt.grid_search(betas)

# display_result(T, grid, x)


# #%%
# betas = np.linspace(1e-3, 1e-2, 10)

# l1_opt = FB_Optmizer(K=K, y=y, x=x, eps=0.015, penal='ent')

# grid = l1_opt.grid_search(betas)

# display_result(T, grid, x)
# %%

if __name__ == "__main__":

    (T, x, t, y, K) = sample_data()

    # plt.plot(T, x)
    # plt.xscale("log")
    # plt.show()

    # plt.plot(t, y)
    # plt.xlim(left=0)
    # plt.show()

    # betas = np.linspace(1e-2, 0.6, 10)

    # l2_proj_opt = SmoothProject(K=K, y=y, x=x, eps=0.005)

    # with Pool(cpu_count() - 1) as pool:
    #     grid = list(
    #         tqdm(
    #             pool.imap(SmoothProject(K=K, y=y, x=x, eps=0.005).optimize, betas),
    #             total=len(betas),
    #         )
    #     )

    l2_opt = FB_Optmizer(K=K, y=y, x=x, eps=0.012, penal="ent")

    betas = np.linspace(1e-4, 1, 10)

    grid = l2_opt.grid_search(betas)

    display_result(T, grid, x)
# %%
