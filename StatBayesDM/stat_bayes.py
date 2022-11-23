import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

sns.set()


def get_pseudo_likelihood(
    _lambda: int, gamma: float, delta: float, data: np.ndarray
) -> float:
    """
    It computes the probability of each possible value of $\lambda$ given the data and the parameters
    $\gamma$ and $\delta$

    :param _lambda: the number of time steps before the change point
    :type _lambda: int
    :param gamma: the rate of the exponential distribution for the first lambda observations
    :type gamma: float
    :param delta: the rate of the exponential distribution for the time between events after the change
    point
    :type delta: float
    :param data: the data we're trying to fit
    :type data: np.ndarray
    :return: The probability of each value of lambda
    """
    T = len(data)
    s = np.prod(
        [gamma ** data[i] if i < _lambda else delta ** data[i] for i in range(T - 1)]
    )
    return s * np.exp(-_lambda * gamma) * np.exp(-(T - _lambda) * delta)


def get_lambda_distrib(gamma: float, delta: float, data: np.ndarray) -> np.ndarray:
    """
    It takes in a gamma, delta, and data, and returns a probability distribution over the possible
    values of lambda

    :param gamma: the probability of a new customer arriving
    :type gamma: float
    :param delta: the probability of a customer arriving at the store
    :type delta: float
    :param data: the data we're trying to fit
    :type data: np.ndarray
    :return: The probability of each lambda value.
    """
    un_norm = np.array(
        [
            get_pseudo_likelihood(_lambda=i, gamma=gamma, delta=delta, data=data)
            for i in range(len(data) - 1)
        ]
    )
    return un_norm / np.sum(un_norm)


def get_data(filepath: str = "acc_usine.txt") -> np.ndarray:
    """
    It reads the file at `filepath` and returns the data as a numpy array

    :param filepath: str = "acc_usine.txt", defaults to acc_usine.txt
    :type filepath: str (optional)
    :return: A numpy array of the data
    """
    with open(filepath, "r") as f:
        data = [int(a) for a in f.readlines()]
    return np.array(data)


def main(N=1000, a1=1, a2=1, d1=1, d2=1, plot=False):
    """
    It samples from the posterior distribution of the parameters of the model, and returns the mean of
    the sampled values of $\lambda$ (the changing point) and the mean of the number of accidents before
    and after the changing point

    :param N: number of iterations, defaults to 1000 (optional)
    :param a1: shape parameter for gamma prior on gamma, defaults to 1 (optional)
    :param a2: prior for lambda, defaults to 1 (optional)
    :param d1: prior number of accidents before the change point, defaults to 1 (optional)
    :param d2: the prior for the rate of the second Poisson distribution, defaults to 1 (optional)
    :param plot: whether to plot the results or not, defaults to False (optional)
    """

    data = get_data()

    T = len(data)

    _lambda = np.random.randint(T - 1)

    gammas = []
    deltas = []
    lambdas = []

    for _ in tqdm(range(N)):
        gamma = np.random.gamma(
            a1 + sum(data[i] for i in range(_lambda)), (_lambda + a2) ** -1
        )
        delta = np.random.gamma(
            d1 + sum(data[i] for i in range(_lambda, T)), (T - _lambda + d2) ** -1
        )
        _lambda = np.random.choice(T - 1, p=get_lambda_distrib(gamma, delta, data))

        gammas.append(gamma)
        deltas.append(delta)
        lambdas.append(_lambda)

    n_lambda = np.mean(lambdas[:-50])
    std_lambda = np.std(lambdas[:-50])

    if plot:
        plt.hist(lambdas)
        plt.title("Frequency of lambda values")
        plt.show()

        plt.plot(data)
        plt.title("Experimental data")
        plt.ylabel("Number of accidents")
        plt.xlabel("Years since 1852")
        plt.show()

    print(f" Estimated changing point : {n_lambda} ({std_lambda})")
    print(
        f" Number of accidents : {np.mean(data[:int(n_lambda)])} (ante), {np.mean(data[int(n_lambda):])} (post)"
    )


if __name__ == "__main__":
    main()
