import operator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import time
from scipy.optimize import minimize


rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')


def phi_n_of_x(n: int) -> np.ndarray:
    """
    Basis function factory. Allows for easy generation of \phi_n(x).
    :param n: Dimensionality of the basis function
    :return: \phi_n(x)
    """

    def phi_of_x(x):
        return np.array([x ** i for i in range(n)])

    return phi_of_x


def kernel_se(x_1, x_2, l, sigma_f):
    """
    Squared exponential kernel

    :param x_1:
    :param x_2:
    :param l: length scale
    :param sigma_f:
    :return:
    """
    l2 = np.ones(x_1.shape[1])/l/l
    M = np.diag(l2)
    K = np.zeros([x_1.shape[0], x_2.shape[0]])
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            x1i = x_1[i]
            x2j = x_2[j]
            dist = x1i-x2j
            
            K[i,j] = sigma_f**2 * (np.exp(-0.5*dist.T@ M @dist)).item()
    return K

def gp_prediction_a(data: np.ndarray, X: np.ndarray, phi: callable, sigma_n: float, Sigma_p: np.ndarray) \
        -> (np.ndarray, np.ndarray):
    """
    Implementation of 2.11
    :param data: Data on which the model is fit
    :param X: Data to predict the mean and variance
    :param phi: basis functions
    :param sigma_n: variance for points in data
    :param Sigma_p: covariance
    :return: mean and variance for all points in X
    """
    # compute parts
    X0, y0 = (zip(*data))
    X0 = np.array(X0)
    y0 = np.array(y0)
    phi0 = phi(X0)
    sigma_n2_inv = pow(sigma_n,-2)
    A = sigma_n2_inv * phi0@phi0.T + np.linalg.inv(Sigma_p)
    A_inv = np.linalg.inv(A)

    # for storing the results
    means = []
    variances = []
    # loop over all X
    for xs in X:
        phis = phi(xs)
        mean = sigma_n2_inv*phis.T@A_inv@phi0@y0
        var = phis.T @ A_inv @ phis
        means.append(mean.item())
        variances.append(var.item())


    return np.array(means), np.array(variances)


def gp_prediction_b(data: np.ndarray, X: np.ndarray, phi: callable, sigma_n: float, Sigma_p: np.ndarray) \
        -> (np.ndarray, np.ndarray):
    """
    Implementation of 2.12
    :param data: Data on which the model is fit
    :param X: Data to predict the mean and variance
    :param phi: basis functions
    :param sigma_n: variance for points in data
    :param Sigma_p: covariance
    :return: mean and variance for all points in X
    """
    # compute parts
    X0, y0 = (zip(*data))
    X0 = np.array(X0)
    y0 = np.array(y0).reshape([-1,1])
    phi0 = phi(X0)
    sigma_n2 = pow(sigma_n,2)
    n = y0.shape[0]
    A = Sigma_p@phi0
    B = phi0.T@Sigma_p@phi0 + sigma_n2*np.eye(n)
    C = A@np.linalg.inv(B)

    # for storing the results
    means = []
    variances = []
    # loop over all X
    for xs in X:
        phis = phi(xs).reshape([-1,1])
        mean = phis.T@C@y0
        var = phis.T@Sigma_p@phis - phis.T@C@phi0.T@Sigma_p@phis
        means.append(mean.item())
        variances.append(var.item())

    return np.array(means), np.array(variances)


def gp_prediction_b_kernel(data: np.ndarray, X: np.ndarray, l: float, sigma_n: float, sigma_f: float) \
        -> (np.ndarray, np.ndarray):
    """
    Implementation of 2.12 in a kernelized way
    :param data: Data on which the model is fit
    :param X: Data to predict the mean and variance
    :param l: length-scale
    :param sigma_n: variance for points in data
    :param sigma_f:
    :return:
    """
    X_train = np.array(list(map(operator.itemgetter(0), data))).reshape(-1, 1)
    Y_train = np.array(list(map(operator.itemgetter(1), data)))

    X = X.reshape(-1, 1)
    # For Exercise c
    K_f = kernel_se(X_train, X_train, l, sigma_f)
    K_y = K_f + sigma_n ** 2 * np.eye(len(X_train))
    K_star = kernel_se(X_train, X, l, sigma_f)
    K_starstar = kernel_se(X, X, l, sigma_f) + 1e-8 * np.eye(len(X))
    K_y_inv = np.linalg.inv(K_y)

    cov_s = K_starstar - np.dot(np.dot(np.transpose(K_star), K_y_inv), K_star)
    mu_s = np.dot(np.transpose(K_star), np.dot(K_y_inv, Y_train))

    cov_s = np.diag(cov_s)
    return mu_s, cov_s


def negative_log_likelihood(data: np.ndarray):
    """
    Implementation of the *negative* log likelihood as a factory method
    :param data: Data on which the model is fit
    :return:
    """
    X_train = np.array(list(map(operator.itemgetter(0), data))).reshape(-1, 1)
    Y_train = np.array(list(map(operator.itemgetter(1), data))).reshape(-1, 1)

    def nll(theta):
        """
        The actual negative log likelihood function
        :param theta: a list or tuple containing the GP parameters to be optimized.
        :return:
        """
        l, sigma_f, sigma_n = theta
        K_f = kernel_se(X_train, X_train, l, sigma_f)
        K_y = K_f + sigma_n ** 2 * np.eye(len(X_train))

        L = np.linalg.cholesky(K_y)

        alpha = np.linalg.inv(L.T)@np.linalg.inv(L)@Y_train
        term_1 = (0.5*Y_train.T@alpha).item()

        # You are given term 2 and 3
        term_2 = np.sum(np.log(np.diag(L)))

        term_3 = 0.5 * len(X_train) * np.log(2 * np.pi)
        return term_1 + term_2 + term_3

    return nll


def main():
    # ################################################### Data to use ######################################################
    train_data = np.array([(np.array(x), np.sin(x) + 0.1 * np.cos(10 * x)) for x in np.linspace(0, 2 * np.pi, 64)])
    X = np.array([np.array(x) for x in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    x, y = (zip(*train_data))

    # ##################################################### Exercise (a)
    # Plot model predictions for the bayesian linear regression:
    num_features = 2
    means, variances = gp_prediction_a(train_data, x, phi_n_of_x(num_features), 1.0, np.eye(num_features))

    plt.figure(figsize=(5, 5))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.fill_between(x, means - 2 * np.sqrt(variances), means + 2 * np.sqrt(variances), alpha=0.2, color='g')
    plt.fill_between(x, means - 1 * np.sqrt(variances), means + 1 * np.sqrt(variances), alpha=0.2, color='g')
    plt.plot(x, means, label='posterior mean', color='g')
    plt.scatter(x, y, label='observations')
    plt.title('Bayesian linear regression example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    # plt.savefig('plot1.pdf')

    # ##################################################### Exercise (b)
    # compare the predictions
    mean1, var1 = gp_prediction_a(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
    mean2, var2 = gp_prediction_b(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
    # decide if mean and variance are equal
    print('Equal means: ', np.allclose(mean1, mean2))
    print('Equal variances: ', np.allclose(var1, var2))
    print()

    # 'Warm-up' processor for fair time measurements
    for i in range(100):
        gp_prediction_a(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))

    plot the computation time
    tsa = []
    tsb = []
    dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for n in dimensions:
        print('Computing in feature dimension: ', n)
        I_n = np.eye(n)  # setup Identity of dimensionality n
        phi_n = phi_n_of_x(n)  # get basis functions

        start = time.time()
        gp_prediction_a(train_data, X, phi_n, 1.0, I_n)
        run_time = time.time() - start
        tsa.append(run_time)
        start = time.time()
        gp_prediction_b(train_data, X, phi_n, 1.0, I_n)
        run_time = time.time() - start
        tsb.append(run_time)

    plt.figure(figsize=(5, 5))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.loglog(dimensions, tsa, label='Implementation 2.11', drawstyle='steps-post')
    plt.loglog(dimensions, tsb, label='Implementation 2.12', drawstyle='steps-post')
    plt.xlabel('input dimension')
    plt.ylabel('runtime[sec]')
    plt.legend(loc=2)
    plt.show()

    # ##################################################### Exercise (c)
    # ################################################### Data to use for this exercise
    np.random.seed(666)
    train_data = np.array([(np.random.uniform(0, 2 * np.pi), np.sin(x) + 1.5 * np.random.random() * 5) for x in
                           np.linspace(0, 2 * np.pi, 9)])
    X = np.array([np.array(x) for x in np.linspace(0, 2 * np.pi, 128, endpoint=False)])
    x, y = (zip(*train_data))

    # Initial se kernel parameters
    l_init, sigma_f_init, sigma_n_init = 0.2, 0.7, 1.0
    m_init, v_init = gp_prediction_b_kernel(train_data, X, l=l_init, sigma_n=sigma_n_init, sigma_f=sigma_f_init)
    nll_init = negative_log_likelihood(train_data)([l_init, sigma_f_init, sigma_n_init])

    # Optimized kernel parameters
    # Use call back to record optimization trajectory
    intermediate_xs = [np.array([l_init, sigma_f_init, sigma_n_init])]
    callback_recorder = lambda x: intermediate_xs.append(np.array(x))

    res = minimize(negative_log_likelihood(train_data), [l_init, sigma_f_init, sigma_n_init],
                   bounds=((1e-5, None), (1e-5, None), (1e-5, None)), method='L-BFGS-B', callback=callback_recorder)
    trajectory = np.array(intermediate_xs)
    l_opt, sigma_f_opt, sigma_n_opt = res.x
    m_opt, v_opt = gp_prediction_b_kernel(train_data, X, l=l_opt, sigma_n=sigma_n_opt, sigma_f=sigma_f_opt)
    nll_opt = res.fun

    # Plot comparison
    plt.figure(figsize=(7, 5))
    plt.title('GP Hyperparameter Optimization')
    plt.subplot(2, 1, 1)
    plt.title(
        'INIT NLL: {:.3f}, l: {:.3f}, sigma_n: {:.3f}, sigma_f: {:.3f}'.format(nll_init, l_init, sigma_n_init,
                                                                               sigma_f_init))
    plt.fill_between(X, m_init - 2 * np.sqrt(v_init), m_init + 2 * np.sqrt(v_init), alpha=0.2, color='g')
    plt.fill_between(X, m_init - 1 * np.sqrt(v_init), m_init + 1 * np.sqrt(v_init), alpha=0.2, color='g')
    plt.plot(X, m_init, label='posterior mean', color='g')
    plt.scatter(x, y, label='observations')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2, 1, 2)
    plt.title(
        'OPT: NLL: {:.3f}, l: {:.3f}, sigma_n: {:.3f}, sigma_f: {:.3f}'.format(nll_opt, l_opt, sigma_n_opt,
                                                                               sigma_f_opt))
    plt.fill_between(X, m_opt - 2 * np.sqrt(v_opt), m_opt + 2 * np.sqrt(v_opt), alpha=0.2, color='g')
    plt.fill_between(X, m_opt - 1 * np.sqrt(v_opt), m_opt + 1 * np.sqrt(v_opt), alpha=0.2, color='g')
    plt.plot(X, m_opt, label='posterior mean', color='g')
    plt.scatter(x, y, label='observations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Additional plot to visualize the negative log likelihood landscape
    l_res, sigma_f_res = 100, 100
    l_grid = np.logspace(-2, 1, l_res)
    sigma_f_grid = np.logspace(-1, 1.0, sigma_f_res)
    l, sigma_f = np.meshgrid(l_grid, sigma_f_grid)
    nll_grid = np.zeros((l_res, sigma_f_res))
    nll = negative_log_likelihood(train_data)
    for i in range(l_res):
        for j in range(sigma_f_res):
            try:
                nll_grid[i, j] = - nll([l[i, j], sigma_f[i, j], sigma_n_opt])
            except np.linalg.LinAlgError as e:
                print(e)
                nll_grid[i, j] = np.nan
    plt.contour(l, sigma_f, nll_grid, 1000, cmap="RdBu_r", zorder=1)
    cbar = plt.colorbar()
    plt.scatter([l_opt], [sigma_f_opt], label='OPT', marker="*", color="black", s=250, zorder=3)
    plt.quiver(trajectory[:-1, 0], trajectory[:-1, 1], trajectory[1:, 0] - trajectory[:-1, 0],
               trajectory[1:, 1] - trajectory[:-1, 1], scale_units='xy', angles='xy', color="green", scale=1, zorder=2,
               label='Trajectory')
    plt.xlabel('l')
    plt.ylabel('sigma_f')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(frameon=True)
    cbar.set_label('Log Likelihood', rotation=270, labelpad=15)
    plt.show()


if __name__ == '__main__':
    main()
