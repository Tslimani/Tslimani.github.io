# -*- coding: utf-8 -*-

"""
The sequential stretch move algorithm.
"""

# import numpy as np
# import matplotlib.pyplot as plt

# def target_distribution(x, rho=0.5):
#     # Define the target distribution (bivariate normal with correlation)
#     mean = np.zeros(len(x))
#     cov = np.array([[1, rho], [rho, 1]])  # Covariance matrix with correlation coefficient rho
#     return np.exp(-0.5 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean))) / (2 * np.pi * np.sqrt(1 - rho ** 2))

# def draw_walker(S_i_t, i):
#     # Draw a walker X_j at random from the complementary ensemble S(i)(t)
#     # Exclude the walker at index i
#     complementary_indices = list(range(len(S_i_t)))
#     complementary_indices.remove(i)
#     j = np.random.choice(complementary_indices)
#     return S_i_t[j]

# def stretch_move(i, S_i_t, N):
#     # Perform a single stretch move update step
#     X_k = S_i_t[i]
#     X_j = draw_walker(S_i_t, i)  # Draw a walker X_j at random from the complementary ensemble S(i)(t)

#     # Draw a random variable Z from the proposal distribution
#     Z = 0.5 * np.random.rand() ** 2 + np.random.rand() + 0.5

#     # Calculate the proposed new position Y
#     Y = X_j + Z * (X_k - X_j)

#     # Calculate the acceptance probability q
#     p_Y = target_distribution(Y)
#     p_Xk = target_distribution(X_k)
#     q = min(1, Z ** (N - 1) * (p_Y / p_Xk))
#     r = np.random.uniform(0, 1)

#     # Accept or reject the proposed move based on q
#     if r <= q:
#         return Y  # Accept the move
#     else:
#         return X_k  # Reject the move

# def stretch_move_algorithm(num_walkers, num_samples, dimension):
#     # Initialize ensemble of walkers
#     S_i_t = np.random.randn(num_walkers, dimension)

#     # Store samples for each walker
#     samples = np.zeros((num_samples, num_walkers, dimension))

#     # Main loop over samples
#     for sample in range(num_samples):
#         # Loop over walkers
#         for i in range(num_walkers):
#             S_i_t[i] = stretch_move(i, S_i_t, dimension)
#             samples[sample, i] = S_i_t[i]

#     return samples

# # Example usage
# num_walkers = 100
# num_samples = 200
# dimension = 2

# # Run stretch move algorithm to generate samples from the posterior distribution
# ensemble_samples = stretch_move_algorithm(num_walkers, num_samples, dimension)

# # Define colors for each walker
# colors = plt.cm.jet(np.linspace(0, 1, num_walkers))

# # Plotting the ensemble results with color-coded walkers
# plt.figure(figsize=(10, 6))
# for i in range(num_walkers):
#     plt.scatter(ensemble_samples[:, i, 0], ensemble_samples[:, i, 1], alpha=0.5, color=colors[i], label=f'Walker {i}')
# plt.title('Ensemble Results')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=5)
# plt.grid(True)
# plt.show()

"""The **Parallel** stretch move algorithm"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from timeit import default_timer as timer
from datetime import timedelta

def target_distribution(x, rho=0.5):
    # Define the target distribution (bivariate normal with correlation)
    mean = np.zeros(len(x))
    cov = np.array([[1, rho], [rho, 1]])  # Covariance matrix with correlation coefficient rho
    return np.exp(-0.5 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean))) / (2 * np.pi * np.sqrt(1 - rho ** 2))

def draw_walker(S_i_t, i):
    # Draw a walker X_j at random from the complementary ensemble S(i)(t)
    # Exclude the walker at index i
    complementary_indices = list(range(len(S_i_t)))
    complementary_indices.remove(i)
    j = np.random.choice(complementary_indices)
    return S_i_t[j]

def stretch_move(args):
    i, S_i_t, N = args[0], args[1], args[2]
    # Perform a single stretch move update step
    X_k = S_i_t[i]
    # Draw a walker X_j at random from the complementary ensemble S(i)(t)
    X_j = draw_walker(S_i_t, i)

    # Draw a random variable Z from the proposal distribution
    Z = 0.5 * np.random.rand() ** 2 + np.random.rand() + 0.5

    # Calculate the proposed new position Y
    Y = X_j + Z * (X_k - X_j)

    # Calculate the acceptance probability q
    p_Y = target_distribution(Y)
    p_Xk = target_distribution(X_k)
    q = min(1, Z ** (N - 1) * (p_Y / p_Xk))
    r = np.random.uniform(0, 1)

    # Accept or reject the proposed move based on q
    if r <= q:
        return Y  # Accept the move
    else:
        return X_k  # Reject the move

def stretch_move_algorithm(args):
    num_walkers, num_samples, dimension = args

    # Initialize ensemble of walkers
    S_i_t = np.random.randn(num_walkers, dimension)

    # Store samples for each walker
    samples = np.zeros((num_samples, num_walkers, dimension))

    # Main loop over samples
    for sample in range(num_samples):
        with mp.Pool(mp.cpu_count()) as pool:
            args_list = [(i, S_i_t, dimension) for i in range(num_walkers)]
            results = pool.map(stretch_move, args_list)

        # Update ensemble after processing all walkers
        for i, result in enumerate(results):
            S_i_t[i] = result

        samples[sample] = S_i_t  # Store samples after each iteration

    return samples

if __name__ == "__main__":
    # Example usage
    num_walkers = 100
    num_samples = 200
    dimension = 2

    # SEQUENTIAL #################################################################
    # print(f'running stretch move on {mp.cpu_count()} cores.\n')
    # print(f'parameters: n_samples: {num_samples}, n_walkers: {num_walkers}\n')
    # start = timer()
    # ensemble_samples = stretch_move_algorithm([num_walkers, num_samples, dimension])
    # end = timer()
    # print(f'elapsed time: {timedelta(seconds=(round(end-start,2)))}')
    # colors = plt.cm.jet(np.linspace(0, 1, num_walkers))

    # # Plotting the ensemble results with color-coded walkers
    # plt.figure(figsize=(10, 6))
    # for i in range(num_walkers):
    #     plt.scatter(ensemble_samples[:, i, 0], ensemble_samples[:, i, 1], alpha=0.5, color=colors[i], label=f'Walker {i}')
    # plt.title('Ensemble Results')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=5)
    # plt.grid(True)
    # plt.show()
    #############################################################################

    # Run stretch move algorithm to generate samples from the posterior distribution
    print(f'running stretch move on {mp.cpu_count()} cores.\n')
    print(f'parameters: n_samples: {num_samples}, n_walkers: {num_walkers}\n')
    #Timestamp to evaluate the running time of the algorithm
    start = timer()
    ensemble_samples = stretch_move_algorithm([num_walkers, num_samples, dimension])
    end = timer()
    print(f'elapsed time: {timedelta(seconds=(round(end-start,2)))}')


    # Define colors for each walker
    colors = plt.cm.jet(np.linspace(0, 1, num_walkers))

    # Plotting the ensemble results with color-coded walkers
    plt.figure(figsize=(10, 6))
    for i in range(num_walkers):
        plt.scatter(ensemble_samples[:, i, 0], ensemble_samples[:, i, 1], alpha=0.5, color=colors[i], label=f'Walker {i}')
    plt.title('Ensemble Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=5)
    plt.grid(True)
    plt.show()