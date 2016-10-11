# Copyright (c) 2016 Paulo Eduardo Rauber

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#                 ^
#                / \
#                 |
#                 |
#
# License included because this module is a heavily modified version based on
# Paulo's implementation of dynamic t-SNE.
# (https://github.com/paulorauber/thesne)


import math
import numpy as np
import theano
import theano.tensor as T
from sklearn.utils import check_random_state
from scipy.spatial.distance import pdist
from modules.layout_io import save_drawing

epsilon = 1e-16
floath = np.float32


class SigmaTooLowException(Exception):
    pass


class NaNException(Exception):
    pass


# Squared Euclidean distance between all pairs of row-vectors
def sqeuclidean_var(X):
    N = X.shape[0]
    ss = (X ** 2).sum(axis=1)
    return ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * X.dot(X.T)


# Euclidean distance between all pairs of row-vectors
def euclidean_var(X):
    return T.maximum(sqeuclidean_var(X), epsilon) ** 0.5


# Conditional probabilities of picking (ordered) pairs in high-dim space.
def p_ij_conditional_var(X, sigma):
    N = X.shape[0]

    sqdistance = X**2

    esqdistance = T.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    esqdistance_zd = T.fill_diagonal(esqdistance, 0)

    row_sum = T.sum(esqdistance_zd, axis=1).reshape((N, 1))

    return esqdistance_zd / row_sum  # Possibly dangerous


# Symmetrized probabilities of picking pairs in high-dim space.
def p_ij_sym_var(p_ij_conditional):
    return (p_ij_conditional + p_ij_conditional.T) / (2 * p_ij_conditional.shape[0])


# Probabilities of picking pairs in low-dim space (using Student
# t-distribution).
def q_ij_student_t_var(Y):
    sqdistance = sqeuclidean_var(Y)
    one_over = T.fill_diagonal(1 / (sqdistance + 1), 0)
    return one_over / one_over.sum()


# Probabilities of picking pairs in low-dim space (using Gaussian).
def q_ij_gaussian_var(Y):
    sqdistance = sqeuclidean_var(Y)
    gauss = T.fill_diagonal(T.exp(-sqdistance), 0)
    return gauss / gauss.sum()


# Per point cost function
def cost_var(X, Y, sigma, Adj, l_kl, l_e, l_c, l_r, r_eps):
    N = X.shape[0]
    num_edges = 0.5 * T.sum(Adj)

    # Used to normalize s.t. the l_*'s sum up to one.
    l_sum = l_kl + l_e + l_c + l_r

    p_ij_conditional = p_ij_conditional_var(X, sigma)
    p_ij = p_ij_sym_var(p_ij_conditional)
    q_ij = q_ij_student_t_var(Y)

    p_ij_safe = T.maximum(p_ij, epsilon)
    q_ij_safe = T.maximum(q_ij, epsilon)

    # Kullback-Leibler term
    kl = T.sum(p_ij * T.log(p_ij_safe / q_ij_safe), axis=1)

    # Edge contraction term
    edge_contraction = (1 / (2 * num_edges)) * T.sum(Adj * sqeuclidean_var(Y), axis=1)

    # Compression term
    compression = (1 / (2 * N)) * T.sum(Y**2, axis=1)

    # Repulsion term
    # repulsion = (1 / (2 * N**2)) * T.sum(T.fill_diagonal(1 / (euclidean_var(Y) + r_eps), 0), axis=1)
    repulsion = -(1 / (2 * N**2)) * T.sum(T.fill_diagonal(T.log(euclidean_var(Y) + r_eps), 0), axis=1)

    cost = (l_kl / l_sum) * kl + (l_e / l_sum) * edge_contraction + (l_c / l_sum) * compression + (l_r / l_sum) * repulsion

    return cost


# Binary search on sigma for a given perplexity
def find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, verbose=0):
    X = T.fmatrix('X')
    sigma = T.fvector('sigma')

    target = np.log(perplexity)

    P = T.maximum(p_ij_conditional_var(X, sigma), epsilon)

    entropy = -T.sum(P * T.log(P), axis=1)

    # Setting update for binary search interval
    sigmin_shared = theano.shared(np.full(N, np.sqrt(epsilon), dtype=floath))
    sigmax_shared = theano.shared(np.full(N, np.inf, dtype=floath))

    sigmin = T.fvector('sigmin')
    sigmax = T.fvector('sigmax')

    upmin = T.switch(T.lt(entropy, target), sigma, sigmin)
    upmax = T.switch(T.gt(entropy, target), sigma, sigmax)

    givens = {X: X_shared, sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigmin_shared, upmin), (sigmax_shared, upmax)]

    update_intervals = theano.function([], entropy, givens=givens, updates=updates)

    # Setting update for sigma according to search interval
    upsigma = T.switch(T.isinf(sigmax), sigma * 2, (sigmin + sigmax) / 2.)

    givens = {sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigma_shared, upsigma)]

    update_sigma = theano.function([], sigma, givens=givens, updates=updates)

    for i in range(sigma_iters):
        e = update_intervals()
        update_sigma()
        if verbose:
            print('[find_sigma] Iteration {0}: Perplexities in [{1:.4f}, {2:.4f}].'.format(i + 1, np.exp(e.min()), np.exp(e.max())), end='\r')
    if verbose:
        print('\n[find_sigma] Done! Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(e.min()), np.exp(e.max())))

    if np.any(np.isnan(np.exp(e))):
        raise SigmaTooLowException('Invalid sigmas. The perplexity is probably too low.')


# Receives vectors in Y, and moves co-located vertices in opposite directions,
# to assist in the repulsion of vertices.
def switch_shake(Y, magnitude=1e-5):
    N = Y.shape[0]

    # Auxiliary functions for translating from square to condensed indexing
    # of the distance matrix.
    def calc_row_idx(k, n):
        return int(math.ceil((1 / 2.) * (- (-8 * k + 4 * n**2 - 4 * n - 7)**0.5 + 2 * n - 1) - 1))

    def elem_in_i_rows(i, n):
        return i * (n - 1 - i) + (i * (i + 1)) / 2

    def calc_col_idx(k, i, n):
        return int(n - elem_in_i_rows(i + 1, n) + k)

    def condensed_to_square(k, n):
        i = calc_row_idx(k, n)
        j = calc_col_idx(k, i, n)
        return i, j

    euclid_dist = pdist(Y)
    max_dist = euclid_dist.max()
    for idx in np.where(euclid_dist <= np.finfo(np.float32).eps)[0]:
        (i, j) = condensed_to_square(idx, N)
        nudge = np.random.normal(0, max_dist * magnitude, 2)

        # v_i and v_j are co-located. Move v_i in a direction, and move v_j in
        # the opposite direction.
        Y[i, :] += nudge
        Y[j, :] -= nudge
    return Y


# Perform momentum-based gradient descent on the cost function with the given
# parameters. Return the vertex coordinates and per-vertex cost.
def find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
           initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
           final_momentum, momentum_switch,
           initial_l_kl, final_l_kl, l_kl_switch,
           initial_l_e, final_l_e, l_e_switch,
           initial_l_c, final_l_c, l_c_switch,
           initial_l_r, final_l_r, l_r_switch,
           r_eps,
           Adj_shared, g=None, save_every=None, output_folder=None, verbose=0):
    # Optimization hyperparameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    # Hyperparameters used within Theano
    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)
    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Cost parameters
    initial_l_kl = np.array(initial_l_kl, dtype=floath)
    final_l_kl = np.array(final_l_kl, dtype=floath)
    initial_l_e = np.array(initial_l_e, dtype=floath)
    final_l_e = np.array(final_l_e, dtype=floath)
    initial_l_c = np.array(initial_l_c, dtype=floath)
    final_l_c = np.array(final_l_c, dtype=floath)
    initial_l_r = np.array(initial_l_r, dtype=floath)
    final_l_r = np.array(final_l_r, dtype=floath)

    # Cost parameters used within Theano
    l_kl = T.fscalar('l_kl')
    l_kl_shared = theano.shared(initial_l_kl)
    l_e = T.fscalar('l_e')
    l_e_shared = theano.shared(initial_l_e)
    l_c = T.fscalar('l_c')
    l_c_shared = theano.shared(initial_l_c)
    l_r = T.fscalar('l_r')
    l_r_shared = theano.shared(initial_l_r)

    # High-dimensional observations (connectivities of vertices)
    X = T.fmatrix('X')
    # 2D projection (coordinates of vertices)
    Y = T.fmatrix('Y')

    # Adjacency matrix
    Adj = T.fmatrix('Adj')

    # Standard deviations used for Gaussians to attain perplexity
    sigma = T.fvector('sigma')

    # Y velocities (for momentum-based descent)
    Yv = T.fmatrix('Yv')
    Yv_shared = theano.shared(np.zeros((N, output_dims), dtype=floath))

    # Function for retrieving cost for all individual data points
    costs = cost_var(X, Y, sigma, Adj, l_kl, l_e, l_c, l_r, r_eps)

    # Sum of all costs (scalar)
    cost = T.sum(costs)

    # Gradient of the cost w.r.t. Y
    grad_Y = T.grad(cost, Y)

    # Update step for velocity
    update_Yv = theano.function(
        [], None,
        givens={
            X: X_shared,
            sigma: sigma_shared,
            Y: Y_shared,
            Yv: Yv_shared,
            Adj: Adj_shared,
            lr: lr_shared,
            momentum: momentum_shared,
            l_kl: l_kl_shared,
            l_e: l_e_shared,
            l_c: l_c_shared,
            l_r: l_r_shared
        },
        updates=[
            (Yv_shared, momentum * Yv - lr * grad_Y)
        ]
    )

    # Gradient descent step
    update_Y = theano.function(
        [], [],
        givens={
            Y: Y_shared, Yv: Yv_shared
        },
        updates=[
            (Y_shared, Y + Yv)
        ]
    )

    # Build function to retrieve cost
    get_cost = theano.function(
        [], cost,
        givens={
            X: X_shared,
            sigma: sigma_shared,
            Y: Y_shared,
            Adj: Adj_shared,
            l_kl: l_kl_shared,
            l_e: l_e_shared,
            l_c: l_c_shared,
            l_r: l_r_shared
        }
    )

    # Build function to retrieve per-vertex cost
    get_costs = theano.function(
        [], costs,
        givens={
            X: X_shared,
            sigma: sigma_shared,
            Y: Y_shared,
            Adj: Adj_shared,
            l_kl: l_kl_shared,
            l_e: l_e_shared,
            l_c: l_c_shared,
            l_r: l_r_shared
        }
    )

    # Optimization loop
    for epoch in range(n_epochs):

        # Switch parameter if a switching point is reached.
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)
        if epoch == l_kl_switch:
            l_kl_shared.set_value(final_l_kl)
        if epoch == l_e_switch:
            l_e_shared.set_value(final_l_e)
        if epoch == l_c_switch:
            l_c_shared.set_value(final_l_c)
        if epoch == l_r_switch:
            l_r_shared.set_value(final_l_r)
            if final_l_r != 0:
                # Give a nudge to co-located vertices in the epoch before the
                # repulsion kicks in (otherwise they don't feel any).
                Y_shared.set_value(switch_shake(Y_shared.get_value()))

        # Do update step for velocity
        update_Yv()
        # Do a gradient descent step
        update_Y()

        c = get_cost()
        if np.isnan(float(c)):
            raise NaNException('Encountered NaN for cost.')

        if verbose:
            print('[tsne] Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)), end='\r')

        if output_folder is not None and g is not None and save_every is not None and epoch % save_every == 0:
            # Get per-vertex cost for colour-coding
            cs = get_costs()

            # Save a snapshot
            save_drawing(output_folder, g, Y_shared.get_value().T, 'tsne_snap_' + str(epoch).zfill(5), formats=['jpg'], verbose=False, edge_colors="rgb", draw_vertices=False, opacity=0.3)

    # Get per-vertex cost
    cs = get_costs()

    if verbose:
        print('\n[tsne] Done! ')

    return np.array(Y_shared.get_value()), cs


def tsne(X, perplexity=30, Y=None, output_dims=2, n_epochs=1000,
         initial_lr=10, final_lr=4, lr_switch=None, init_stdev=1e-4,
         sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
         momentum_switch=250,
         initial_l_kl=None, final_l_kl=None, l_kl_switch=None,
         initial_l_e=None, final_l_e=None, l_e_switch=None,
         initial_l_c=None, final_l_c=None, l_c_switch=None,
         initial_l_r=None, final_l_r=None, l_r_switch=None,
         r_eps=1, random_state=None, Adj=None, g=None,
         save_every=None, snaps_output_folder=None, verbose=1):
    random_state = check_random_state(random_state)

    N = X.shape[0]

    X_shared = theano.shared(np.asarray(X, dtype=floath))
    sigma_shared = theano.shared(np.ones(N, dtype=floath))

    if Y is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
    Y_shared = theano.shared(np.asarray(Y, dtype=floath))

    # Find sigmas to attain the given perplexity.
    find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, verbose)

    # Do the optimization to find Y (the vertex coordinates).
    Y, costs = find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
                      initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
                      final_momentum, momentum_switch,
                      initial_l_kl, final_l_kl, l_kl_switch,
                      initial_l_e, final_l_e, l_e_switch,
                      initial_l_c, final_l_c, l_c_switch,
                      initial_l_r, final_l_r, l_r_switch,
                      r_eps,
                      Adj, g, save_every,
                      snaps_output_folder, verbose)

    # Return the vertex coordinates and the per-vertex costs.
    return Y, costs
