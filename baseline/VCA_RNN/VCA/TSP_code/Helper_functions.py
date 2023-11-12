import numpy as np
import tensorflow as tf


#######################################################################
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
######################################################################


# Loading Functions --------------------------
def TSP_energy_1D(samples):
    """
    samples: np.array of size (numsamples, N)
    """

    numsamples = samples.shape[0]
    N = samples.shape[1]
    energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N):
        energies += np.abs(samples[:,(i+1)%N]-samples[:,i])

    return energies/(N-1)

def TSP_energy_2D(coordinates, samples):
    """
    samples: np.array of size (numsamples, N)
    """

    numsamples = samples.shape[0]
    N = samples.shape[1]
    energies = np.zeros((numsamples), dtype = np.float64)

    for i,sample in enumerate(samples):
        energies[i] = np.sum(np.sqrt(np.sum(np.square(coordinates[np.roll(sample, -1)]-coordinates[sample]), axis = 1)))

    return energies


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Code inspired by Google OR Tools plot:
# https://github.com/wouterkool/attention-learn-to-route/blob/master/simple_tsp.ipynb
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def plot_tsp(xy, tour, ax1):
    """
    Plot the TSP tour on matplotlib axis ax1.
    """

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    length = np.sum(d)

    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color='blue')
    # Starting node
    ax1.scatter([xs[0]], [ys[0]], s=100, color='red')

    # Arcs
    qv = ax1.quiver(
        xs, ys, dx, dy,
        scale_units='xy',
        angles='xy',
        scale=1,
    )

    ax1.set_title('{} nodes, total length {:.2f}'.format(len(tour), length))


# # Loading Functions --------------------------
# def TSP_energy(coordinates, samples):
#     """
#     samples: np.array of size (numsamples, Nbinary), Number of cities is Nbinary+1
#     """
#
#     numsamples = samples.shape[0]
#     Nbinary = samples.shape[1]-1
#     samples = samples+1 #adding 1
#     energies = np.zeros((numsamples), dtype = np.float64)
#
#     energies += np.abs(samples[:,0]) #distance with respect to the origin x0 = 0 (we assume we start at x0 = 0 since it is a tour)
#     for i in range(Nbinary):
#         energies += np.abs(samples[:,(i+1)]-samples[:,i])
#     energies += np.abs(samples[:,Nbinary]) #distance with respect to the origin
#
#     return energies/Nbinary
