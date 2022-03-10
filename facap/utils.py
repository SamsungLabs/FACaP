import numpy as np

from matplotlib import pyplot as plt


def read_npy(path):
    paths = np.load(path, allow_pickle=True)

    edges = []

    for path in paths:
        ed = list(zip(path[1:].tolist(), path[:-1].tolist()))
        ed = [(tuple(i[0]), tuple(i[1])) for i in ed if i[0] != i[1]]
        edges.extend(ed)
    corners = [tuple(elem) for edge in edges for elem in edge]
    corners = set(corners)

    return corners, edges


def plot_graph(edges, color, name, markersize=20):
    for j, (st, end) in enumerate(edges):
        if j == 0:
            plt.plot([st[0], end[0]], [st[1], end[1]], f"{color}.-", label=name, markersize=markersize)
        else:
            plt.plot([st[0], end[0]], [st[1], end[1]], f"{color}.-", markersize=markersize)
