from IPython import display
from torch import nn , optim
import torch
import matplotlib.pyplot as plt

from typing import Callable,List,Tuple
from torch import Tensor
from torch.nn import Module

def plot_model(model, input_data, target, list_loss=None):
    """
    evaluate input by model and print it
    model:one linear model
    input_data: (*,1) tnesor
    target: (*,1) tnesor
    loss_function: loss function
    list_loss: list of prives loss to plot it
    """
    y = model(input_data)  # to plot it we need change output to regular tensor
    W, b = None, None
    if type(model) == nn.Linear:
        W = model.weight.data.tolist()  # get W and b as float
        b = model.bias.data.item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.7))
    ax1.scatter(input_data, target, label="Y")
    ax1.plot(input_data, y.detach(), label=f"{W=},{b=:.3f}")
    ax1.legend()

    if list_loss is not None:
        x_loss = range(len(list_loss))
        loss_label = 0 if not list_loss else list_loss[-1]
        ax2.plot(x_loss, list_loss, label=loss_label)
        ax2.legend()
    display.clear_output(wait=True)
    return fig


def plot_3d_model(model, input_data, target, list_loss=None):
    """
    evaluate input by model and print it
    model:one linear model
    input_data: (*,2) tnesor
    target: (*,1) tnesor
    loss_function: loss function
    list_loss: list of prives loss to plot it
    """
    x_grid, y_grid = torch.meshgrid(*input_data.T, indexing="xy")
    pairs = torch.cartesian_prod(*input_data.T)
    out = model(pairs).detach()
    z_grid = out.view(x_grid.shape)

    fig = plt.figure(figsize=(15, 7))
    ax1, ax2 = fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2)

    ax1.set_ylim(0, 15)
    ax1.set_xlim(0, 15)
    ax1.set_zlim(0, 15)

    ax1.scatter(*input_data.T, target, label="Y", color="green")
    ax1.plot_surface(x_grid, y_grid, z_grid)
    ax1.legend()

    if list_loss is not None:
        x_loss = range(len(list_loss))
        xlim = max(len(list_loss), 10)
        ylim = max(max(list_loss), 10)
        ax2.set_xlim(0, xlim)
        ax2.set_ylim(0, ylim)
        ax2.plot(x_loss, list_loss, label=f"{list_loss[-1]}")
        ax2.legend()
    return fig


def ploter(plot_func, model, x, target, op_list_loss=None):
    loss = [] if op_list_loss is None else op_list_loss

    def f():
        plot_func(model, x, target, loss)

    return f
