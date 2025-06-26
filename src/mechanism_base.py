import torch, einops, copy
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import torch.nn as nn

from utils import Config
from functools import partial     


########## Fourier Methods ##########
def normalize_to_pi(value): return (value + np.pi) % (2 * np.pi) - np.pi

def get_fourier_basis(p, device):
    """
    Generates the Fourier basis for a given dimensionality `p`.
    
    Args:
        p (int): The dimensionality of the Fourier basis.
        device (str): The device to place the Fourier basis tensor on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: A matrix where each row is a Fourier basis vector.
        list: A list of names corresponding to the Fourier basis vectors.
    """
    # Initialize the list to store Fourier basis vectors and names
    fourier_basis = []
    fourier_basis_names = []

    # Add the constant term (normalized)
    fourier_basis.append(torch.ones(p) / np.sqrt(p))
    fourier_basis_names.append('Const')

    # Generate Fourier basis for cosines and sines
    for i in range(1, p // 2 + 1):
        # Compute cosine and sine basis terms
        cosine = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        sine = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        # Normalize each basis function
        cosine /= cosine.norm()
        sine /= sine.norm()
        # Append basis vectors and their names
        fourier_basis.append(cosine)
        fourier_basis.append(sine)
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    
    # Special case for even p: cos(k*pi), alternating +1 and -1
    if p % 2 == 0:
        cosine = torch.cos(torch.pi * torch.arange(p))
        cosine /= cosine.norm()
        fourier_basis.append(cosine)
        fourier_basis_names.append(f'cos {p // 2}')
    
    # Stack the basis vectors into a matrix and move to the desired device
    fourier_basis = torch.stack(fourier_basis, dim=0).to(device)
    
    return fourier_basis, fourier_basis_names

def get_fourier_basis_unstd(p, device):
    """
    Generates the Fourier basis for a given dimensionality `p`.
    
    Args:
        p (int): The dimensionality of the Fourier basis.
        device (str): The device to place the Fourier basis tensor on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: A matrix where each row is a Fourier basis vector.
        list: A list of names corresponding to the Fourier basis vectors.
    """
    # Initialize the list to store Fourier basis vectors and names
    fourier_basis = []
    fourier_basis_names = []

    # Add the constant term (normalized)
    fourier_basis.append(torch.ones(p) / np.sqrt(p))
    fourier_basis_names.append('Const')

    # Generate Fourier basis for cosines and sines
    for i in range(1, p // 2 + 1):
        # Compute cosine and sine basis terms
        cosine = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        sine = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        # Append basis vectors and their names
        fourier_basis.append(cosine)
        fourier_basis.append(sine)
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    
    # Special case for even p: cos(k*pi), alternating +1 and -1
    if p % 2 == 0:
        cosine = torch.cos(torch.pi * torch.arange(p))
        cosine /= cosine.norm()
        fourier_basis.append(cosine)
        fourier_basis_names.append(f'cos {p // 2}')
    
    # Stack the basis vectors into a matrix and move to the desired device
    fourier_basis = torch.stack(fourier_basis, dim=0).to(device)
    
    return fourier_basis, fourier_basis_names

def fft1d(tensor, fourier_basis):
    # Converts a tensor with dimension p into the Fourier basis
    return tensor @ fourier_basis.T

def fft2d(mat, p, fourier_basis):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
    #fourier_mat = torch.einsum('xy,fX,FY->fFY', mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)

def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def unflatten_first(tensor, p):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor

def decode_weights(model_load, fourier_basis_unstd):
    """
    Decodes the weights using the given model and Fourier basis, and computes the maximum frequency list.

    Parameters:
        model_load (dict): A dictionary containing the model's weights.
        fourier_basis_unstd (torch.Tensor): The Fourier basis matrix.

    Returns:
        tuple: A tuple containing:
            - W_in_decode (torch.Tensor): Decoded weights for W_in.
            - W_out_decode (torch.Tensor): Decoded weights for W_out.
            - max_freq_ls (list): List of maximum frequencies derived from W_in_decode.
    """
    # Decode the weights
    W_in_decode = model_load['mlp.W_in'] @ fourier_basis_unstd.T
    W_out_decode = model_load['mlp.W_out'].T @ fourier_basis_unstd.T

    # Find the maximum frequency list
    max_ls = torch.argmax(abs(W_in_decode), dim=1)
    max_freq_ls = [(id.item() + 1) // 2 for id in max_ls]

    return W_in_decode, W_out_decode, max_freq_ls

def compute_neuron(neuron, max_freq_ls, W_decode):
    """
    Computes the scale and phase coefficients for a given neuron.

    Parameters:
        neuron (int): Index of the neuron to compute coefficients for.
        max_freq_ls (list): List of maximum frequencies derived from W_in_decode.
        W_in_decode (torch.Tensor): Decoded weights for W_in.

    Returns:
        tuple: A tuple containing:
            - coeff_in_scale (float): Scale coefficient.
            - coeff_in_phi (float): Phase coefficient.
    """
    if max_freq_ls[neuron] != 0:
        # Get the coefficients for the neuron
        neuron_coeff = W_decode[neuron, [max_freq_ls[neuron] * 2 - 1, max_freq_ls[neuron] * 2]]
        # Compute scale and phase
        coeff_scale = np.sqrt(torch.sum(neuron_coeff.pow(2)).item())
        coeff_phi = np.arctan2(-neuron_coeff[1].item(), neuron_coeff[0].item())
    else:
        # Default values if max frequency is zero
        coeff_phi = 0
        coeff_scale = W_decode[neuron, 0].item()

    return coeff_scale, coeff_phi

import torch

def decode_scales_phis(model_load: dict, fourier_basis_unstd: torch.Tensor):
    """
    Decode W_in into scale & phase for **all** frequencies.

    Returns:
      scales: Tensor[n_neurons, K+1]
      phis:   Tensor[n_neurons, K+1]
    """
    # 1) decode W_in
    W = model_load['mlp.W_in'] @ fourier_basis_unstd.T  # [n_neurons, p]
    W_out = model_load['mlp.W_out'].T @ fourier_basis_unstd.T  # [n_neurons, p]

    # 2) set up
    n_neurons, p = W.shape
    K = (p - 1) // 2

    scales = torch.zeros(n_neurons, K+1, device=W.device, dtype=W.dtype)
    phis   = torch.zeros(n_neurons, K+1, device=W.device, dtype=W.dtype)
    psis   = torch.zeros(n_neurons, K+1, device=W.device, dtype=W.dtype)

    # 3) DC (f=0)
    scales[:, 0] = W[:, 0].abs()
    # phis[:,0] stays 0

    # 4) all other freqs
    for f in range(1, K+1):
        real = W[:, 2*f - 1]
        imag = W[:, 2*f]
        scales[:, f] = torch.sqrt(real.pow(2) + imag.pow(2))
        phis[:,   f] = torch.atan2(-imag, real)
        psis[:,   f] = torch.atan2(-W_out[:, 2*f], W_out[:, 2*f - 1])

    return scales, phis, psis


########## Neuron Tracking ########## 
def sort_model(model_load, sort_order_mlp, sort_order_d):
    """
    Reorders the weights of a model based on the provided sorting orders.

    Parameters:
        model_load (dict): The original loaded model dictionary.
        sort_order_mlp (list or array): Sorting order for the MLP dimensions.
        sort_order_d (list or array): Sorting order for the embedding dimensions.

    Returns:
        dict: A deep copy of the reordered model.
    """
    # Create a deep copy of the model to avoid modifying the original
    sorted_model_load = copy.deepcopy(model_load)

    # Reorder MLP weights and biases
    sorted_model_load['mlp.W_in'] = sorted_model_load['mlp.W_in'][sort_order_mlp]
    sorted_model_load['mlp.W_in'] = sorted_model_load['mlp.W_in'][:, sort_order_d]
    sorted_model_load['mlp.W_out'] = sorted_model_load['mlp.W_out'][sort_order_d]
    sorted_model_load['mlp.W_out'] = sorted_model_load['mlp.W_out'][:, sort_order_mlp]
    sorted_model_load['mlp.b_in'] = sorted_model_load['mlp.b_in'][sort_order_mlp]

    # Reorder embedding weights
    sorted_model_load['embed.W_E'] = sorted_model_load['embed.W_E'][sort_order_d]
    sorted_model_load['unembed.embed_layer.W_E'] = sorted_model_load['embed.W_E']

    return sorted_model_load 

########## Plotting Helper ##########
def imshow(tensor, xaxis=None, yaxis=None, animation_name='Snapshot', **kwargs):
    if tensor.shape[0]==p*p:
        tensor = unflatten_first(tensor, p)
    tensor = torch.squeeze(tensor)
    px.imshow(to_numpy(tensor, flat=False), 
              labels={'x':xaxis, 'y':yaxis, 'animation_name':animation_name}, 
              **kwargs).show()

# Set default colour scheme
imshow = partial(imshow, color_continuous_scale='Blues')
# Creates good defaults for showing divergent colour scales (ie with both 
# positive and negative values, where 0 is white)
imshow_div = partial(imshow, color_continuous_scale='RdBu', color_continuous_midpoint=0.0)
# Presets a bunch of defaults to imshow to make it suitable for showing heatmaps 
# of activations with x axis being input 1 and y axis being input 2.

inputs_heatmap = partial(imshow, xaxis='Input 1', yaxis='Input 2', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, width=1000, height=800)

def line(x, y=None, hover=None, xaxis='', yaxis='', **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x=to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()

def scatter(x, y, **kwargs):
    px.scatter(x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), **kwargs).show()

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()
def line_marker(x, **kwargs):
    lines([x], mode='lines+markers', **kwargs)
def animate_lines(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, xaxis='x', yaxis='y', **kwargs):
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    px.line(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover,**kwargs).show()

def imshow_fourier(tensor, p, fourier_basis_names, title='', animation_name='snapshot', facet_labels=[], width=1000, height=800,  **kwargs):
    if tensor.shape[0] == p * p:
        tensor = unflatten_first(tensor, p)
    tensor = torch.squeeze(tensor)
    fig = px.imshow(
        to_numpy(tensor),
        x=fourier_basis_names,
        y=fourier_basis_names,
        labels={
            'x': 'x Component',
            'y': 'y Component',
            'animation_frame': animation_name
        },
        title=title,
        color_continuous_midpoint=0.,
        color_continuous_scale='RdBu',
        width=width,
        height=height,
        **kwargs
    )
    fig.update(data=[{'hovertemplate': "%{x}x * %{y}y<br>Value:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show()


def animate_multi_lines(lines_list, y_index=None, snapshot_index = None, snapshot='snapshot', hover=None, swap_y_animate=False, **kwargs):
    # Can plot an animation of lines with multiple lines on the plot.
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j])+[snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index+[snapshot, 'x'])
    px.line(df, x='x', y=y_index, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs).show()

def animate_scatter(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, yaxis='y', xaxis='x', color=None, color_name = 'color', **kwargs):
    # Can plot an animated scatter plot
    # lines_list has shape snapshot x 2 x line
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if type(color)==torch.Tensor:
        color = to_numpy(color)
    if len(color.shape)==1:
        color = einops.repeat(color, 'x -> snapshot x', snapshot=lines_list.shape[0])
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append([lines_list[i, 0, j].item(), lines_list[i, 1, j].item(), snapshot_index[i], color[i, j]])
    print([lines_list[:, 0].min(), lines_list[:, 0].max()])
    print([lines_list[:, 1].min(), lines_list[:, 1].max()])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    px.scatter(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()], range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()], hover_name=hover, color=color_name, **kwargs).sh

def plot_angles_on_circle(angles, multipliers = [1, 2, 4, 6], title_prefix="Angles Multiplication"):
    """
    Visualize multiple sets of angles (in radians) on unit circles.

    Parameters:
    - angles: list or array-like of angles in radians (should be in range [-π, π]).
    - title_prefix: Prefix for titles of the subplots (default is "Angles Multiplication").
    """

    # Create a figure with 4 subplots in a row
    plt.figure(figsize=(20, 5))

    # Loop through each multiplier to create the subplots
    for i, multiplier in enumerate(multipliers):
        # Multiply the angles
        modified_angles = angles * multiplier

        # Convert angles to x and y coordinates on a unit circle
        x = np.cos(modified_angles)
        y = np.sin(modified_angles)

        # Plot the unit circle
        theta = np.linspace(0, 2 * np.pi, 500)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)

        plt.subplot(1, 4, i + 1)
        plt.plot(circle_x, circle_y, color='lightgray', label='Unit Circle')  # Unit circle
        plt.scatter(x, y, color='red', label='Points')  # Points corresponding to the angles
        plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line
        plt.axvline(0, color='black', linewidth=0.5)  # Vertical line

        # Annotate each point with its angle
        for j, angle in enumerate(modified_angles):
            plt.text(x[j] * 1.1, y[j] * 1.1, f'{angle:.2f}', fontsize=9, ha='center')

        # Set title and formatting
        plt.title(f"{title_prefix}: {multiplier}*Angles")
        plt.axis('equal')  # Equal scaling for x and y
        plt.legend()
        plt.grid(True)

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

