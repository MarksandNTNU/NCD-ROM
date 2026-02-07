"""
Visualization utilities for Kuramoto-Sivashinsky equation.
Handles plotting state trajectories, POD modes, animations, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from typing import List, Tuple, Optional, Callable
import io
from IPython.display import Image as DisplayImage, display, clear_output


# Try to import cmcrameri for colormap, fall back to viridis
try:
    import cmcrameri.cm as cmc
    DEFAULT_CMAP = cmc.vik
except ImportError:
    DEFAULT_CMAP = 'viridis'


def plot_state(u: np.ndarray, x: np.ndarray = None, T: float = 200.0,
               cmap=None, title: str = None, fontsize: int = None,
               ticks: bool = True, colorbar: bool = True, levels: int = 50,
               vmax: float = None, vmin: float = None, ax=None):
    """
    Plot state trajectory as contour plot.
    
    Args:
        u: State trajectory with dimension (ntimes, nstate)
        x: Spatial coordinates
        T: Final time
        cmap: Colormap
        title: Plot title
        fontsize: Font size
        ticks: Show ticks
        colorbar: Show colorbar
        levels: Number of contour levels
        vmax, vmin: Color limits
        ax: Matplotlib axes (optional)
    """
    if cmap is None:
        cmap = DEFAULT_CMAP
    
    nstate = u.shape[1] if u.ndim == 2 else u.shape[0]
    ntimes = u.shape[0] if u.ndim == 2 else 1
    
    if x is None:
        x = np.linspace(0, 22, nstate)  # Default domain
    
    times = np.linspace(0, T, ntimes)
    timesgrid, xgrid = np.meshgrid(times, x)
    
    if ax is None:
        fig = plt.contourf(timesgrid, xgrid, u.T, cmap=cmap, levels=levels, vmax=vmax, vmin=vmin)
        plt.xlabel(r"Time $t$", fontsize=fontsize)
        plt.ylabel(r"Space $x$", fontsize=fontsize)
        plt.xlim(times[0], times[-1])
        plt.ylim(x[0], x[-1])
        plt.title(title, fontsize=fontsize)
        
        if not ticks:
            plt.xticks([], [])
            plt.yticks([], [])
        
        if colorbar:
            plt.colorbar(fig)
    else:
        fig = ax.contourf(timesgrid, xgrid, u.T, cmap=cmap, levels=levels, vmax=vmax, vmin=vmin)
        ax.set_xlabel(r"Time $t$", fontsize=fontsize)
        ax.set_ylabel(r"Space $x$", fontsize=fontsize)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(x[0], x[-1])
        ax.set_title(title, fontsize=fontsize)
        
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])


def plot_mode(mode: np.ndarray, x: np.ndarray = None, title: str = None,
              color: str = "magenta", linewidth: int = 2, grid: bool = True):
    """
    Plot a POD mode.
    
    Args:
        mode: POD mode of shape (nstate,)
        x: Spatial coordinates
        title: Plot title
        color: Line color
        linewidth: Line width
        grid: Show grid
    """
    if x is None:
        x = np.linspace(0, 22, len(mode))
    
    plt.plot(x, mode, color=color, linewidth=linewidth)
    if grid:
        plt.grid()
    if title:
        plt.title(title)


def plot_singular_values(S: np.ndarray, k: int = None, figsize: Tuple = (8, 5)):
    """
    Plot singular values decay.
    
    Args:
        S: Singular values
        k: Number of modes to highlight
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    modes = np.arange(1, len(S) + 1)
    plt.semilogy(modes, S, 'o-', color='teal', markersize=5, linewidth=1)
    
    if k is not None:
        plt.axvline(x=k, color='red', linestyle='--', alpha=0.7, label=f'k={k}')
        plt.legend()
    
    plt.xlabel("Mode")
    plt.ylabel("Singular Value")
    plt.title("Singular Values Decay")
    plt.grid(True, alpha=0.3)


def plot_training_curves(train_losses: List[float], valid_losses: List[float],
                         title: str = "Training Curves", labels: Tuple[str, str] = None,
                         figsize: Tuple = (10, 6)):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: Training losses per epoch
        valid_losses: Validation losses per epoch
        title: Plot title
        labels: Tuple of (train_label, valid_label)
        figsize: Figure size
    """
    if labels is None:
        labels = ("Train Loss", "Validation Loss")
    
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label=labels[0], linewidth=2)
    plt.plot(valid_losses, label=labels[1], linewidth=2, linestyle='--')
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_comparison(train_losses1: List, valid_losses1: List,
                    train_losses2: List, valid_losses2: List,
                    labels: Tuple[str, str] = ("Model 1", "Model 2"),
                    figsize: Tuple = (12, 6)):
    """
    Plot training curves comparison between two models.
    
    Args:
        train_losses1, valid_losses1: First model losses
        train_losses2, valid_losses2: Second model losses
        labels: Model labels
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(train_losses1, label=f"{labels[0]} Train", linewidth=2)
    plt.plot(valid_losses1, label=f"{labels[0]} Valid", linewidth=2, linestyle='--')
    plt.plot(train_losses2, label=f"{labels[1]} Train", linewidth=2)
    plt.plot(valid_losses2, label=f"{labels[1]} Valid", linewidth=2, linestyle='--')
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)


def multiplot(plotlist: List, plot_func: Callable, titles: Tuple[str] = None,
              fontsize: int = 12, figsize: Tuple = (15, 5), axis: bool = True):
    """
    Create multiple subplots.
    
    Args:
        plotlist: List of data to plot
        plot_func: Function to call for each plot
        titles: Tuple of subplot titles
        fontsize: Font size
        figsize: Figure size
        axis: Show axes
    """
    n = len(plotlist)
    plt.figure(figsize=figsize)
    
    for i, data in enumerate(plotlist):
        plt.subplot(1, n, i + 1)
        plot_func(data)
        if titles is not None and i < len(titles):
            plt.title(titles[i], fontsize=fontsize)
        if not axis:
            plt.axis('off')
    
    plt.tight_layout()


def create_trajectory_animation(u: np.ndarray, x: np.ndarray = None,
                                sensors_coordinates: np.ndarray = None,
                                T: float = 200.0, step: int = 5,
                                cmap=None, title: str = None,
                                filename: str = None, fps: int = 10,
                                figsize: Tuple = (8, 4)):
    """
    Create animated GIF of state trajectory evolution.
    
    Args:
        u: State trajectory (ntimes, nstate)
        x: Spatial coordinates
        sensors_coordinates: Sensor locations to mark
        T: Final time
        step: Frame step
        cmap: Colormap
        title: Animation title
        filename: Output filename (if None, returns frames)
        fps: Frames per second
        figsize: Figure size
        
    Returns:
        Animated GIF displayed in notebook
    """
    if cmap is None:
        cmap = DEFAULT_CMAP
    
    nstate = u.shape[1]
    ntimes = u.shape[0]
    
    if x is None:
        x = np.linspace(0, 22, nstate)
    
    times = np.linspace(0, T, ntimes)
    vmin, vmax = u.min(), u.max()
    
    frames = []
    
    # Start from at least 2 frames to avoid contour issues
    for i in range(max(2, step), ntimes + step, step):
        idx = min(i, ntimes)
        
        fig, ax = plt.subplots(figsize=figsize)
        timesgrid, xgrid = np.meshgrid(times[:idx], x)
        
        cf = ax.contourf(timesgrid, xgrid, u[:idx].T, cmap=cmap, 
                         levels=50, vmin=vmin, vmax=vmax)
        
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Space $x$")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(x[0], x[-1])
        if title:
            ax.set_title(title)
        
        # Plot sensors
        if sensors_coordinates is not None and i < ntimes:
            for k, sc in enumerate(sensors_coordinates):
                ax.plot(times[idx-1], sc, 'o', mfc='magenta', mec='black', 
                        ms=10, mew=1.5)
        
        plt.tight_layout()
        
        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)
    
    if filename:
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Saved animation to {filename}")
        return DisplayImage(filename=filename)
    else:
        return frames


def create_comparison_animation(uts: List[np.ndarray], x: np.ndarray = None,
                                 sensors_coordinates: np.ndarray = None,
                                 T: float = 200.0, step: int = 5,
                                 titles: List[str] = None,
                                 filename: str = None, fps: int = 10,
                                 figsize: Tuple = (18, 4)):
    """
    Create animated GIF comparing multiple trajectories.
    
    Args:
        uts: List of trajectories to compare
        x: Spatial coordinates
        sensors_coordinates: Sensor locations
        T: Final time
        step: Frame step
        titles: List of subplot titles
        filename: Output filename
        fps: Frames per second
        figsize: Figure size
        
    Returns:
        Animated GIF displayed in notebook
    """
    cmap = DEFAULT_CMAP
    n_plots = len(uts)
    
    nstate = uts[0].shape[1]
    ntimes = uts[0].shape[0]
    
    if x is None:
        x = np.linspace(0, 22, nstate)
    
    times = np.linspace(0, T, ntimes)
    
    # Compute global vmin/vmax
    vmin = min(u.min() for u in uts)
    vmax = max(u.max() for u in uts)
    
    frames = []
    
    # Start from at least 2 frames to avoid contour issues
    for i in range(max(2, step), ntimes + step, step):
        idx = min(i, ntimes)
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        for j, (ax, u) in enumerate(zip(axes, uts)):
            timesgrid, xgrid = np.meshgrid(times[:idx], x)
            
            cf = ax.contourf(timesgrid, xgrid, u[:idx].T, cmap=cmap,
                             levels=50, vmin=vmin, vmax=vmax)
            
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(r"Space $x$")
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(x[0], x[-1])
            
            if titles is not None and j < len(titles):
                ax.set_title(titles[j])
            
            # Plot sensors (except for last plot which is usually error)
            if sensors_coordinates is not None and i < ntimes and j < n_plots - 1:
                for k, sc in enumerate(sensors_coordinates):
                    ax.plot(times[idx-1], sc, 'o', mfc='magenta', mec='black',
                            ms=8, mew=1.5)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)
    
    if filename:
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Saved animation to {filename}")
        return DisplayImage(filename=filename)
    else:
        return frames


def plot_model_comparison(test_rmse1: float, test_rmse2: float,
                          labels: Tuple[str, str] = ("SHRED PyTorch", "SHRED JAX"),
                          figsize: Tuple = (8, 5)):
    """
    Plot bar chart comparing model performance.
    
    Args:
        test_rmse1, test_rmse2: Test RMSE values
        labels: Model labels
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['steelblue', 'seagreen']
    bars = ax.bar(labels, [test_rmse1, test_rmse2], color=colors)
    
    ax.set_ylabel('Test RMSE (%)')
    ax.set_title('Model Comparison - Test RMSE')
    
    for bar, val in zip(bars, [test_rmse1, test_rmse2]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', fontsize=12)


def plot_snapshot_comparison(u_true: np.ndarray, u_pred: np.ndarray,
                              x: np.ndarray = None, T: float = 200.0,
                              titles: Tuple = ("True", "Predicted", "Error"),
                              figsize: Tuple = (18, 4)):
    """
    Plot comparison of true vs predicted trajectory with error.
    
    Args:
        u_true: True trajectory (ntimes, nstate)
        u_pred: Predicted trajectory (ntimes, nstate)
        x: Spatial coordinates
        T: Final time
        titles: Subplot titles
        figsize: Figure size
    """
    error = np.sqrt((u_true - u_pred)**2)
    
    vmin = min(u_true.min(), u_pred.min())
    vmax = max(u_true.max(), u_pred.max())
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for ax, u, title in zip(axes, [u_true, u_pred, error], titles):
        if title == titles[2]:  # Error plot
            plot_state(u, x=x, T=T, title=title, colorbar=False, ax=ax)
        else:
            plot_state(u, x=x, T=T, title=title, vmin=vmin, vmax=vmax, 
                       colorbar=False, ax=ax)
    
    plt.tight_layout()
