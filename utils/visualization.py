"""
Visualization utilities for GoPro video reconstruction.
Handles plotting and animation generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io
from IPython.display import Image as DisplayImage, display
import seaborn as sns


def plot_singular_values(S, k: int = None, figsize=(8, 5)):
    """
    Plot singular values decay.
    
    Args:
        S: Singular values array
        k: Number of modes to highlight
        figsize: Figure size
    """
    if k is None:
        k = len(S)
    
    teal = sns.light_palette("teal", 15)
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(S) + 1), S, color=teal[14], marker='s', markersize=5, linewidth=1)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if k <= len(S):
        plt.loglog(k, S[k-1], color=teal[14], marker='s', linestyle='--')
    plt.title("Singular Values Decay")
    plt.xlabel("Mode")
    plt.ylabel("Singular Value")
    plt.tight_layout()
    plt.show()


def plot_sensor_locations(video_frame, sensors_coordinates, Lx: int, Ly: int, 
                          figsize=(8, 6), cmap='gray'):
    """
    Plot a video frame with sensor locations marked.
    
    Args:
        video_frame: Single video frame (flattened or 2D)
        sensors_coordinates: Sensor locations (nsensors, 2)
        Lx, Ly: Video dimensions
        figsize: Figure size
        cmap: Colormap
    """
    plt.figure(figsize=figsize)
    frame = video_frame.reshape(Lx, Ly) if video_frame.ndim == 1 else video_frame
    plt.imshow(frame, cmap=cmap)
    plt.plot(sensors_coordinates[:, 1], sensors_coordinates[:, 0], 'rX', markersize=12, 
             markeredgewidth=2, label='Sensors')
    plt.colorbar(label='Intensity')
    plt.title('Video Frame with Sensor Locations')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, valid_losses, title='Training Curves', 
                         labels=None, figsize=(12, 6)):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: Training loss array (or list of arrays)
        valid_losses: Validation loss array (or list of arrays)
        title: Plot title
        labels: Labels for multiple curves
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if isinstance(train_losses, list) and isinstance(train_losses[0], (list, np.ndarray)):
        # Multiple models
        for i, (train, valid) in enumerate(zip(train_losses, valid_losses)):
            label = labels[i] if labels else f'Model {i+1}'
            ax.plot(train, label=f'{label} Train')
            ax.plot(valid, linestyle='--', label=f'{label} Valid')
    else:
        # Single model
        ax.plot(train_losses, label='Train')
        ax.plot(valid_losses, label='Valid')
    
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(original, predicted, Lx: int, Ly: int, idx: int = 0, 
                    input_data=None, figsize=(20, 6)):
    """
    Plot comparison of original vs predicted frames.
    
    Args:
        original: Original video frames
        predicted: Predicted video frames
        Lx, Ly: Video dimensions
        idx: Frame index to plot
        input_data: Optional input sensor data
        figsize: Figure size
    """
    ncols = 4 if input_data is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    
    col = 0
    if input_data is not None:
        axes[col].plot(input_data[idx])
        axes[col].set_title('Input Sensor Data')
        axes[col].grid(True)
        col += 1
    
    orig_frame = original[idx].reshape(Lx, Ly) if original[idx].ndim == 1 else original[idx]
    pred_frame = predicted[idx].reshape(Lx, Ly) if predicted[idx].ndim == 1 else predicted[idx]
    
    im0 = axes[col].imshow(orig_frame, cmap='viridis')
    axes[col].set_title(f'Original (frame {idx})')
    axes[col].axis('off')
    plt.colorbar(im0, ax=axes[col])
    
    im1 = axes[col+1].imshow(pred_frame, cmap='viridis')
    axes[col+1].set_title('Predicted')
    axes[col+1].axis('off')
    plt.colorbar(im1, ax=axes[col+1])
    
    diff = orig_frame - pred_frame
    im2 = axes[col+2].imshow(diff, cmap='bwr', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[col+2].set_title('Difference')
    axes[col+2].axis('off')
    plt.colorbar(im2, ax=axes[col+2])
    
    plt.tight_layout()
    plt.show()


def create_comparison_gif(original, predicted, Lx: int, Ly: int, 
                         save_path: str = 'comparison.gif',
                         start_frame: int = 0, n_frames: int = 100,
                         frame_duration: int = 80, cmap: str = 'bwr',
                         vmin: float = 0, vmax: float = 1,
                         show: bool = True, figsize=(12, 4)):
    """
    Create animated GIF comparing original vs predicted frames.
    
    Args:
        original: Original video frames (n_frames, nframe) or (n_frames, Lx, Ly)
        predicted: Predicted video frames
        Lx, Ly: Video dimensions
        save_path: Path to save GIF
        start_frame: Starting frame index
        n_frames: Number of frames in animation
        frame_duration: Duration per frame in ms
        cmap: Colormap
        vmin, vmax: Value range for colormap
        show: Whether to display the GIF
        figsize: Figure size
        
    Returns:
        Path to saved GIF
    """
    frames = []
    end_frame = min(start_frame + n_frames, len(original))
    
    for t in range(start_frame, end_frame):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        orig_frame = original[t].reshape(Lx, Ly) if original[t].ndim == 1 else original[t]
        pred_frame = predicted[t].reshape(Lx, Ly) if predicted[t].ndim == 1 else predicted[t]
        
        # Original
        axes[0].imshow(orig_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Original (frame {t})')
        axes[0].axis('off')
        
        # Predicted
        axes[1].imshow(pred_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted')
        axes[1].axis('off')
        
        # Difference
        diff = orig_frame - pred_frame
        axes[2].imshow(diff, cmap='bwr', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        axes[2].set_title('Difference')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        frame = PILImage.open(buf).convert('P')
        frames.append(frame)
        plt.close(fig)
    
    # Save GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )
    
    print(f"Saved GIF to {save_path}")
    
    if show:
        display(DisplayImage(filename=save_path))
    
    return save_path


def create_pod_coefficient_gif(input_data, predictions, targets, 
                               save_path: str = 'coefficients.gif',
                               start_frame: int = 0, n_frames: int = 100,
                               frame_duration: int = 80, show: bool = True,
                               figsize=(12, 4)):
    """
    Create animated GIF of POD coefficient predictions.
    
    Args:
        input_data: Input sensor data (n_samples, lag, nsensors)
        predictions: Predicted POD coefficients (n_samples, k)
        targets: Target POD coefficients (n_samples, k)
        save_path: Path to save GIF
        start_frame: Starting frame index
        n_frames: Number of frames
        frame_duration: Duration per frame in ms
        show: Whether to display the GIF
        figsize: Figure size
        
    Returns:
        Path to saved GIF
    """
    frames = []
    end_frame = min(start_frame + n_frames, len(predictions))
    
    for t in range(start_frame, end_frame):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Input
        axes[0].plot(input_data[t].flatten() if input_data.ndim > 2 else input_data[t])
        axes[0].set_title(f'Input (frame {t})')
        axes[0].grid(True)
        
        # Predictions vs targets
        axes[1].plot(predictions[t], 'b-', label='Predicted')
        axes[1].plot(targets[t], 'r-', alpha=0.7, label='Target')
        axes[1].set_title('POD Coefficients')
        axes[1].legend()
        axes[1].grid(True)
        
        # Difference
        diff = targets[t] - predictions[t]
        axes[2].plot(diff)
        axes[2].set_title('Difference')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        frame = PILImage.open(buf).convert('P')
        frames.append(frame)
        plt.close(fig)
    
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )
    
    print(f"Saved GIF to {save_path}")
    
    if show:
        display(DisplayImage(filename=save_path))
    
    return save_path


def plot_model_comparison(models_results: dict, figsize=(12, 6)):
    """
    Plot comparison of multiple models.
    
    Args:
        models_results: Dictionary with model names as keys and 
                       {'train_loss', 'valid_loss', 'test_rmse'} as values
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    for name, results in models_results.items():
        if 'train_loss' in results:
            axes[0].plot(results['train_loss'], label=f'{name} Train')
            axes[0].plot(results['valid_loss'], '--', label=f'{name} Valid')
    
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bar chart of test RMSE
    names = list(models_results.keys())
    rmses = [models_results[n].get('test_rmse', 0) for n in names]
    
    bars = axes[1].bar(names, rmses, color=sns.color_palette('viridis', len(names)))
    axes[1].set_ylabel('Test RMSE (%)')
    axes[1].set_title('Model Comparison - Test RMSE')
    
    for bar, rmse in zip(bars, rmses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rmse:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
