import matplotlib.animation as animation
import torch 
import numpy as np
import matplotlib.pyplot as plt
import imageio
from IPython.display import clear_output as clc
from IPython.display import display


def create_animation(x, sensor_data, prediction, ground_truth, pixel_coordinates, nsensors = 3, model_name = 'SHRED ROM'):
    """
    Create animation with sensor measurements on left, model predictions vs ground truth in middle, and MSE over time on right
    """
    nt, nx = np.shape(ground_truth)
    # Convert sensor_data to numpy if it's a tensor
    if isinstance(sensor_data, torch.Tensor):
        sensor_data = sensor_data.detach().cpu().numpy()
    
    # Get the actual lag from sensor_data shape
    actual_lag = sensor_data.shape[1]  # Shape: (nt, lag, nsensors)
    
    # Pre-compute MSE for all time steps
    mse_over_time = np.mean((prediction - ground_truth)**2, axis=1)
    
    # Set up the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{model_name} Animation - Trajectory ', fontsize=14)
    
    # Left subplot: Sensor measurements over lag window
    ax1.set_title('Sensor Measurements (Lag Window)')
    ax1.set_xlabel('Lag Steps')
    ax1.set_ylabel('Sensor Value')
    ax1.grid(True)
    ax1.set_xlim(0, actual_lag-1)
    ax1.set_ylim(np.min(sensor_data[..., :-1]) * 1.1, np.max(sensor_data[..., :-1]) * 1.1)
    
    # Initialize sensor lines
    sensor_lines = []
    actual_nsensors = sensor_data.shape[2]  # Handle cases where we might have +1 for time
    for s in range(min(nsensors, actual_nsensors)):  # Use minimum to avoid index errors
        line, = ax1.plot([], [], label=f'Sensor {s+1}', linewidth=2)
        sensor_lines.append(line)
    ax1.legend()
    
    # Middle subplot: Spatial predictions
    ax2.set_title('Prediction vs Ground Truth')
    ax2.set_xlabel('Spatial coordinate x')
    ax2.set_ylabel('u(x,t)')
    ax2.grid(True)
    ax2.set_xlim(x[0], x[-1])
    y_min = min(np.min(prediction), np.min(ground_truth)) * 0.9
    y_max = max(np.max(prediction), np.max(ground_truth)) * 1.1
    ax2.set_ylim(y_min, y_max)
    
    # Initialize spatial lines
    pred_line, = ax2.plot([], [], 'bo', label='Model Prediction', linewidth=2)
    truth_line, = ax2.plot([], [], 'k--', label='Ground Truth', linewidth=2)
    sensor_points, = ax2.plot([], [], 'ko', markersize=8, label='Sensor Locations')
    ax2.legend()
    
    # Right subplot: MSE over time
    ax3.set_title('MSE over Time')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Mean Squared Error')
    ax3.grid(True)
    ax3.set_xlim(0, nt-1)
    ax3.set_ylim(0, np.max(mse_over_time) * 1.1)
    
    # Initialize MSE line
    mse_line, = ax3.plot([], [], 'r-', linewidth=2, label='MSE')
    current_point, = ax3.plot([], [], 'ro', markersize=8, label='Current')
    ax3.legend()
    
    def animate(frame_idx):
        """Animation function called for each frame"""
        if frame_idx < nt:
            # Update sensor measurements (left plot)
            lag_data = sensor_data[frame_idx]  # Shape: (lag, nsensors)
            for s in range(len(sensor_lines)):
                x_sensor = np.arange(actual_lag)
                y_sensor = lag_data[:, s]
                sensor_lines[s].set_data(x_sensor, y_sensor)
            
            # Update spatial predictions (middle plot)
            pred_line.set_data(x, prediction[frame_idx])
            truth_line.set_data(x, ground_truth[frame_idx])
            
            # Update sensor location markers
            sensor_x = x[pixel_coordinates]
            sensor_y = ground_truth[frame_idx, pixel_coordinates]
            sensor_points.set_data(sensor_x, sensor_y)
            
            # Update MSE plot (right plot)
            time_steps = np.arange(frame_idx + 1)
            mse_values = mse_over_time[:frame_idx + 1]
            mse_line.set_data(time_steps, mse_values)
            current_point.set_data([frame_idx], [mse_over_time[frame_idx]])
            
            # Update titles with time info
            ax1.set_title(f'Sensor Measurements (t={frame_idx}/{nt-1})')
            ax2.set_title(f'{model_name} Prediction vs Ground Truth (t={frame_idx}/{nt-1})')
            ax3.set_title(f'MSE over Time (Current MSE: {mse_over_time[frame_idx]:.6f})')
        
        return sensor_lines + [pred_line, truth_line, sensor_points, mse_line, current_point]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=nt, interval=200, blit=True, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return anim

def create_comparison_plot(x, sensor_data, prediction, ground_truth, pixel_coordinates, nsensors=3, model_name = 'SHRED ROM'):
    """
    Create a static plot showing sensor measurements, model prediction, and ground truth at a specific time.
    
    Parameters:
    x: spatial coordinates
    sensor_data: sensor measurements over lag period (shape: lag, nsensors or nsensors+1)  
    prediction: model prediction at specific time and trajectory (shape: nx)
    ground_truth: ground truth data at specific time and trajectory (shape: nx)
    pixel_coordinates: spatial locations of sensors
    nsensors: number of sensors
    """
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f'{model_name} Predictions vs Ground Truth and Sensor Measurements', fontsize=16)
    
    # Sensor measurements at current time
    axes[0].set_title('Sensor measurements')
    for i in range(nsensors):
        axes[0].plot(sensor_data[:, i], label=f'Sensor {i+1}')
    axes[0].legend(loc=0)
    axes[0].set_xlabel('Time steps (lag)')
    axes[0].set_ylabel('Sensor response')
    axes[0].grid(True)
    
    # Model prediction
    axes[1].set_title(f'{model_name} Prediction')
    axes[1].plot(x, prediction, label='Prediction')
    axes[1].plot(x[pixel_coordinates], prediction[pixel_coordinates], 'r.', markersize=10, label='Sensor locations')
    axes[1].legend(loc=0)
    axes[1].set_xlabel('Spatial coordinate x')
    axes[1].set_ylabel(r'prediction $\hat{u}(x,t)$')
    axes[1].grid(True)
    
    # Ground truth
    axes[2].set_title('Ground Truth')
    axes[2].plot(x, ground_truth, label='Ground Truth')
    axes[2].plot(x[pixel_coordinates], ground_truth[pixel_coordinates], 'r.', markersize=10, label='Sensor locations')
    axes[2].legend(loc=0)
    axes[2].set_xlabel('Spatial coordinate x')
    axes[2].set_ylabel(r'true $u(x,t)$')
    axes[2].grid(True)
    
    # Error
    axes[3].set_title('Error')
    error = prediction - ground_truth
    axes[3].plot(x, error, label='Error')
    axes[3].plot(x[pixel_coordinates], error[pixel_coordinates], 'r.', markersize=10, label='Sensor locations')
    axes[3].legend(loc=0)
    axes[3].set_xlabel('Spatial coordinate x')
    axes[3].set_ylabel('Error')
    axes[3].grid(True)
    
    plt.tight_layout()
    return fig, axes