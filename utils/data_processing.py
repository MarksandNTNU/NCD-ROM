"""
Data processing utilities for GoPro video reconstruction.
Handles loading, preprocessing, and dataset creation for SHRED and Neural CDE models.
"""

import numpy as np
from PIL import Image
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import MinMaxScaler
import jax
import diffrax


def load_videos(data_dir: str, nvideos: int = 2):
    """
    Load GoPro videos from GIF files.
    
    Args:
        data_dir: Directory containing video files
        nvideos: Number of videos to load
        
    Returns:
        videos: Array of shape (nvideos, nframes, Lx, Ly)
        filenames: List of filenames
        nframes, Lx, Ly, nframe, nsnapshots: Video dimensions
    """
    filenames = [f'{data_dir}/GoPro_video{i}.gif' for i in range(1, nvideos + 1)]
    videos = []

    for filename in filenames:
        file = Image.open(filename)
        video = []
        try:
            while True:
                frame = file.convert('L')
                # Convert to numpy array and normalize to [0, 1]
                frame_arr = np.array(frame, dtype=np.float32) / 255.0
                video.append(frame_arr)
                file.seek(file.tell() + 1)
        except EOFError:
            pass
        videos.append(np.stack(video))

    videos = np.stack(videos)
    nframes, Lx, Ly = videos[0].shape
    nframe = Lx * Ly
    nsnapshots = nvideos * nframes
    
    return videos, filenames, nframes, Lx, Ly, nframe, nsnapshots


def get_video_timing(filename: str):
    """
    Extract timing information from a GIF file.
    
    Args:
        filename: Path to GIF file
        
    Returns:
        time: Cumulative time array in ms
        time_normalized: Normalized time array [0, 1]
        total_duration: Total duration in ms
    """
    total_duration = 0
    time = []
    with Image.open(filename) as im:
        for frame_index in range(im.n_frames):
            im.seek(frame_index)
            frame_delay = im.info['duration']
            total_duration += frame_delay
            time.append(frame_delay)
    
    time = np.cumsum(np.array(time))
    time_normalized = (time - np.min(time)) / np.max(time)
    
    return time, time_normalized, total_duration


def preprocess_videos(videos, threshold: float = 0.5):
    """
    Preprocess videos by thresholding and normalizing.
    
    Args:
        videos: Array of shape (nvideos, nframes, Lx, Ly)
        threshold: Threshold value for background removal
        
    Returns:
        Preprocessed videos array
    """
    videos = videos.copy()  # Don't modify original
    nvideos, nframes = videos.shape[0], videos.shape[1]
    
    for i in range(nvideos):
        for j in range(nframes):
            videos[i, j][videos[i, j] > threshold] = threshold
            vmin, vmax = videos[i, j].min(), videos[i, j].max()
            if vmax > vmin:
                videos[i, j] = (videos[i, j] - vmin) / (vmax - vmin)
    
    return videos


def create_train_valid_test_split(nsnapshots: int, train_ratio: float = 0.8, seed: int = 0):
    """
    Create train/validation/test splits.
    
    Args:
        nsnapshots: Total number of snapshots
        train_ratio: Ratio of training data
        seed: Random seed
        
    Returns:
        idx_train, idx_valid, idx_test: Index arrays
    """
    np.random.seed(seed)
    ntrain = round(train_ratio * nsnapshots)
    
    idx_train = np.random.choice(nsnapshots, size=ntrain, replace=False)
    mask = np.ones(nsnapshots)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, nsnapshots)[np.where(mask != 0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]
    
    return idx_train, idx_valid, idx_test


def compute_pod(videos, idx_train, k: int = 100):
    """
    Compute Principal Orthogonal Decomposition (POD).
    
    Args:
        videos: Flattened videos array (nsnapshots, nframe)
        idx_train: Training indices
        k: Number of POD modes
        
    Returns:
        U, S, V: SVD components
        videos_POD: POD coefficients
        videos_reconstructed: Reconstructed videos
    """
    U, S, V = randomized_svd(videos[idx_train], n_components=k)
    videos_POD = videos @ V.T
    videos_reconstructed = videos @ V.T @ V
    
    return U, S, V, videos_POD, videos_reconstructed


def extract_sensor_data(videos, Lx: int, Ly: int, nsensors: int = 3, 
                        time_normalized: np.ndarray = None, seed: int = None):
    """
    Extract sensor data from random locations in the videos.
    
    Args:
        videos: Videos array of shape (nvideos, nframes, nframe)
        Lx, Ly: Video dimensions
        nsensors: Number of sensors
        time_normalized: Normalized time array
        seed: Random seed for sensor placement
        
    Returns:
        sensors_data: Sensor readings (nvideos, nframes, nsensors+1)
        sensors_coordinates: Sensor locations (nsensors, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    nvideos, nframes, nframe = videos.shape
    
    sensors_coordinates = np.vstack((
        np.random.choice(Lx // 2, size=nsensors, replace=True) + Lx // 4,
        np.random.choice(Ly // 2, size=nsensors, replace=True) + Ly // 2
    )).T
    
    sensors_data = np.zeros((nvideos, nframes, nsensors + 1))
    
    for i in range(nvideos):
        for j in range(nframes):
            sensors_data[i, j, :-1] = videos[i, j].reshape(Lx, Ly)[
                sensors_coordinates[:, 0], sensors_coordinates[:, 1]
            ]
            if time_normalized is not None:
                sensors_data[i, j, -1] = time_normalized[j]
    
    return sensors_data, sensors_coordinates


def create_sliding_windows(sensors_data: np.ndarray, videos_flat: np.ndarray, 
                           time_normalized: np.ndarray, lag: int) -> dict:
    """
    Create sliding window datasets for state reconstruction.
    Each window of sensor data predicts the CURRENT frame (last frame in the window).
    
    Uses vectorized operations for speed.
    
    Args:
        sensors_data: Sensor readings, shape (nvideos, nframes, nsensors)
        videos_flat: Flattened video frames, shape (nsnapshots, nframe)
        time_normalized: Normalized timestamps, shape (nframes,)
        lag: Window length
        
    Returns:
        dict with:
            'S': Sensor windows (n_windows, lag, nsensors) - input for SHRED
            'ts': Time windows (n_windows, lag) - timestamps for CDE
            'Y': Target frames (n_windows, nframe) - current frame to predict
    """
    nvideos, nframes, nsensors = sensors_data.shape
    nframe = videos_flat.shape[1]
    n_windows_per_video = nframes - lag + 1
    n_total = nvideos * n_windows_per_video
    
    # Preallocate output arrays
    S = np.zeros((n_total, lag, nsensors), dtype=np.float32)
    ts = np.zeros((n_total, lag), dtype=np.float32)
    Y = np.zeros((n_total, nframe), dtype=np.float32)
    
    # Reshape videos for indexing
    videos_by_vid = videos_flat.reshape(nvideos, nframes, nframe)
    
    # Build windows using stride tricks for efficiency
    idx = 0
    for v in range(nvideos):
        for i in range(n_windows_per_video):
            S[idx] = sensors_data[v, i:i+lag]
            ts[idx] = time_normalized[i:i+lag]
            Y[idx] = videos_by_vid[v, i+lag-1]  # Current frame (end of window)
            idx += 1
    
    return {'S': S, 'ts': ts, 'Y': Y}


def prepare_cde_datasets(X_cde, Y_cde, ts_cde, ys_cde, V, 
                         train_ratio: float = 0.8, valid_ratio: float = 0.1):
    """
    Prepare train/valid/test datasets for Neural CDE training.
    
    Args:
        X_cde: Input windows
        Y_cde: Target frames
        ts_cde: Timestamps
        ys_cde: Sensor values
        V: POD basis
        train_ratio: Training data ratio
        valid_ratio: Validation data ratio
        
    Returns:
        train_data_cde, valid_data_cde, test_data_cde: Dataset dictionaries
        train_data_shred, valid_data_shred, test_data_shred: SHRED dataset dictionaries
        scaler: Fitted MinMaxScaler
    """
    n_samples = X_cde.shape[0]
    idx_train_cde = int(train_ratio * n_samples)
    idx_valid_cde = int((train_ratio + valid_ratio) * n_samples)
    
    # Prepare Y data with POD and scaling
    Y_cde_train = Y_cde[:idx_train_cde]
    Y_cde_train_POD = Y_cde_train @ V.T
    scaler = MinMaxScaler()
    scaler.fit(Y_cde_train_POD)
    Y_cde_train_POD = scaler.transform(Y_cde_train_POD)
    
    Y_cde_valid = Y_cde[idx_train_cde:idx_valid_cde]
    Y_cde_valid_POD = scaler.transform(Y_cde_valid @ V.T)
    
    Y_cde_test = Y_cde[idx_valid_cde:]
    Y_cde_test_POD = scaler.transform(Y_cde_test @ V.T)
    
    # Compute Hermite coefficients for CDE
    train_coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(
        ts_cde[:idx_train_cde], ys_cde[:idx_train_cde]
    )
    valid_coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(
        ts_cde[idx_train_cde:idx_valid_cde], ys_cde[idx_train_cde:idx_valid_cde]
    )
    test_coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(
        ts_cde[idx_valid_cde:], ys_cde[idx_valid_cde:]
    )
    
    # CDE datasets
    train_data_cde = {
        'ts': ts_cde[:idx_train_cde],
        'Y': Y_cde_train_POD,
        'coeffs': train_coeffs
    }
    valid_data_cde = {
        'ts': ts_cde[idx_train_cde:idx_valid_cde],
        'Y': Y_cde_valid_POD,
        'coeffs': valid_coeffs
    }
    test_data_cde = {
        'ts': ts_cde[idx_valid_cde:],
        'Y': Y_cde_test_POD,
        'coeffs': test_coeffs
    }
    
    # SHRED datasets
    train_data_shred = {
        'S_i': X_cde[:idx_train_cde, :, :-1],
        'Y': Y_cde_train_POD
    }
    valid_data_shred = {
        'S_i': X_cde[idx_train_cde:idx_valid_cde, :, :-1],
        'Y': Y_cde_valid_POD
    }
    test_data_shred = {
        'S_i': X_cde[idx_valid_cde:, :, :-1],
        'Y': Y_cde_test_POD
    }
    
    return (train_data_cde, valid_data_cde, test_data_cde,
            train_data_shred, valid_data_shred, test_data_shred,
            scaler, Y_cde_train, Y_cde_valid, Y_cde_test)


def rmse(datapred, datatrue):
    """Compute relative RMSE."""
    return np.sqrt(np.mean((datapred - datatrue)**2)) / np.sqrt(np.mean(datatrue**2))


def padding(data, lag):
    """
    Extract time-series of length equal to lag from longer time series in data.
    Uses zero-padding for early frames (original SHRED approach).
    
    Args:
        data: Input array of shape (nvideos, nframes, nsensors)
        lag: Window length
        
    Returns:
        Padded windows of shape (nvideos * nframes, lag, nsensors)
    """
    data_out = np.zeros((data.shape[0] * data.shape[1], lag, data.shape[2]))
    
    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                # Zero-pad early frames
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                # Full window
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]
    
    return data_out


class GoproDataset:
    """
    Complete dataset class for GoPro video reconstruction.
    Handles all data loading, preprocessing, and dataset creation.
    """
    
    def __init__(self, data_dir: str = 'data', nvideos: int = 2, nsensors: int = 3,
                 lag: int = 150, k: int = 100, threshold: float = 0.5, seed: int = 0):
        """
        Initialize and load all data.
        
        Args:
            data_dir: Directory containing video files
            nvideos: Number of videos to load
            nsensors: Number of sensors to use
            lag: Sliding window size
            k: Number of POD modes
            threshold: Preprocessing threshold
            seed: Random seed
        """
        self.data_dir = data_dir
        self.nvideos = nvideos
        self.nsensors = nsensors
        self.lag = lag
        self.k = k
        self.threshold = threshold
        self.seed = seed
        
        # Load and process data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess all data."""
        # Load videos
        self.videos, self.filenames, self.nframes, self.Lx, self.Ly, self.nframe, self.nsnapshots = \
            load_videos(self.data_dir, self.nvideos)
        
        # Get timing
        self.time, self.time_normalized, self.total_duration = \
            get_video_timing(self.filenames[0])
        
        # Preprocess
        self.videos = preprocess_videos(self.videos, self.threshold)
        
        # Flatten (already numpy arrays now)
        self.videos_flat = self.videos.reshape(self.nsnapshots, self.nframe)
        
        # Create splits for original SHRED approach
        self.idx_train, self.idx_valid, self.idx_test = \
            create_train_valid_test_split(self.nsnapshots, seed=self.seed)
        
        # Compute POD
        self.U, self.S, self.V, self.videos_POD, self.videos_reconstructed = \
            compute_pod(self.videos_flat, self.idx_train, self.k)
        
        # Reshape videos for sensor extraction
        videos_reshaped = self.videos_flat.reshape(self.nvideos, self.nframes, self.nframe)
        
        # Extract sensors
        self.sensors_data, self.sensors_coordinates = extract_sensor_data(
            videos_reshaped, self.Lx, self.Ly, self.nsensors, self.time_normalized, self.seed
        )
        
        # ========== Create sliding windows (shared by SHRED and CDE) ==========
        # No padding - only full windows. Each window predicts current frame.
        self.windows = create_sliding_windows(
            self.sensors_data[:, :, :-1],  # Exclude time column from sensors
            self.videos_flat,
            self.time_normalized,
            self.lag
        )
        
        n_samples = self.windows['Y'].shape[0]
        
        # Random split (80/10/10) for both SHRED and CDE
        np.random.seed(self.seed)
        ntrain = int(0.8 * n_samples)
        nvalid = int(0.1 * n_samples)
        
        # Create random indices
        all_idx = np.random.permutation(n_samples)
        self.idx_train_win = all_idx[:ntrain]
        self.idx_valid_win = all_idx[ntrain:ntrain+nvalid]
        self.idx_test_win = all_idx[ntrain+nvalid:]
        
        # Project targets to POD space and scale
        Y_POD = self.windows['Y'] @ self.V.T  # (n_samples, k)
        self.scaler = MinMaxScaler()
        self.scaler.fit(Y_POD[self.idx_train_win])
        Y_POD_scaled = self.scaler.transform(Y_POD)
        
        # ========== SHRED datasets (random split, no padding) ==========
        # SHRED uses sensor windows only (no time)
        self.train_data_shred = {
            'S_i': self.windows['S'][self.idx_train_win],
            'Y': Y_POD_scaled[self.idx_train_win]
        }
        self.valid_data_shred = {
            'S_i': self.windows['S'][self.idx_valid_win],
            'Y': Y_POD_scaled[self.idx_valid_win]
        }
        self.test_data_shred = {
            'S_i': self.windows['S'][self.idx_test_win],
            'Y': Y_POD_scaled[self.idx_test_win]
        }
        
        # Store raw Y for inverse transform
        self.Y_train_raw = self.windows['Y'][self.idx_train_win]
        self.Y_valid_raw = self.windows['Y'][self.idx_valid_win]
        self.Y_test_raw = self.windows['Y'][self.idx_test_win]
        
        # ========== CDE datasets (same random split, adds time + coeffs) ==========
        self._prepare_cde_datasets(Y_POD_scaled)
    
    def get_cde_data(self):
        """Return CDE training data."""
        return self.train_data_cde, self.valid_data_cde, self.test_data_cde
    
    def get_shred_data(self):
        """Return SHRED training data."""
        return self.train_data_shred, self.valid_data_shred, self.test_data_shred
    
    def _prepare_cde_datasets(self, Y_POD_scaled):
        """Prepare CDE datasets with same random split as SHRED."""
        ts = self.windows['ts']
        ys = self.windows['S']  # Sensor values for CDE interpolation
        
        # Compute Hermite coefficients for CDE (using indexed samples)
        train_coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(
            ts[self.idx_train_win], ys[self.idx_train_win]
        )
        valid_coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(
            ts[self.idx_valid_win], ys[self.idx_valid_win]
        )
        test_coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(
            ts[self.idx_test_win], ys[self.idx_test_win]
        )
        
        # CDE datasets (same random split as SHRED)
        self.train_data_cde = {
            'ts': ts[self.idx_train_win],
            'Y': Y_POD_scaled[self.idx_train_win],
            'coeffs': train_coeffs
        }
        self.valid_data_cde = {
            'ts': ts[self.idx_valid_win],
            'Y': Y_POD_scaled[self.idx_valid_win],
            'coeffs': valid_coeffs
        }
        self.test_data_cde = {
            'ts': ts[self.idx_test_win],
            'Y': Y_POD_scaled[self.idx_test_win],
            'coeffs': test_coeffs
        }
    
    def inverse_transform(self, y_hat):
        """Transform predictions back to original video space."""
        return np.dot(self.scaler.inverse_transform(y_hat), self.V)
    
    def compute_reconstruction_error(self):
        """Compute reconstruction errors."""
        errors = {
            'train_pod': rmse(self.videos_flat[self.idx_train], 
                             self.videos_reconstructed[self.idx_train]) * 100,
            'valid_pod': rmse(self.videos_flat[self.idx_valid], 
                             self.videos_reconstructed[self.idx_valid]) * 100,
            'test_pod': rmse(self.videos_flat[self.idx_test], 
                            self.videos_reconstructed[self.idx_test]) * 100,
        }
        return errors
    
    def get_config(self):
        """Return configuration dictionary for MLflow logging."""
        return {
            'nvideos': self.nvideos,
            'nsensors': self.nsensors,
            'lag': self.lag,
            'k': self.k,
            'threshold': self.threshold,
            'seed': self.seed,
            'nframes': self.nframes,
            'Lx': self.Lx,
            'Ly': self.Ly,
            'nframe': self.nframe,
            'nsnapshots': self.nsnapshots,
            'n_train_shred': self.train_data_shred['Y'].shape[0],
            'n_valid_shred': self.valid_data_shred['Y'].shape[0],
            'n_test_shred': self.test_data_shred['Y'].shape[0],
            'n_train_cde': self.train_data_cde['Y'].shape[0],
            'n_valid_cde': self.valid_data_cde['Y'].shape[0],
            'n_test_cde': self.test_data_cde['Y'].shape[0],
        }
