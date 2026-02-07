"""
Data processing utilities for Kuramoto-Sivashinsky equation.
Handles state computation, snapshot generation, POD, and dataset creation.
"""

import numpy as np
from tqdm import tqdm
import sys
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional


def compute_state(nu: float, omega: float, dt: float, T: float, 
                  L: float = 22, nstate: int = 100, nstepsoutput: int = 300) -> np.ndarray:
    """
    Compute the state trajectory for Kuramoto-Sivashinsky equation using ETDRK4.
    
    Args:
        nu: Viscosity parameter
        omega: Initial condition frequency
        dt: Time step
        T: Final time
        L: Domain length
        nstate: Number of spatial points
        nstepsoutput: Number of output steps
        
    Returns:
        u: State trajectory array of shape (nstepsoutput+1, nstate)
        
    Credits: https://github.com/E-Renshaw/kuramoto-sivashinsky
    """
    x = np.linspace(0, L, nstate)
    u0 = lambda x: np.cos(omega * 2 * np.pi * x / L) * (1 + np.sin(omega * 2 * np.pi * x / L))
    
    v = np.fft.fft(u0(x))
    
    # Scalars for the Exponential Time Differencing fourth-order Runge-Kutta (ETDRK4)
    domain_length = x[-1] - x[0]
    k = 2 * np.pi / domain_length * np.transpose(
        np.conj(np.concatenate((np.arange(0, nstate/2), np.array([0]), np.arange(-nstate/2+1, 0))))
    )
    l = k**2 - nu * k**4
    E = np.exp(dt * l)
    E2 = np.exp(dt * l / 2)
    
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt * np.transpose(np.repeat([l], M, axis=0)) + np.repeat([r], nstate, axis=0)
    Q = dt * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    
    u = np.array([u0(x)])
    ntimesteps = round(T / dt)
    freqoutput = round((T / nstepsoutput) / dt)
    g = -0.5j * k
    
    for i in tqdm(range(1, ntimesteps + 1), colour="cyan", file=sys.stdout, 
                  bar_format='Computing state |{bar}| {n}/{total} {elapsed}<{remaining}'):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if i % freqoutput == 0:
            un = np.real(np.fft.ifft(v))
            u = np.append(u, np.array([un]), axis=0)
    
    return u


def generate_snapshots(ntrajectories: int, params_range: List[Tuple[float, float]],
                       dt: float = 1e-2, T: float = 200.0, nstepsoutput: int = 200,
                       L: float = 22, nstate: int = 100, 
                       filename: str = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate snapshot data for multiple parameter configurations.
    
    Args:
        ntrajectories: Number of trajectories to generate
        params_range: List of (min, max) tuples for each parameter
        dt: Time step
        T: Final time
        nstepsoutput: Number of output steps
        L: Domain length
        nstate: Number of spatial points
        filename: If provided, save data to this file
        verbose: Print progress
        
    Returns:
        U: State trajectories of shape (ntrajectories, ntimes, nstate)
        MU: Parameter values of shape (ntrajectories, ntimes, nparams)
    """
    ntimes = nstepsoutput + 1
    nparams = len(params_range)
    
    U = np.zeros((ntrajectories, ntimes, nstate))
    MU = np.zeros((ntrajectories, ntimes, nparams))
    
    for i in range(ntrajectories):
        if verbose:
            print(f"Generating snapshot {i+1}/{ntrajectories}...")
        
        # Sample random parameters
        params = []
        for j, (pmin, pmax) in enumerate(params_range):
            params.append(np.random.uniform(pmin, pmax))
        
        # Compute state trajectory
        ut = compute_state(params[0], params[1], dt, T, L, nstate, nstepsoutput)
        U[i] = ut
        
        # Store parameters for each time step
        for j in range(ntimes):
            MU[i, j] = np.array(params)
    
    if filename is not None:
        np.savez(filename.replace(".npz", "") + ".npz", u=U, mu=MU)
        if verbose:
            print(f"Snapshots saved to {filename}")
    
    return U, MU


def load_snapshots(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load snapshot data from file.
    
    Args:
        filename: Path to npz file
        
    Returns:
        U: State trajectories
        MU: Parameter values
    """
    data = np.load(filename.replace(".npz", "") + ".npz")
    U = data["u"]
    MU = data["mu"]
    return U, MU


def create_train_valid_test_split(ntrajectories: int, train_ratio: float = 0.8, 
                                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation/test splits for trajectories.
    
    Args:
        ntrajectories: Total number of trajectories
        train_ratio: Ratio of training data
        seed: Random seed
        
    Returns:
        idx_train, idx_valid, idx_test: Index arrays
    """
    np.random.seed(seed)
    ntrain = round(train_ratio * ntrajectories)
    
    idx_train = np.random.choice(ntrajectories, size=ntrain, replace=False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask != 0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]
    
    return idx_train, idx_valid, idx_test


def compute_pod(U_flat: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Principal Orthogonal Decomposition.
    
    Args:
        U_flat: Flattened state array of shape (n_samples, nstate)
        k: Number of POD modes
        
    Returns:
        W, S, V: SVD components
    """
    W, S, V = randomized_svd(U_flat, n_components=k)
    return W, S, V


def apply_pod(U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply POD transformation and compute reconstruction.
    
    Args:
        U: State array of shape (n_samples, nstate)
        V: POD modes of shape (k, nstate)
        
    Returns:
        U_POD: POD coefficients of shape (n_samples, k)
        U_reconstructed: Reconstructed states of shape (n_samples, nstate)
    """
    U_POD = U @ V.T
    U_reconstructed = U_POD @ V
    return U_POD, U_reconstructed


def extract_sensor_data(U: np.ndarray, idx_sensors: List[int]) -> np.ndarray:
    """
    Extract sensor data from state trajectories.
    
    Args:
        U: State trajectories of shape (ntrajectories, ntimes, nstate) or (ntimes, nstate)
        idx_sensors: Indices of sensor locations
        
    Returns:
        sensors_data: Sensor readings
    """
    return U[..., idx_sensors]


def create_padding(data: np.ndarray, lag: int) -> np.ndarray:
    """
    Create sliding window padding for time series data.
    
    Args:
        data: Input data of shape (ntrajectories, ntimes, nfeatures) or (ntimes, nfeatures)
        lag: Window size
        
    Returns:
        padded: Padded data with sliding windows
    """
    if data.ndim == 2:
        # Single trajectory: (ntimes, nfeatures)
        ntimes, nfeatures = data.shape
        padded = np.zeros((ntimes, lag, nfeatures))
        for i in range(ntimes):
            start_idx = max(0, i - lag + 1)
            window = data[start_idx:i+1]
            # Pad with zeros if needed
            if len(window) < lag:
                padded[i, lag - len(window):] = window
            else:
                padded[i] = window
        return padded
    else:
        # Multiple trajectories: (ntrajectories, ntimes, nfeatures)
        ntrajectories, ntimes, nfeatures = data.shape
        padded = np.zeros((ntrajectories, ntimes, lag, nfeatures))
        for t in range(ntrajectories):
            padded[t] = create_padding(data[t], lag)
        # Flatten trajectory dimension
        return padded.reshape(-1, lag, nfeatures)


def mre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Relative Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean relative error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2)) / np.sqrt(np.mean(y_true**2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error as percentage.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE as percentage
    """
    return mre(y_true, y_pred) * 100


class KSDataset:
    """
    Dataset class for Kuramoto-Sivashinsky data.
    Handles loading, POD, scaling, and dataset creation.
    """
    
    def __init__(self, filename: str = None, U: np.ndarray = None, MU: np.ndarray = None,
                 ntrajectories: int = 100, train_ratio: float = 0.8,
                 kstate: int = 20, lag: int = 10, idx_sensors: List[int] = None,
                 L: float = 22, nstate: int = 100, dt: float = 1e-2, T: float = 200.0,
                 seed: int = 0):
        """
        Initialize KS dataset.
        
        Args:
            filename: Path to load data from (optional)
            U: State trajectories array (optional, if not loading from file)
            MU: Parameter values array (optional)
            ntrajectories: Number of trajectories to use
            train_ratio: Ratio of training data
            kstate: Number of POD modes
            lag: Lag for SHRED input
            idx_sensors: Sensor indices (default: [25, 50, 75])
            L: Domain length
            nstate: Number of spatial points
            dt: Time step
            T: Final time
            seed: Random seed
        """
        self.L = L
        self.nstate = nstate
        self.dt = dt
        self.T = T
        self.kstate = kstate
        self.lag = lag
        self.seed = seed
        self.ntrajectories = ntrajectories
        self.train_ratio = train_ratio
        
        self.x = np.linspace(0, L, nstate)
        
        if idx_sensors is None:
            idx_sensors = [25, 50, 75]
        self.idx_sensors = idx_sensors
        self.nsensors = len(idx_sensors)
        self.sensors_coordinates = self.x[idx_sensors]
        
        # Load or use provided data
        if filename is not None:
            self.U, self.MU = load_snapshots(filename)
        elif U is not None:
            self.U = U
            self.MU = MU
        else:
            raise ValueError("Must provide either filename or U array")
        
        # Use subset of trajectories
        self.U = self.U[:ntrajectories]
        if self.MU is not None:
            self.MU = self.MU[:ntrajectories]
        
        self.ntimes = self.U.shape[1]
        
        # Process data
        self._process_data()
    
    def _process_data(self):
        """Process data: split, POD, scale, extract sensors."""
        # Create splits
        self.idx_train, self.idx_valid, self.idx_test = create_train_valid_test_split(
            self.ntrajectories, self.train_ratio, self.seed
        )
        
        self.ntrain = len(self.idx_train)
        self.nvalid = len(self.idx_valid)
        self.ntest = len(self.idx_test)
        
        # Split data
        self.Utrain = self.U[self.idx_train]
        self.Uvalid = self.U[self.idx_valid]
        self.Utest = self.U[self.idx_test]
        
        if self.MU is not None:
            self.MUtrain = self.MU[self.idx_train]
            self.MUvalid = self.MU[self.idx_valid]
            self.MUtest = self.MU[self.idx_test]
        
        # Flatten for POD
        Utrain_flat = self.Utrain.reshape(-1, self.nstate)
        Uvalid_flat = self.Uvalid.reshape(-1, self.nstate)
        Utest_flat = self.Utest.reshape(-1, self.nstate)
        
        # Compute POD
        self.W, self.S, self.V = compute_pod(Utrain_flat, self.kstate)
        
        # Apply POD
        self.Utrain_POD_flat, self.Utrain_reconstructed_flat = apply_pod(Utrain_flat, self.V)
        self.Uvalid_POD_flat, self.Uvalid_reconstructed_flat = apply_pod(Uvalid_flat, self.V)
        self.Utest_POD_flat, self.Utest_reconstructed_flat = apply_pod(Utest_flat, self.V)
        
        # Scale POD coefficients
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.Utrain_POD_flat)
        
        self.Utrain_POD_scaled = self.scaler.transform(self.Utrain_POD_flat)
        self.Uvalid_POD_scaled = self.scaler.transform(self.Uvalid_POD_flat)
        self.Utest_POD_scaled = self.scaler.transform(self.Utest_POD_flat)
        
        # Reshape back to trajectory form
        self.Utrain_POD = self.Utrain_POD_scaled.reshape(self.ntrain, self.ntimes, self.kstate)
        self.Uvalid_POD = self.Uvalid_POD_scaled.reshape(self.nvalid, self.ntimes, self.kstate)
        self.Utest_POD = self.Utest_POD_scaled.reshape(self.ntest, self.ntimes, self.kstate)
        
        self.Utrain_reconstructed = self.Utrain_reconstructed_flat.reshape(self.ntrain, self.ntimes, self.nstate)
        self.Uvalid_reconstructed = self.Uvalid_reconstructed_flat.reshape(self.nvalid, self.ntimes, self.nstate)
        self.Utest_reconstructed = self.Utest_reconstructed_flat.reshape(self.ntest, self.ntimes, self.nstate)
        
        # Extract sensor data
        self.sensors_train = extract_sensor_data(self.Utrain, self.idx_sensors)
        self.sensors_valid = extract_sensor_data(self.Uvalid, self.idx_sensors)
        self.sensors_test = extract_sensor_data(self.Utest, self.idx_sensors)
    
    def get_shred_datasets(self) -> dict:
        """
        Get datasets for SHRED training.
        
        Returns:
            Dictionary with train, valid, test data
        """
        # Create padded inputs
        train_in = create_padding(self.sensors_train, self.lag)
        valid_in = create_padding(self.sensors_valid, self.lag)
        test_in = create_padding(self.sensors_test, self.lag)
        
        # Create outputs (POD coefficients)
        train_out = self.Utrain_POD.reshape(-1, self.kstate)
        valid_out = self.Uvalid_POD.reshape(-1, self.kstate)
        test_out = self.Utest_POD.reshape(-1, self.kstate)
        
        return {
            'train': {'S_i': train_in.astype(np.float32), 'Y': train_out.astype(np.float32)},
            'valid': {'S_i': valid_in.astype(np.float32), 'Y': valid_out.astype(np.float32)},
            'test': {'S_i': test_in.astype(np.float32), 'Y': test_out.astype(np.float32)}
        }
    
    def inverse_transform_pod(self, Y_pod: np.ndarray) -> np.ndarray:
        """
        Transform POD coefficients back to full state.
        
        Args:
            Y_pod: POD coefficients (scaled)
            
        Returns:
            Full state reconstruction
        """
        # Inverse scale
        Y_unscaled = self.scaler.inverse_transform(Y_pod)
        # Inverse POD
        return Y_unscaled @ self.V
    
    def get_pod_errors(self) -> dict:
        """
        Compute POD reconstruction errors.
        
        Returns:
            Dictionary with train, valid, test errors
        """
        return {
            'train': rmse(self.Utrain.reshape(-1, self.nstate), 
                         self.Utrain_reconstructed.reshape(-1, self.nstate)),
            'valid': rmse(self.Uvalid.reshape(-1, self.nstate), 
                         self.Uvalid_reconstructed.reshape(-1, self.nstate)),
            'test': rmse(self.Utest.reshape(-1, self.nstate), 
                        self.Utest_reconstructed.reshape(-1, self.nstate))
        }
    
    def get_cde_datasets(self) -> dict:
        """
        Get datasets for Neural CDE training.
        Creates sliding windows within each trajectory to maintain time continuity.
        
        Returns:
            Dictionary with train, valid, test data containing ts, ys, Y, and coeffs
        """
        import diffrax
        import jax
        
        def prepare_cde_data(sensors, pod_coeffs, lag):
            """Prepare CDE data for a set of trajectories."""
            ntrajectories = sensors.shape[0]
            ntimes = sensors.shape[1]
            nsensors = sensors.shape[2]
            kstate = pod_coeffs.shape[2]
            
            # Create normalized time for each trajectory
            time_normalized = np.linspace(0, 1, ntimes)
            
            # Build windows within each trajectory
            all_ts = []
            all_ys = []
            all_Y = []
            
            for traj_idx in range(ntrajectories):
                traj_sensors = sensors[traj_idx]  # (ntimes, nsensors)
                traj_pod = pod_coeffs[traj_idx]   # (ntimes, kstate)
                
                for i in range(lag, ntimes):
                    # Window of sensor data
                    window_sensors = traj_sensors[i-lag:i]  # (lag, nsensors)
                    window_times = time_normalized[i-lag:i]  # (lag,)
                    
                    all_ts.append(window_times)
                    all_ys.append(window_sensors)
                    all_Y.append(traj_pod[i])  # Target POD coefficients
            
            ts = np.array(all_ts, dtype=np.float32)
            ys = np.array(all_ys, dtype=np.float32)
            Y = np.array(all_Y, dtype=np.float32)
            
            # Compute Hermite coefficients for interpolation
            coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
            
            return {'ts': ts, 'ys': ys, 'Y': Y, 'coeffs': coeffs}
        
        train_cde = prepare_cde_data(self.sensors_train, self.Utrain_POD, self.lag)
        valid_cde = prepare_cde_data(self.sensors_valid, self.Uvalid_POD, self.lag)
        test_cde = prepare_cde_data(self.sensors_test, self.Utest_POD, self.lag)
        
        return {
            'train': train_cde,
            'valid': valid_cde,
            'test': test_cde
        }
