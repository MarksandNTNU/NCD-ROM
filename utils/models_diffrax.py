import math 
import time 

import diffrax 
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr 
import jax.scipy as jsp

import matplotlib.pyplot as plt
import optax


def gelu(x):
    """GELU activation function (matches PyTorch's implementation)."""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3)))) 


class Func(eqx.Module):
    """Function network for Neural CDE."""
    
    mlp: eqx.nn.MLP
    data_size: int 
    hidden_size: int 

    def __init__(self, data_size, hidden_size, width_size, depth, activation, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size, 
            out_size=hidden_size*data_size,
            width_size=width_size, 
            depth=depth, 
            activation=activation,  
            key=key 
        )
    
    def __call__(self, t, y, args):
        """Compute ODE function."""
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    """Neural Controlled Differential Equation model."""
    
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear 
    decoder: eqx.nn.Sequential
    
    def __init__(self, data_size, hidden_size, width_size, depth, activation_cde, 
                 activation_decoder, output_size, decoder_sizes, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        dkey = jr.split(key, len(decoder_sizes))
        
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, 
                                  activation=activation_cde, final_activation=activation_cde, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, activation_cde, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, output_size, key=lkey)
        
        layers = []
        layers.append(eqx.nn.Linear(hidden_size, decoder_sizes[0], key=dkey[0]))
        layers.append(eqx.nn.Lambda(activation_decoder))
        
        for i in range(len(decoder_sizes) - 1):
            layers.append(eqx.nn.Linear(decoder_sizes[i], decoder_sizes[i+1], key=dkey[i]))
            layers.append(eqx.nn.Lambda(activation_decoder))
        
        layers.append(eqx.nn.Linear(decoder_sizes[-1], output_size, key=dkey[-1]))
        self.decoder = eqx.nn.Sequential(layers)
    
    def __call__(self, ts, coeffs):
        """Forward pass through Neural CDE.
        
        Args:
            ts: Time points
            coeffs: Hermite coefficients for interpolation
        
        Returns:
            Model prediction
        """
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.Tsit5()
        dt0 = None 
        y0 = self.initial(control.evaluate(ts[0]))
        saveat = diffrax.SaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            term, 
            solver, 
            ts[0], 
            ts[-1], 
            dt0, 
            y0, 
            max_steps=1000, 
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5), 
            saveat=saveat
        )
        final_hidden = solution.ys[-1] 
        prediction = self.decoder(final_hidden)
        return prediction


def dataloader(arrays, batch_size, *, key):
    """Create infinite dataloader with shuffling.
    
    Args:
        arrays: Tuple of arrays to load
        batch_size: Batch size
        key: JAX random key
    
    Yields:
        Batches of data
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indicies = jnp.arange(dataset_size)
    
    while True: 
        perm = jr.permutation(key, indicies)
        (key,) = jr.split(key, 1)
        start = 0 
        end = batch_size
        
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def prepare_data_CDE(X, Y):
    """Prepare data for Neural CDE.
    
    Args:
        X: Input data
        Y: Output data
    
    Returns:
        Dict with time points, outputs, and Hermite coefficients
    """
    ts = X[:,:, -1]
    ys = X[:,:,:-1]
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    _, _, data_size = ys.shape
    return {'ts': ts, 'Y': Y, 'coeffs': coeffs}, data_size


@eqx.filter_jit 
def loss_mse_CDE(model, ti, y_i, coeff_i):
    """Compute MSE loss for Neural CDE."""
    preds = jax.vmap(model)(ti, coeff_i)
    return jnp.mean((preds - y_i)**2)


@eqx.filter_jit 
def loss_rmsre_CDE(model, ti, y_i, coeff_i):
    """Compute relative MSE loss for Neural CDE."""
    preds = jax.vmap(model)(ti, coeff_i)
    return jnp.sqrt(jnp.mean((preds - y_i)**2)) / jnp.sqrt(jnp.mean((y_i**2)))


@eqx.filter_jit 
def make_step_CDE(model, data_i, opt_state, optim):
    """Single training step for Neural CDE."""
    ti, y_i, *coeff_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_CDE, has_aux=False)(model, ti, y_i, coeff_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state

class SHRED_diffrax(eqx.Module):
    """SHRED model with PyTorch-compatible initialization."""
    
    lstms: tuple
    decoder: eqx.nn.Sequential
    hidden_size: int
    hidden_layers: int

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        decoder_sizes=(350, 400),
        activation=jax.nn.relu,
        dropout=0.0,
        *,
        key,
        **kwargs,
    ):
        """Initialize SHRED model matching PyTorch exactly."""
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        
        keys = jr.split(key, hidden_layers + len(decoder_sizes) + 1)
        lstm_keys = keys[:hidden_layers]
        dec_keys = keys[hidden_layers:]
        
        # Create LSTM cells
        lstms = []
        for i in range(hidden_layers):
            lstm = CustomLSTMCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                key=lstm_keys[i],
            )
            lstms.append(lstm)
        self.lstms = tuple(lstms)

        # Create decoder (same as before)
        sizes = (hidden_size,) + tuple(decoder_sizes) + (output_size,)
        layers = []
        
        for i in range(len(sizes) - 1):
            in_features = sizes[i]
            out_features = sizes[i + 1]
            
            linear = eqx.nn.Linear(in_features, out_features, key=dec_keys[i])
            
            a = math.sqrt(5)
            bound = math.sqrt(6.0 / ((1.0 + a**2) * in_features))
            
            layer_key = jr.fold_in(dec_keys[i], i)
            new_weight = jr.uniform(layer_key, linear.weight.shape, minval=-bound, maxval=bound)
            new_bias = jnp.zeros_like(linear.bias)
            
            linear = eqx.tree_at(
                lambda m: (m.weight, m.bias),
                linear,
                (new_weight, new_bias)
            )
            
            layers.append(linear)
            
            if i != len(sizes) - 2:
                layers.append(eqx.nn.Lambda(activation))

        self.decoder = eqx.nn.Sequential(layers)

    def __call__(self, x):
        """Forward pass through SHRED model."""
        init_states = tuple(
            (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
            for _ in range(self.hidden_layers)
        )
        
        def step(states, x_t):
            new_states = []
            layer_input = x_t
            
            for lstm, (h, c) in zip(self.lstms, states):
                h, c = lstm(layer_input, (h, c))
                new_states.append((h, c))
                layer_input = h
            
            return tuple(new_states), layer_input
        
        final_states, outputs = jax.lax.scan(step, init_states, x)
        final_hidden = outputs[-1]
        return self.decoder(final_hidden)


class CustomLSTMCell(eqx.Module):
    """Custom LSTM cell matching PyTorch nn.LSTM exactly."""
    
    weight_ih: jnp.ndarray
    weight_hh: jnp.ndarray
    bias_ih: jnp.ndarray
    bias_hh: jnp.ndarray
    input_size: int
    hidden_size: int
    
    def __init__(self, input_size: int, hidden_size: int, *, key):
        """Initialize LSTM cell matching PyTorch nn.LSTM."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        key1, key2, key3, key4 = jr.split(key, 4)
        
        bound = jnp.sqrt(1.0 / hidden_size)
        
        self.weight_ih = jr.uniform(key1, shape=(4 * hidden_size, input_size), minval=-bound, maxval=bound)
        self.weight_hh = jr.uniform(key2, shape=(4 * hidden_size, hidden_size), minval=-bound, maxval=bound)
        
        # Initialize biases randomly
        self.bias_ih = jr.uniform(key3, shape=(4 * hidden_size,), minval=-bound, maxval=bound)
        self.bias_hh = jr.uniform(key4, shape=(4 * hidden_size,), minval=-bound, maxval=bound)
        
        # ✅ Set forget gate bias to 1.0 (CRITICAL!)
        self.bias_ih = self.bias_ih.at[hidden_size:2*hidden_size].set(1.0)
        self.bias_hh = self.bias_hh.at[hidden_size:2*hidden_size].set(1.0)
    
    def __call__(self, x, state):
        """Forward pass through LSTM cell."""
        h, c = state
        
        gates_ih = jnp.dot(self.weight_ih, x) + self.bias_ih
        gates_hh = jnp.dot(self.weight_hh, h) + self.bias_hh
        gates = gates_ih + gates_hh
        
        i, f, g, o = jnp.split(gates, 4)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        
        return h_new, c_new
@eqx.filter_jit 
def loss_mse_SHRED(model, S_i, y_i):
    """Compute MSE loss matching PyTorch behavior.
    
    Args:
        model: SHRED_diffrax model
        S_i: Input sensor data, shape (batch_size, seq_length, input_size)
        y_i: Target data, shape (batch_size, output_size)
    
    Returns:
        Mean squared error scalar
    """
    preds = jax.vmap(model)(S_i)  # (batch_size, output_size)
    per_sample_loss = jnp.sum((preds - y_i)**2, axis=-1)  # (batch_size,)
    return jnp.mean(per_sample_loss)


@eqx.filter_jit 
def loss_rmsre_SHRED(model, S_i, y_i):
    """Compute relative MSE loss.
    
    Args:
        model: SHRED_diffrax model
        S_i: Input sensor data, shape (batch_size, seq_length, input_size)
        y_i: Target data, shape (batch_size, output_size)
    
    Returns:
        Mean relative error scalar
    """
    preds = jax.vmap(model)(S_i)  # (batch_size, output_size)
    
    # Per-sample RMSE
    numerator = jnp.sqrt(jnp.sum((preds - y_i)**2, axis=-1))  # (batch_size,)
    denominator = jnp.sqrt(jnp.sum((y_i**2), axis=-1))  # (batch_size,)
    
    # Per-sample relative error with numerical stability
    per_sample_rmsre = numerator / (denominator + 1e-10)
    
    # Return mean over batch
    return jnp.mean(per_sample_rmsre)


@eqx.filter_jit 
def make_step_SHRED(model, data_i, opt_state, optim):
    """Single training step for SHRED.
    
    Args:
        model: SHRED_diffrax model
        data_i: Batch of (S_i, y_i) data
        opt_state: Optimizer state
        optim: Optimizer
    
    Returns:
        (loss, updated_model, updated_opt_state)
    """
    S_i, y_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_SHRED, has_aux=False)(model, S_i, y_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state

def fit_SHRED(model, train_data, valid_data, steps, batch_size, lr, seed=42, early_stopping=60):
    """Train SHRED model - FIXED to match PyTorch behavior."""
    
    key = jr.PRNGKey(seed)
    train_key = jr.split(key, 1)[0]
    
    # optim = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.adam(lr)
    # )
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    train_losses = []
    valid_losses = []
    best_loss = jnp.inf
    best_model = model
    early_stopping_counter = 0
    
    step_counter = 0
    
    # Training loop - ONE dataloader for training
    train_loader = dataloader((train_data['S_i'], train_data['Y']), batch_size, key=train_key)
    
    for data_train_i in train_loader:
        if step_counter >= steps:
            break
        
        # Training step on batch
        train_mse, model, opt_state = make_step_SHRED(model, data_train_i, opt_state, optim)
        train_losses.append(train_mse)
        
        # Validation on FULL dataset (not mini-batch!)
        valid_mse = loss_mse_SHRED(model, valid_data['S_i'], valid_data['Y'])
        valid_losses.append(valid_mse)
        
        # Print
        if (step_counter + 1) % 10 == 0 or step_counter == 0:
            print(f"Step {step_counter:5d} | Train {train_mse:.4e} | Valid {valid_mse:.4e}")
        
        # Early stopping on FULL validation loss
        if valid_mse < best_loss:
            best_loss = valid_mse 
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1 
        
        if early_stopping_counter >= early_stopping:
            print(f'\n✓ Early stopping at step {step_counter}, best loss: {best_loss:.4e}')
            break
        
        step_counter += 1
    
    return best_model, jnp.array(train_losses), jnp.array(valid_losses)



import torch
import jax.numpy as jnp
import equinox as eqx

def transfer_weights_pytorch_to_jax(jax_model, pytorch_model):
    """
    Transfer weights from PyTorch SHRED model to JAX SHRED_diffrax model.
    
    Args:
        jax_model: SHRED_diffrax model instance
        pytorch_model: SHRED (PyTorch) model instance
    
    Returns:
        jax_model with weights transferred from pytorch_model
    """
    
    # Transfer LSTM layer 0
    jax_model = eqx.tree_at(
        lambda m: m.lstms[0].weight_ih,
        jax_model,
        jnp.array(pytorch_model.lstm.weight_ih_l0.detach().cpu().numpy())
    )
    jax_model = eqx.tree_at(
        lambda m: m.lstms[0].weight_hh,
        jax_model,
        jnp.array(pytorch_model.lstm.weight_hh_l0.detach().cpu().numpy())
    )
    jax_model = eqx.tree_at(
        lambda m: m.lstms[0].bias_ih,
        jax_model,
        jnp.array(pytorch_model.lstm.bias_ih_l0.detach().cpu().numpy())
    )
    jax_model = eqx.tree_at(
        lambda m: m.lstms[0].bias_hh,
        jax_model,
        jnp.array(pytorch_model.lstm.bias_hh_l0.detach().cpu().numpy())
    )
    
    # Transfer LSTM layer 1
    jax_model = eqx.tree_at(
        lambda m: m.lstms[1].weight_ih,
        jax_model,
        jnp.array(pytorch_model.lstm.weight_ih_l1.detach().cpu().numpy())
    )
    jax_model = eqx.tree_at(
        lambda m: m.lstms[1].weight_hh,
        jax_model,
        jnp.array(pytorch_model.lstm.weight_hh_l1.detach().cpu().numpy())
    )
    jax_model = eqx.tree_at(
        lambda m: m.lstms[1].bias_ih,
        jax_model,
        jnp.array(pytorch_model.lstm.bias_ih_l1.detach().cpu().numpy())
    )
    jax_model = eqx.tree_at(
        lambda m: m.lstms[1].bias_hh,
        jax_model,
        jnp.array(pytorch_model.lstm.bias_hh_l1.detach().cpu().numpy())
    )
    
    # Transfer decoder - iterate through Linear layers only
    linear_count = 0
    for i, layer in enumerate(jax_model.decoder):
        if isinstance(layer, eqx.nn.Linear):
            # Find corresponding PyTorch linear layer
            pytorch_linear_count = 0
            pytorch_layer = None
            for pt_layer in pytorch_model.decoder:
                if isinstance(pt_layer, torch.nn.Linear):
                    if pytorch_linear_count == linear_count:
                        pytorch_layer = pt_layer
                        break
                    pytorch_linear_count += 1
            
            if pytorch_layer is not None:
                jax_model = eqx.tree_at(
                    lambda m, idx=i: m.decoder[idx].weight,
                    jax_model,
                    jnp.array(pytorch_layer.weight.detach().cpu().numpy())
                )
                jax_model = eqx.tree_at(
                    lambda m, idx=i: m.decoder[idx].bias,
                    jax_model,
                    jnp.array(pytorch_layer.bias.detach().cpu().numpy())
                )
                linear_count += 1
    
    return jax_model