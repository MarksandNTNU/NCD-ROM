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
import jax.nn.initializers as init



def gelu(x):
    """GELU activation function (matches PyTorch's default)."""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3)))) 


class Func(eqx.Module):

    mlp: eqx.nn.MLP
    data_size: int 
    hidden_size: int 

    def __init__(self, data_size, hidden_size, width_size, depth, activation, *, key,  **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size = hidden_size, 
            out_size = hidden_size*data_size,
            width_size = width_size, 
            depth = depth, 
            activation = activation,  
            key = key 
        )
    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)

class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear 
    decoder: eqx.nn.Sequential
    
    def __init__(self, data_size, hidden_size, width_size, depth, activation_cde, activation_decoder, output_size, decoder_sizes, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        dkey = jr.split(key, len(decoder_sizes))
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, activation = activation_cde, final_activation = activation_cde, key = ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, activation_cde, key = fkey)
        self.linear = eqx.nn.Linear(hidden_size, output_size, key = lkey)
        layers = []
        layers.append(eqx.nn.Linear(hidden_size, decoder_sizes[0], key = dkey[0]))
        layers.append(eqx.nn.Lambda(activation_decoder))
        for i in range(len(decoder_sizes) -1):
            layers.append(eqx.nn.Linear(decoder_sizes[i], decoder_sizes[i+1], key = dkey[i]))
            layers.append(eqx.nn.Lambda(activation_decoder))
        layers.append(eqx.nn.Linear(decoder_sizes[-1], output_size, key=dkey[-1]))
        self.decoder = eqx.nn.Sequential(layers)
    def __call__(self, ts, coeffs):
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
            max_steps= 1000, 
            stepsize_controller = diffrax.PIDController(rtol = 1e-3, atol = 1e-5), 
            saveat = saveat
        )
        final_hidden = solution.ys[-1] 
        # prediction = self.linear(final_hidden)
        prediction = self.decoder(final_hidden)
        return prediction

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    for array in arrays:
        print(array.shape[0], dataset_size)
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
    ts = X[:,:, -1]
    ys = X[:,:,:-1]
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    _, _, data_size = ys.shape
    return {'ts': ts, 'Y':Y, 'coeffs':coeffs}, data_size


@eqx.filter_jit 
def loss_mse_CDE(model, ti, y_i, coeff_i):
    preds = jax.vmap(model)(ti, coeff_i)

    return jnp.mean(jnp.sum((preds - y_i)**2, axis = -1))

@eqx.filter_jit 
def loss_rmsre_CDE(model, ti, y_i, coeff_i):
    preds = jax.vmap(model)(ti, coeff_i)
    return jnp.sqrt(jnp.mean((preds - y_i)**2))/ jnp.sqrt(jnp.mean((y_i**2)))



@eqx.filter_jit 
def make_step_CDE(model, data_i, opt_state, optim):
    ti, y_i, *coeff_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_CDE, has_aux=False)(model, ti, y_i, coeff_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state

def make_warmup_cosine_schedule(lr, warmup_steps, total_steps):
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=lr,
                transition_steps=warmup_steps,
            ),
            optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=total_steps - warmup_steps,
            ),
        ],
        boundaries=[warmup_steps],
    )

import optax

def make_warmup_piecewise_schedule_explicit(
    learning_rates,
    boundaries,
    warmup_steps,
):
    assert len(boundaries) == len(learning_rates), (
        "boundaries must have same length as learning_rates"
    )

    schedules = []


    schedules.append(
        optax.linear_schedule(init_value=0.0, end_value=learning_rates[0], transition_steps=warmup_steps))
    schedules += [optax.constant_schedule(lr) for lr in learning_rates]

    return optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries,
    )



def fit_CDE(model, train_data, valid_data, epochs, batch_size, lr, seed=42, early_stopping=20, verbose=True):
    """
    Epoch-based training for Neural CDE model (matches PyTorch behavior).
    
    Each epoch iterates through ALL batches in the training set,
    then evaluates on the FULL validation set once per epoch.
    """
    key = jr.PRNGKey(seed)
    
    # Setup optimizer (no gradient clipping to match PyTorch)
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    train_losses = []
    valid_losses = []
    
    # Get dataset info
    n_samples = train_data['ts'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    early_stopping_counter = 0
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        
        # Shuffle indices at the start of each epoch (like PyTorch DataLoader with shuffle=True)
        key, shuffle_key = jr.split(key)
        indices = jr.permutation(shuffle_key, n_samples)
        
        # Iterate through ALL batches in the dataset
        epoch_train_loss = 0.0
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            ts_batch = train_data['ts'][batch_indices]
            Y_batch = train_data['Y'][batch_indices]
            # Handle coefficients (tuple of arrays)
            coeffs_batch = tuple(c[batch_indices] for c in train_data['coeffs'])
            
            # Gradient step
            train_mse, model, opt_state = make_step_CDE(
                model, (ts_batch, Y_batch) + coeffs_batch, opt_state, optim
            )
            epoch_train_loss += float(train_mse)
        
        # Average training loss over batches
        epoch_train_loss /= n_batches
        
        # Evaluate on FULL validation set (once per epoch, like PyTorch)
        valid_mse = float(loss_mse_CDE(
            model, 
            valid_data['ts'], 
            valid_data['Y'], 
            valid_data['coeffs']
        ))
        
        end = time.time()
        
        train_losses.append(epoch_train_loss)
        valid_losses.append(valid_mse)
        
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_train_loss:.4e} | Valid {valid_mse:.4e} | "
                  f"Time {(end-start):.2f}s | Patience {early_stopping_counter}/{early_stopping}", end='\r')
        
        # Early stopping check
        if valid_mse < best_loss:
            best_loss = valid_mse
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}, best validation loss: {best_loss:.4e}")
            break
    
    if verbose and early_stopping_counter < early_stopping:
        print(f"\nTraining done: Train loss = {train_losses[-1]:.4e} | Valid loss = {valid_losses[-1]:.4e}")
    
    return best_model, train_losses, valid_losses


class SHRED(eqx.Module):
    """SHRED model with stacked LSTMs and decoder (matches PyTorch implementation)."""
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
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        
        # Split keys for LSTM layers and decoder layers
        num_decoder_layers = len(decoder_sizes) + 1
        keys = jr.split(key, hidden_layers + num_decoder_layers)
        lstm_keys = keys[:hidden_layers]
        dec_keys = keys[hidden_layers:]
        
        # Create stacked LSTM cells (like PyTorch nn.LSTM with num_layers)
        lstms = []
        for i in range(hidden_layers):
            lstm = eqx.nn.LSTMCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                key=lstm_keys[i],
            )
            lstms.append(lstm)
        self.lstms = tuple(lstms)

        # Build decoder: Linear -> Activation -> ... -> Linear
        sizes = (hidden_size,) + tuple(decoder_sizes) + (output_size,)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=dec_keys[i]))
            # Add activation to all but the last layer
            if i != len(sizes) - 2:
                layers.append(eqx.nn.Lambda(activation))

        self.decoder = eqx.nn.Sequential(layers)

    def __call__(self, x):
        """
        x shape: (seq_length, input_size)
        Returns: (output_size,)
        
        Uses jax.lax.scan to efficiently process sequences through stacked LSTMs.
        """
        # Initialize hidden states for all layers: tuple of (h, c) pairs
        init_states = tuple(
            (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
            for _ in range(self.hidden_layers)
        )
        
        def step(states, x_t):
            """
            Process one timestep through all stacked LSTM layers.
            
            Args:
                states: tuple of (h, c) pairs for each layer
                x_t: input at timestep t, shape (input_size,)
            
            Returns:
                new_states: updated (h, c) pairs
                output: output of final LSTM layer
            """
            new_states = []
            layer_input = x_t
            
            # Process through each LSTM layer
            for layer_idx, (lstm, (h, c)) in enumerate(zip(self.lstms, states)):
                h, c = lstm(layer_input, (h, c))
                new_states.append((h, c))
                # Output of this layer becomes input to next layer
                layer_input = h
            
            # Return updated states and the output of the final layer
            return tuple(new_states), layer_input
        
        # Scan over entire sequence
        final_states, outputs = jax.lax.scan(step, init_states, x)
        
        # Get the output from the final timestep of the final LSTM layer
        final_hidden = outputs[-1]
        
        # Pass through decoder
        return self.decoder(final_hidden)

@eqx.filter_jit 
def loss_mse_SHRED(model, S_i, y_i):
    preds = jax.vmap(model)(S_i)
    loss_by_batch = jnp.sum((preds - y_i)**2, axis = -1)
    return jnp.mean(loss_by_batch)


@eqx.filter_jit 
def loss_rmsre_SHRED(model, S_i, y_i):
    preds = jax.vmap(model)(S_i)
    return jnp.mean(jnp.sqrt(jnp.sum((preds - y_i)**2), axis = -1)/ jnp.sqrt(jnp.sum((y_i**2)), axis = -1))

@eqx.filter_jit 
def make_step_SHRED(model, data_i, opt_state, optim):
    S_i, y_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_SHRED, has_aux=False)(model, S_i, y_i)
    # Clip gradients by global norm to prevent exploding gradients
    grads = jax.tree_util.tree_map(lambda g: g, grads)  # No-op for now, will use optax chain
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state

def make_warmup_cosine_schedule(lr, warmup_steps, total_steps):
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=lr,
                transition_steps=warmup_steps,
            ),
            optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=total_steps - warmup_steps,
            ),
        ],
        boundaries=[warmup_steps],
    )

import optax

def make_warmup_piecewise_schedule_explicit(
    learning_rates,
    boundaries,
    warmup_steps,
):
    assert len(boundaries) == len(learning_rates), (
        "boundaries must have same length as learning_rates"
    )

    schedules = []


    schedules.append(
        optax.linear_schedule(init_value=0.0, end_value=learning_rates[0], transition_steps=warmup_steps))
    schedules += [optax.constant_schedule(lr) for lr in learning_rates]

    return optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries,
    )



def fit_SHRED(model, train_data, valid_data, epochs, batch_size, lr, seed=42, early_stopping=20, verbose=True):
    """
    Epoch-based training for SHRED model (matches PyTorch behavior).
    
    Each epoch iterates through ALL batches in the training set, 
    then evaluates on the FULL validation set once per epoch.
    """
    key = jr.PRNGKey(seed)
    
    # Setup optimizer (no gradient clipping to match PyTorch)
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    train_losses = []
    valid_losses = []
    
    # Get dataset info
    n_samples = train_data['S_i'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    early_stopping_counter = 0
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        
        # Shuffle indices at the start of each epoch (like PyTorch DataLoader with shuffle=True)
        key, shuffle_key = jr.split(key)
        indices = jr.permutation(shuffle_key, n_samples)
        
        # Iterate through ALL batches in the dataset
        epoch_train_loss = 0.0
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            S_i_batch = train_data['S_i'][batch_indices]
            Y_batch = train_data['Y'][batch_indices]
            
            # Gradient step
            train_mse, model, opt_state = make_step_SHRED(
                model, (S_i_batch, Y_batch), opt_state, optim
            )
            epoch_train_loss += float(train_mse)
        
        # Average training loss over batches
        epoch_train_loss /= n_batches
        
        # Evaluate on FULL validation set (once per epoch, like PyTorch)
        valid_mse = float(loss_mse_SHRED(model, valid_data['S_i'], valid_data['Y']))
        
        end = time.time()
        
        train_losses.append(epoch_train_loss)
        valid_losses.append(valid_mse)
        
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_train_loss:.4e} | Valid {valid_mse:.4e} | "
                  f"Time {(end-start):.2f}s | Patience {early_stopping_counter}/{early_stopping}", end='\r')
        
        # Early stopping check
        if valid_mse < best_loss:
            best_loss = valid_mse
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}, best validation loss: {best_loss:.4e}")
            break
    
    if verbose and early_stopping_counter < early_stopping:
        print(f"\nTraining done: Train loss = {train_losses[-1]:.4e} | Valid loss = {valid_losses[-1]:.4e}")
    
    return best_model, train_losses, valid_losses

    







