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

def initialize_lstm_weights(lstm_cell, key):
    """Initialize LSTM weights using PyTorch's LSTM initialization."""
    try:
        # Get dimensions
        input_size = lstm_cell.input_size
        hidden_size = lstm_cell.hidden_size
        
        # Xavier initialization for weight matrices (matches PyTorch)
        weight_scale_ih = jnp.sqrt(1.0 / input_size)
        weight_scale_hh = jnp.sqrt(1.0 / hidden_size)
        
        key1, key2, key3 = jr.split(key, 3)
        
        # Initialize input-hidden weights
        weight_ih = jr.uniform(key1, lstm_cell.weight_ih.shape, minval=-weight_scale_ih, maxval=weight_scale_ih)
        # Initialize hidden-hidden weights  
        weight_hh = jr.uniform(key2, lstm_cell.weight_hh.shape, minval=-weight_scale_hh, maxval=weight_scale_hh)
        
        # Initialize bias: set forget gate bias to 1 (matches PyTorch default)
        bias = lstm_cell.bias
        if bias is not None:
            # The forget gate bias is typically in the middle section for PyTorch LSTMs
            # Set it to 1 to help with vanishing gradients
            bias_new = bias.at[hidden_size:2*hidden_size].set(1.0)
            lstm_cell = eqx.tree_at(
                lambda m: (m.weight_ih, m.weight_hh, m.bias),
                lstm_cell,
                (weight_ih, weight_hh, bias_new)
            )
        else:
            lstm_cell = eqx.tree_at(
                lambda m: (m.weight_ih, m.weight_hh),
                lstm_cell,
                (weight_ih, weight_hh)
            )
    except Exception as e:
        # If attributes don't exist, return unchanged
        pass
    
    return lstm_cell

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
    return jnp.mean((preds - y_i)**2)

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



def fit_CDE(model, train_data, valid_data,  steps, batch_size, lr, seed = 42,  early_stopping = 60, warmup_steps = 100):
    key = jr.PRNGKey(seed)
    train_loader_key, valid_loader_key = jr.split(key, 2)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients by global L2 norm
        optax.adam(lr)  # Adam optimizer
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    train_losses = []
    valid_losses = []
    best_loss = jnp.inf
    early_stopping_counter = 0 
    for step, data_train_i, data_valid_i in zip(range(steps),  
                                                dataloader((train_data['ts'], train_data['Y']) + train_data['coeffs'], batch_size, key = train_loader_key), 
                                                dataloader((valid_data['ts'], valid_data['Y']) + valid_data['coeffs'], batch_size, key = valid_loader_key)):
        start = time.time()
        train_mse, model, opt_state = make_step_CDE(model, data_train_i, opt_state, optim)
        ti_valid, y_i_valid, *coeff_valid_i = data_valid_i
        valid_mse = loss_mse_CDE(model,ti_valid, y_i_valid, coeff_valid_i)
        end = time.time()
        train_losses.append(train_mse)
        valid_losses.append(valid_mse)

        print(f"Step {step:5d} | Train {train_mse:.4e} | Valid {valid_mse:.4e} | LR {lr:.3e} | Time {(end-start):.3f}s | Early stopping counter = {early_stopping_counter}", end='\r', flush=True)
        if valid_mse < best_loss:
            best_loss = valid_mse 
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter +=1 
        if early_stopping_counter >= early_stopping and steps >= warmup_steps:
            print(f'Early stopping at step {step}, with best validation loss {best_loss}')
            break 
    return best_model, train_losses, valid_losses


class SHRED(eqx.Module):
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
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        
        # Create stacked LSTM cells
        num_decoder_layers = len(decoder_sizes) + 1
        keys = jr.split(key, hidden_layers + num_decoder_layers + hidden_layers + num_decoder_layers)
        lstm_keys = keys[:hidden_layers]
        lstm_init_keys = keys[hidden_layers:hidden_layers + hidden_layers]
        dec_keys = keys[hidden_layers + hidden_layers:hidden_layers + hidden_layers + num_decoder_layers]
        weight_init_keys = keys[hidden_layers + hidden_layers + num_decoder_layers:]
        
        # Stack LSTM cells with proper initialization
        lstms = []
        for i in range(hidden_layers):
            lstm = eqx.nn.LSTMCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                key=lstm_keys[i],
            )
            # Apply Xavier initialization
            lstm = initialize_lstm_weights(lstm, lstm_init_keys[i])
            lstms.append(lstm)
        self.lstms = tuple(lstms)

        # Decoder with proper weight initialization and optional dropout
        sizes = (hidden_size,) + tuple(decoder_sizes) + (output_size,)
        layers = []
        for i in range(len(sizes) - 1):
            linear = eqx.nn.Linear(sizes[i], sizes[i + 1], key=dec_keys[i])
            
            # Initialize weights using Xavier/Glorot initialization
            # This matches PyTorch's default for Linear layers
            weight_scale = jnp.sqrt(6.0 / (sizes[i] + sizes[i+1]))
            linear = eqx.tree_at(
                lambda m: m.weight,
                linear,
                jr.uniform(weight_init_keys[i], linear.weight.shape, minval=-weight_scale, maxval=weight_scale)
            )
            
            layers.append(linear)
            # Add activation and dropout to all but the last layer
            if i != len(sizes) - 2:
                layers.append(eqx.nn.Lambda(activation))
                # Add dropout layer if dropout > 0
                if dropout > 0.0:
                    # Note: JAX dropout is stateful, so we create a no-op lambda for now
                    # Actual dropout will be applied during training in a modified loss function
                    pass

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
    return jnp.mean((preds-y_i)**2)


@eqx.filter_jit 
def loss_rmsre_SHRED(model, S_i, y_i):
    preds = jax.vmap(model)(S_i)
    return jnp.sqrt(jnp.sum((preds - y_i)**2))/ jnp.sqrt(jnp.sum((y_i**2)))

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



def fit_SHRED(model, train_data, valid_data,  steps, batch_size, lr, seed = 42,  early_stopping = 60):
    key = jr.PRNGKey(seed)
    train_loader_key, valid_loader_key = jr.split(key, 2)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients by global L2 norm
        optax.adam(lr)  # Adam optimizer
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    train_losses = []
    valid_losses = []
    best_loss = loss_mse_SHRED(model, valid_data['S_i'], valid_data['Y'])
    early_stopping_counter = 0 
    for step, data_train_i, data_valid_i in zip(range(steps),  
                                                dataloader((train_data['S_i'], train_data['Y']), batch_size, key = train_loader_key), 
                                                dataloader((valid_data['S_i'], valid_data['Y']), batch_size, key = valid_loader_key)):
        start = time.time()
        train_mse, model, opt_state = make_step_SHRED(model, data_train_i, opt_state, optim)
        S_i_valid, y_i_valid = data_valid_i
        valid_mse = loss_mse_SHRED(model,S_i_valid, y_i_valid)
        end = time.time()
        train_losses.append(train_mse)
        valid_losses.append(valid_mse)

        print(f"Step {step:5d} | Train {train_mse:.4e} | Valid {valid_mse:.4e} | LR {lr:.3e} | Time {(end-start):.3f}s | Early stopping counter = {early_stopping_counter}") #, end='\r', flush=True
        if valid_mse < best_loss:
            best_loss = valid_mse 
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter +=1 
        if early_stopping_counter >= early_stopping:
            print(f'Early stopping at step {step}, with best validation loss {best_loss}')
            break 
    return best_model, train_losses, valid_losses

    







