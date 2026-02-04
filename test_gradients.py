#!/usr/bin/env python3
import jax
jax.config.update("jax_enable_x64", True)

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from utils.models_diffrax import SHRED_diffrax, loss_mse_SHRED
import numpy as np

# Load test data
Data = np.load('KuramotoSivashinsky_data.npz')
U = Data['u'][:100]
sensor_idx = [20, 50, 80]
S_i = U[:, :, sensor_idx]  # (batch=100, seq_len=201, input=3)

# Create output target (dummy)
y_i = np.random.randn(100, 20)

# Test model
model = SHRED_diffrax(3, 20, hidden_size=64, hidden_layers=2, 
                      decoder_sizes=[350, 400], key=jr.PRNGKey(0))

# Test loss forward pass
loss = loss_mse_SHRED(model, S_i[:4], y_i[:4])
print(f'Loss value: {loss:.6e}')

# Test gradient
from functools import partial
grad_fn = jax.grad(partial(loss_mse_SHRED, S_i=S_i[:4], y_i=y_i[:4]))
try:
    grads = grad_fn(model)
    print('✓ Gradients computed successfully')
    grads_filtered = eqx.filter(grads, eqx.is_inexact_array)
    grads_leaves = jax.tree_util.tree_leaves(grads_filtered)
    non_none = sum(1 for g in grads_leaves if g is not None)
    print(f'  Non-None gradients: {non_none}/{len(grads_leaves)}')
    
    # Check LSTM gradients
    if grads.lstms[0].weight_ih is not None:
        norm = jnp.linalg.norm(grads.lstms[0].weight_ih)
        print(f'  LSTM[0].weight_ih norm: {norm:.6e}')
    
except TypeError as e:
    print(f'✗ Gradient error: {e}')
