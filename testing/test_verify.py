import jax
import jax.numpy as jnp
import haiku as hk
import jax_verify
# from jax_verify import unjit_inputs
from jax_verify import IntervalBound
import functools
import time

def run_time (func, *args, **kwargs) :
    before = time.time()
    ret = func(*args, **kwargs)
    after = time.time()
    return ret, (after - before)

print(jax.devices())

def model_fn (x: jax.Array) -> jax.Array :
    net = hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(2)
    ])
    return net(x)

net_pre = hk.without_apply_rng(hk.transform(model_fn))
key = jax.random.PRNGKey(0)
x = jnp.array([1,2,3,4], dtype=jnp.float32)
eps = jnp.array([0.1,0.1,0.1,0.1], dtype=jnp.float32)
params = net_pre.init(key, x)

net = functools.partial(net_pre.apply, params)
net_jit = jax.jit(net, backend='gpu')
# net = functools.partial(net_pre.apply, params)
net_jit(x) # jit compile the network

res, runtime = run_time(net, x+eps)
print(res, 'in', runtime)

res, runtime = run_time(net_jit, x+eps)
print(res, 'in', runtime)

input_bounds = jax_verify.IntervalBound(x-eps, x+eps)
net_crown = functools.partial(jax_verify.backward_crown_bound_propagation, net, get_linear_relaxations=True)
# net_crown = functools.partial(jax_verify.interval_bound_propagation, net)

def net_crown_prejit (input_bounds) :
    input_bounds = IntervalBound.from_jittable(input_bounds)
    bounds, _ = net_crown(input_bounds)
    return bounds.lower, bounds.upper

net_crown_jit = jax.jit(net_crown_prejit, backend='gpu')

input_bounds_jittable = input_bounds.to_jittable()

net_crown_jit(input_bounds_jittable) # jit compile crown

ret, runtime = run_time(net_crown, input_bounds)
output_bounds = ret[0]
linear_relaxations = ret[1]
# print(linear_relaxations)
print(output_bounds.lower[0], output_bounds.upper[0], 'in', runtime)

output_bounds, runtime = run_time(net_crown_jit, input_bounds_jittable)
print(output_bounds[0][0], output_bounds[1][0], 'in', runtime)

