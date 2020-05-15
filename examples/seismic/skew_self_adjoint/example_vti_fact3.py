import numpy as np
from sympy import sqrt

from devito import (Grid, Function, TimeFunction, Eq, Operator)
from examples.seismic import RickerSource, TimeAxis

space_order = 8
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 250.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

# shape = (601, 601, 301)
# shape = (801, 801, 401)
# shape = (1001, 1001, 501)
shape = (1201, 1201, 601)
spacing = (10.0, 10.0, 10.0)
origin = tuple([0.0 for s in shape])
extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
grid = Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)

b = Function(name='b', grid=grid, space_order=space_order)
f = Function(name='f', grid=grid, space_order=space_order)
vel0 = Function(name='vel0', grid=grid, space_order=space_order)
eps0 = Function(name='eps0', grid=vel0.grid, space_order=space_order)
eta0 = Function(name='eta0', grid=vel0.grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=vel0.grid, space_order=space_order)

_b = 1.0
_f = 0.84
_eps = 0.2
_eta = 0.4

b.data[:] = _b
f.data[:] = _f
vel0.data[:] = 1.5
eps0.data[:] = _eps
eta0.data[:] = _eta
wOverQ.data[:] = 1.0

t0 = 0.0
t1 = 250.0
dt = 1.0
time_axis = TimeAxis(start=t0, stop=t1, step=dt)

p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = p0.dimensions

src_coords = np.empty((1, len(shape)), dtype=dtype)
src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=vel0.grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]
src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * vel0**2 / b)


def g1(field):
    return field.dx(x0=x+x.spacing/2)


def g2(field):
    return field.dy(x0=y+y.spacing/2)


def g3(field):
    return field.dz(x0=z+z.spacing/2)


def g1_tilde(field):
    return field.dx(x0=x-x.spacing/2)


def g2_tilde(field):
    return field.dy(x0=y-y.spacing/2)


def g3_tilde(field):
    return field.dz(x0=z-z.spacing/2)


# Functions  for additional factorization
b1mf = Function(name='b1mf', grid=grid, space_order=space_order)
b1m2e = Function(name='b1m2e', grid=grid, space_order=space_order)
b1mfe2 = Function(name='b1mfe2', grid=grid, space_order=space_order)
b1mfpfe2 = Function(name='b1mfpfe2', grid=grid, space_order=space_order)
bfes1me2 = Function(name='bfes1me2', grid=grid, space_order=space_order)

# Equations for additional factorization
eq_b1mf = Eq(b1mf, b * (1 - f))
eq_b1m2e = Eq(b1m2e, b * (1 + 2 * eps0))
eq_b1mfe2 = Eq(b1mfe2, b * (1 - f * eta0**2))
eq_b1mfpfe2 = Eq(b1mfpfe2, b * (1 - f + f * eta0**2))
eq_bfes1me2 = Eq(bfes1me2, b * f * eta0 * sqrt(1 - eta0**2))

# Time update equation for quasi-P state variable p
update_p_nl = t.spacing**2 * vel0**2 / b * \
    (g1_tilde(b1m2e * g1(p0)) +
     g2_tilde(b1m2e * g2(p0)) +
     g3_tilde(b * (1 - f * eta0**2) * g3(p0) + bfes1me2 * g3(m0))) + \
    (2 - t.spacing * wOverQ) * p0 + (t.spacing * wOverQ - 1) * p0.backward

# Time update equation for quasi-S state variable m
update_m_nl = t.spacing**2 * vel0**2 / b * \
    (g1_tilde(b1mf * g1(m0)) +
     g2_tilde(b1mf * g2(m0)) +
     g3_tilde(b * (1 - f + f * eta0**2) * g3(m0) + bfes1me2 * g3(p0))) + \
    (2 - t.spacing * wOverQ) * m0 + (t.spacing * wOverQ - 1) * m0.backward

stencil_p_nl = Eq(p0.forward, update_p_nl)
stencil_m_nl = Eq(m0.forward, update_m_nl)

dt = time_axis.step
spacing_map = vel0.grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([eq_b1mf, eq_b1m2e, eq_bfes1me2,
               stencil_p_nl, stencil_m_nl, src_term],
              subs=spacing_map, name='OpExampleVti3')

f = open("operator.vti_fact3.c", "w")
print(op, file=f)
f.close()

bx = 8
by = 8
bx = 120
by = 24
print("\nCache block size (bx,by) = (%3d,%3d)" % (bx, by))
op.apply(x0_blk0_size=bx, y0_blk0_size=by)
