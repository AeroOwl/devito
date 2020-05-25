import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator)
from examples.seismic import RickerSource, TimeAxis

space_order = 8
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 250.0
tmax = 3.0
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
vel0 = Function(name='vel0', grid=grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=vel0.grid, space_order=space_order)

b.data[:] = 1.0
vel0.data[:] = 1.5
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


# works for smaller sizes, seg faults at (1001,1001,501)
# if you comment out the Y derivatives, works at (1001,1001,501)
# Time update equation for quasi-P state variable p
update_p_nl = t.spacing**2 * vel0**2 / b * \
    (g1_tilde(b * g1(p0)) +
     g2_tilde(b * g2(p0)) +
     g3_tilde(b * g3(p0))) + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

stencil_p_nl = Eq(p0.forward, update_p_nl)

dt = time_axis.step
spacing_map = vel0.grid.spacing_map
spacing_map.update({t.spacing: dt})

# 2020.05.20
def callback(n):
    if n == 1:
        return 50
    elif n == 0:
        return 20
    assert False
    
op = Operator([stencil_p_nl, src_term],
              subs=spacing_map, name='OpExampleIso',
              opt=('advanced', {'cire-repeats-inv': 2, 'cire-mincost-inv': callback}))

f = open("operator.iso.c", "w")
print(op, file=f)
f.close()

bx = 92
by = 40
bx = 12
by = 4
op.apply(x0_blk0_size=bx, y0_blk0_size=by)
