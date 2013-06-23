from obm import specs
import numpy as np

builder = specs.SpecBuilder()
builder.add("boundary_layer", 10)
builder.add("stop_on_no_growth", 300)
builder.add("stop_on_time", 20000)
builder.add("stop_on_mass", 2000)
builder.add("light_penetration", 0, 8, 16)
builder.add("diffusion_constant", *np.linspace(0.01, 1.0, 20))
builder.add("uptake_rate", *np.linspace(0.01, 1.0, 20))
builder.add("height", 40)
builder.add("initial_cell_spacing", 2, 8, 16, 64)
builder.build()