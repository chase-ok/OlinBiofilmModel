from obm import specs
import numpy as np

b = specs.SpecBuilder()
b.add('boundary_layer', *range(3, 17, 2))
b.add('light_penetration', *range(0, 32, 2))
b.add('light_penetration', *range(32, 65, 6))
b.add('diffusion_constant', *np.linspace(0.1, 2.0, 15))
b.add('uptake_rate', *np.linspace(0.01, 1.0, 15))
b.add('max_diffusion_iterations', 2500)

b.build()