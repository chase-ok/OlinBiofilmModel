import numpy as np
import cv2
import models
from scipy import optimize
from matplotlib import pyplot as plt
from itertools import product

# -----------------------------------------------------------------------------
# DIFFUSION
# -----------------------------------------------------------------------------

def justify_diffusion():
    boundary_layer = 5
    diffusion_const = 1.0
    cells = generate_static_biofilm()

    gaussian = lambda x: calc_diffusion_by_gaussian(cells, boundary_layer, x)
    herman = lambda x: calc_diffusion_by_hermanowicz(cells, boundary_layer, x)
    def iteration(x, dt=0.1, iters=5000):
        return calc_diffusion_by_iteration(cells, boundary_layer, x, diffusion_const, 
                                           dt=dt, iters=iters)

    depletion = 0.1
    exact = iteration(depletion, dt=0.1, iters=5000)
    ignore = make_boundary_layer(cells, boundary_layer) > 0
    penetration, _ = fit_to_values(gaussian, exact, ignore=ignore)
    herman_const, _ = fit_to_values(herman, exact, guess=0.01, ignore=ignore)
    make_diffusion_plot(exact, 'By Iteration', 'iteration')
    make_diffusion_plot(gaussian(penetration), 'By Gaussian', 'gaussian')
    make_diffusion_plot(herman(herman_const), 'By Hermanowicz', 'herman')

    num_iters = np.logspace(0, 3, 50).astype(int)
    def convergence(dt):
        return [np.sum((iteration(depletion, dt=dt, iters=i) - exact)**2)
                for i in num_iters]
    
    plt.clf(); plt.hold(True)
    for dt in [0.1, 0.2, 1.0, 2.0, 5.0]:
        plt.plot(num_iters, convergence(dt), lw=3, label='dt = %s'%dt)
    plt.semilogx()
    plt.legend()
    plt.xlabel('Num Iterations')
    plt.ylabel('Error')
    plt.savefig('fig/diffusion_convergence.png')

def test_diffusion(boundary_layer=5, diffusion_const=1.0, depletion=0.1):
    cells = generate_static_biofilm()

    gaussian = lambda x: calc_diffusion_by_gaussian(cells, boundary_layer, max(x, 0.5))
    #herman = lambda x: calc_diffusion_by_hermanowicz(cells, boundary_layer, x)
    def iteration(x, dt=0.1, iters=5000):
        return calc_diffusion_by_iteration(cells, boundary_layer, x, diffusion_const, 
                                           dt=dt, iters=iters)

    exact = iteration(depletion, dt=0.1, iters=5000)
    ignore = make_boundary_layer(cells, boundary_layer) > 0
    penetration, _ = fit_to_values(gaussian, exact, ignore=ignore)
    #herman_const, _ = fit_to_values(herman, exact, guess=0.01, ignore=ignore)
    make_diffusion_plot(exact, 'By Iteration', 'test-%s-%s-iteration' % (depletion, diffusion_const))
    make_diffusion_plot(gaussian(penetration), 'By Gaussian', 'test-%s-%s-gaussian' % (depletion, diffusion_const))
    #make_diffusion_plot(herman(herman_const), 'By Hermanowicz', 'test-%s-%s-herman' % (boundary_layer, diffusion_const))

def make_diffusion_plot(values, subtitle, file_name=None):
    plt.clf()
    plt.pcolor(values, cmap=plt.hot())
    plt.ylim([0, values.shape[0]-1])
    plt.xlim([0, values.shape[1]-1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Media Concentration: ' + subtitle)
    plt.colorbar()

    file_name = file_name or subtitle
    plt.savefig('fig/diffusion_%s.png' % file_name)

def calc_diffusion_by_gaussian(cells, boundary_layer, media_penetration):
    boundary = make_boundary_layer(cells, boundary_layer)
    media = cv2.GaussianBlur(boundary, (0, 0), media_penetration)
    media[boundary > 0] = boundary[boundary > 0]
    return media 

def calc_diffusion_by_hermanowicz(cells, boundary_layer, k):
    boundary = make_boundary_layer(cells, boundary_layer)

    r_max, c_max = cells.shape
    def distance_factor(r, c, dr, dc):
        increment = np.sqrt(dr**2 + dc**2)
        distance = 0.0
        while 0 <= r < r_max and 0 <= c < c_max:
            if boundary[r, c] > 0:
                return 1.0/(distance**2)
            distance += increment
            r += dr
            c += dc
        return 0.0

    def media_at(r, c):
        distances = [distance_factor(r, c, dr, dc)
                     for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                     if not (dr == 0 and dc == 0)]
        distances = [d for d in distances if d > 0.0]
        distance_sum = sum(distances)/float(len(distances))
        loss = (k/distance_sum)**0.5
        return 0.0 if loss > 1.0 else (1.0-loss)**2

    media = np.empty_like(cells, float)
    for r in range(media.shape[0]):
        for c in range(media.shape[1]):
            if boundary[r, c] > 0:
                media[r, c] = boundary[r, c]
            else:
                media[r, c] = media_at(r, c)
    return media

def calc_diffusion_by_iteration(cells, boundary_layer, k, d, dt=0.1, iters=5000):
    boundary = make_boundary_layer(cells, boundary_layer)
    in_cells = cells > 0
    in_boundary = boundary > 0

    media = boundary.copy()
    sigma = np.sqrt(2*d*dt)

    for _ in range(iters):
        cv2.GaussianBlur(media, (0, 0), sigma, dst=media)
        media[in_cells] -= k*dt*media[in_cells]
        media[in_boundary] = boundary[in_boundary]

    return media

def make_boundary_layer(cells, thickness, concentration=1.0):
    cells = cells.astype(np.uint8)
    kernel = models._make_circular_kernel(thickness)
    media = np.logical_not(cv2.filter2D(cells, -1, kernel)).astype(np.uint8)
    #remove any non-connected segments
    cv2.floodFill(media, None, (media.shape[0]-1, media.shape[1]/2), 2)
    return (media == 2).astype(float)*concentration

# -----------------------------------------------------------------------------
# SURFACE TENSION
# -----------------------------------------------------------------------------

def calc_surface_tension_by_convolution(cell_block, power):
    tension_kernel = np.array([[1, 2, 1],
                               [2, 0, 2],
                               [1, 2, 1]], float)
    tension_kernel /= tension_kernel.sum()
    tension_min = tension_kernel[0:1, 0].sum()

    probability = cv2.filter2D(cell_block, cv2.cv.CV_32F, tension_kernel, 
                               borderType=cv2.BORDER_CONSTANT)
    _, probability = cv2.threshold(probability, tension_min, 0, 
                                   cv2.THRESH_TOZERO)
    probability[cell_block > 0] = 0
    probability **= power
    return normalize_probability(probability)

def calc_surface_tension_by_energy(cell_block, tension):
    cell_block = cell_block.astype(np.uint8)

    def calc_perimeter():
        horizontal = np.sum(cell_block[:-1, :] != cell_block[1:, :])
        vertical = np.sum(cell_block[:, :-1] != cell_block[:, 1:])
        return horizontal + vertical
    
    base_perimeter = calc_perimeter()
    def calc_perimeter_change(index):
        cell_block[index] = 1
        change = calc_perimeter() - base_perimeter
        cell_block[index] = 0
        return change

    probability = np.empty_like(cell_block, float)
    for index in product(*map(range, cell_block.shape)):
        if cell_block[index] > 0:
            probability[index] = 0.0
        else:
            d_perimeter = calc_perimeter_change(index)
            probability[index] = np.exp(-tension*d_perimeter)

    return normalize_probability(probability)

def justify_surface_tension():
    block_size = 15

    cells = generate_static_biofilm()
    cell_block = np.empty((block_size, block_size), np.uint8)
    models._write_block(cell_block, cells, 23, 23, block_size)
    make_surface_tension_plot(cell_block, "Cells", 'cells')
    plt.clf(); plt.pcolor(cell_block, cmap=plt.hot()); plt.savefig("fig/surface_tension_cell_block.png")

    convolution = lambda x: calc_surface_tension_by_convolution(cell_block, x)
    energy = lambda x: calc_surface_tension_by_energy(cell_block, x)

    tension = 1.0
    power, _ = fit_to_values(convolution, energy(tension))
    make_surface_tension_plot(energy(tension), 'By Energy', 'energy')
    make_surface_tension_plot(convolution(power), 'By Convolution', 'convolution')

    tensions = np.linspace(0.1, 10.0)
    powers, residuals = fit_to_func(convolution, energy, tensions)
    plt.clf()
    plt.plot(tensions, powers, 'k-', lw=3)
    plt.xlabel('Tension')
    plt.ylabel('Power')
    plt.savefig('fig/surface_tension_tension_vs_power.png')

    plt.clf()
    plt.plot(tensions, residuals, 'r-', lw=3)
    plt.ylabel('Error')
    plt.xlabel('Tension')
    plt.savefig('fig/surface_tension_tension_vs_power_error.png')
    

def make_surface_tension_plot(values, subtitle, file_name=None):
    plt.clf()
    plt.pcolor(values, cmap=plt.hot())
    plt.ylim([0, values.shape[0]-1])
    plt.xlim([0, values.shape[1]-1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Surface Tension Probability: " + subtitle)
    plt.colorbar()

    file_name = file_name or subtitle
    plt.savefig('fig/surface_tension_%s.png' % file_name)

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def normalize_probability(probability):
    norm = probability.sum()
    if norm < 1e-10:
        return np.zeros_like(probability)
    else:
        return probability/norm

def fit_to_values(func, values, guess=1.0, ignore=None):
    def error(x):
        diffs = func(x) - values
        if ignore is not None:
            diffs[ignore] = 0
        return diffs
    residual = lambda x: np.sum(error(x)**2)

    optimal = optimize.fmin(residual, guess)[0]
    return optimal, np.sqrt(residual(optimal))/float(values.size)

def fit_to_func(approx_func, exact_func, func_values, ignore=None):
    return zip(*[fit_to_values(approx_func, exact_func(v), ignore=ignore) 
                 for v in func_values])

def generate_static_biofilm(image_file="biofilm.png"):
    return cv2.imread(image_file, 0)[:64, 64:128].astype(float)
