import numpy as np
import cv2
import models
from scipy.linalg import solve
from scipy.ndimage.filters import laplace
from scipy import optimize
from collections import deque
from matplotlib import pyplot as plt
from itertools import product

def justify_gaussian():
    cells = generate_static_biofilm()

    #k = 0.01
    #penetration, _ = calculate_penetration_for_k(cells, 5, k)
    #print penetration

    penetration, _ = calculate_penetration_for_k_d(cells, 5, 0.1, 1.0, plot_error=True)
    print penetration

    #gaussian = calculate_diffusion_by_gaussian(cells, 5, penetration)
    #herman = calculate_diffusion_by_hermanowicz(cells, 5, k)
    #iteration = calculate_diffusion_by_iteration(cells, 5, 0.05, 1.0)
    #plt.figure(); plt.pcolor(gaussian, cmap=plt.hot()); plt.colorbar(); plt.title('Gaussian')
    #plt.figure(); plt.pcolor(herman, cmap=plt.hot()); plt.colorbar(); plt.title('Hermanowicz')
    #plt.figure(); plt.pcolor(iteration, cmap=plt.hot()); plt.colorbar(); plt.title('Iteration')

    #plot_penentration_vs_k(cells, [3, 5, 7, 10])

def plot_penentration_vs_k(cells, boundary_layers):
    results = []
    kRecips = np.linspace(1, 300, 20)
    for boundary in boundary_layers:
        results.append(zip(*[calculate_penetration_for_k(cells, boundary, 1.0/kRecip)
                             for kRecip in kRecips]))

    plt.figure(); plt.hold(True)

    for result, boundary in zip(results, boundary_layers):
        plt.plot(kRecips, result[0], lw=3, label=('Boundary = %d' % boundary))
    plt.ylabel('Penetration Depth')
    plt.legend(loc='lower right')

    plt.twinx()
    for result, boundary in zip(results, boundary_layers):
        plt.plot(kRecips, np.sqrt(result[1])/cells.size, lw=1, label='__none__')
    plt.ylabel('Average Error')
    plt.xlabel('2D/k')

    plt.title('Gaussian Diffusion')

def calculate_penetration_for_k(cells, boundary_layer, k, plot_error=False):
    herman = calculate_diffusion_by_hermanowicz(cells, boundary_layer, k)

    no_cells = np.logical_not(cells)
    def error(penetration):
        penetration = max(0.0001, penetration)
        diff = calculate_diffusion_by_gaussian(cells, boundary_layer, penetration) - herman
        diff[no_cells] = 0.0
        return diff
    residual = lambda x: np.sum(error(x)**2)

    penetration = optimize.fmin(residual, 1.0)[0]
    if plot_error:
        plt.figure(); plt.pcolor(abs(error(penetration))); plt.colorbar(); plt.title('Error')
    return penetration, residual(penetration)

def calculate_penetration_for_k_d(cells, boundary_layer, k, d, plot_error=False):
    iteration = calculate_diffusion_by_iteration(cells, boundary_layer, k, d)
    boundary = make_boundary_layer(cells, boundary_layer)

    no_cells = np.logical_not(cells)
    def error(penetration):
        penetration = max(0.0001, penetration)
        diff = calculate_diffusion_by_gaussian(cells, boundary_layer, penetration) - iteration
        #diff = calculate_diffusion_by_hermanowicz(cells, boundary_layer, penetration) - iteration
        diff[boundary > 0] = 0.0
        return diff
    residual = lambda x: np.sum(error(x)**2)

    penetration = optimize.fmin(residual, 1.0)[0]
    if plot_error:
        plt.figure(); plt.pcolor(error(penetration)); plt.colorbar(); plt.title('Error')
    return penetration, residual(penetration)

def generate_static_biofilm(image_file="biofilm.png"):
    return cv2.imread(image_file, 0)[:64, 64:128].astype(float)

def generate_random_biofilm(n, p, max_density):
    grid = np.zeros((n, n), float)

    daughters = deque([(0, n/2)])
    num_available = int(n*n*max_density)

    while daughters and num_available > 0:
        row, col = daughters.pop()

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == dc: 
                    continue

                new_row, new_col = row+dr, col+dc
                if not (0 <= new_row < n) or not (0 <= new_col < n):
                    continue
                if grid[new_row, new_col] > 0:
                    continue

                if np.random.random() < p:
                    grid[new_row, new_col] = 1.0
                    daughters.appendleft((new_row, new_col))
                    num_available -= 1

    return grid

def calculate_diffusion_by_gaussian(cells, boundary_layer, media_penetration):
    media = make_boundary_layer(cells, boundary_layer)
    return cv2.GaussianBlur(media, (0, 0), media_penetration)

def calculate_diffusion_by_hermanowicz(cells, boundary_layer, k):
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

def calculate_diffusion_by_iteration(cells, boundary_layer, k, d, dt=0.1):
    boundary = make_boundary_layer(cells, boundary_layer)
    in_cells = cells > 0
    in_boundary = boundary > 0

    media = boundary.copy()
    sigma = np.sqrt(2*d*dt)
    def step():
        cv2.GaussianBlur(media, (0, 0), sigma, dst=media)
        media[in_cells] -= k*dt*media[in_cells]
        media[in_boundary] = boundary[in_boundary]

    for i in range(5000): step()
    return media

def make_boundary_layer(cells, thickness, concentration=1.0):
    cells = cells.astype(np.uint8)
    kernel = models._make_circular_kernel(thickness)
    media = np.logical_not(cv2.filter2D(cells, -1, kernel)).astype(np.uint8)
    #remove any non-connected segments
    cv2.floodFill(media, None, (media.shape[0]-1, media.shape[1]/2), 2)
    return (media == 2).astype(float)*concentration

def compute_diffusion_error(cells, u, p_range):
    """
    u = concentration
    ds = possible diffusion constants
    rs = possible poisson terms for live cells
    """
    laplacian = laplace(u, mode='reflect')
    poisson = (cells > 0)*1.0
    error = lambda x: laplacian - x[0]*poisson
    residual = lambda x: np.sum(error(x)**2)

    x, min_resid, info = optimize.fmin_l_bfgs_b(residual, 
                                                [p_range[1]], 
                                                bounds=[p_range],
                                                approx_grad=True)

    plt.pcolor(error(x))
    plt.colorbar()
    plt.show()
    return min_resid, error(x), x[0]

def justify_surface_tension(block_size=7):
    cells = generate_static_biofilm()
    cell_block = np.empty((block_size, block_size), np.uint8)
    models._write_block(cell_block, cells, 23, 23, block_size)

    tension = 1.0
    power = calculate_power_for_tension(cell_block, tension)
    print power
    approx = calculate_probabilities_by_convolution(cell_block.copy(), power)
    exact = calculate_probabilities_by_energy(cell_block.copy(), tension)
    plt.figure(); plt.pcolor(cell_block)
    plt.figure(); plt.pcolor(cells)
    plt.figure(); plt.pcolor(approx, cmap=plt.hot()); plt.colorbar(); plt.title('Approx')
    plt.figure(); plt.pcolor(exact, cmap=plt.hot()); plt.colorbar(); plt.title('Exact')
    plt.show()

def calculate_probabilities_by_convolution(cell_block, power):
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
    

def calculate_probabilities_by_energy(cell_block, tension):
    cell_block = cell_block.astype(np.uint8)

    def calculate_perimeter():
        #contours, _ = cv2.findContours(cell_block.copy(), 
        #                               cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #return sum(cv2.arcLength(contour, True) for contour in contours)
        horizontal = np.sum(cell_block[:-1, :] != cell_block[1:, :])
        vertical = np.sum(cell_block[:, :-1] != cell_block[:, 1:])
        return horizontal + vertical
    
    base_perimeter = calculate_perimeter()
    def calculate_perimeter_change(index):
        cell_block[index] = 1
        change = calculate_perimeter() - base_perimeter
        cell_block[index] = 0
        return change

    probability = np.empty_like(cell_block, float)
    for index in product(*map(range, cell_block.shape)):
        if cell_block[index] > 0:
            probability[index] = 0.0
        else:
            d_perimeter = calculate_perimeter_change(index)
            probability[index] = np.exp(-tension*d_perimeter)

    return normalize_probability(probability)

def normalize_probability(probability):
    norm = probability.sum()
    if norm < 1e-10:
        return np.zeros_like(probability)
    else:
        return probability/norm

def calculate_power_for_tension(cell_block, tension):
    exact = calculate_probabilities_by_energy(cell_block.copy(), tension)

    error = lambda x: calculate_probabilities_by_convolution(cell_block.copy(), x) - exact
    residual = lambda x: np.sum(error(x)**2)

    power = optimize.fmin(residual, 1.0)[0]
    plt.figure(); plt.pcolor(abs(error(power))); plt.colorbar(); plt.title('Error')
    return power

if __name__ == '__main__':
    justify_gaussian()
