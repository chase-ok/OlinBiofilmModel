
import numpy as np
import cv2
from scipy.ndimage.filters import laplace
from matplotlib.pyplot import *

def diffuse(cells=None, boundary=None, d=1.0, uptake=0.1, num_steps=1000):
    dt = 1.0/(4*d)
    u = 1.0*boundary

    # dx = dy = 1
    u_next = np.empty_like(u)
    storage = np.empty_like(u)
    for _ in range(num_steps):
        laplace(u, output=storage)
        u_next = u + d*dt*storage
        u_next[cells] *= 1 - uptake*dt
        u_next[boundary] = 1.0
        u, u_next = u_next, u

    return u

def diffuse_check(cells=None, boundary=None, d=1.0, uptake=0.1, tol=1e-4, max_steps=5000, 
                  check_start=40):
    check_interval = 10
    dt = 1.0/(4*d)
    u = 1.0*boundary

    # dx = dy = 1
    u_next = np.empty_like(u)
    storage = np.empty_like(u)
    for step in range(max_steps):
        laplace(u, output=storage)
        u_next = u + d*dt*storage
        u_next[cells] *= 1 - uptake*dt
        u_next[boundary] = 1.0
        u, u_next = u_next, u

        if step >= check_start and step%check_interval == 0:
            error = np.abs(u_next-u).sum()/(dt*u.size)
            if error <= tol: break
            check_interval += 5

    return u, step

def diffuse_old(cells=None, boundary=None, d=1.0, uptake=0.1, num_steps=1000):
    dt = 1.0/(4*d)
    u = 1.0*boundary
    sigma = np.sqrt(2*d*dt)

    for _ in range(num_steps):
        cv2.GaussianBlur(u, (0, 0), sigma, dst=u)
        u[boundary] = 1.0
        u[cells] *= 1 - uptake*dt

    return u

def narrow_focus(cells, boundary_thickness):
    max_height = 0
    while max_height < cells.shape[0] and cells[max_height, :].any():
        max_height += 1
    narrow_cells = cells[0:max_height+boundary_thickness+1, :]

    return narrow_cells, make_boundary(narrow_cells, boundary_thickness)

def make_boundary(cells, thickness):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    boundary_layer = cv2.filter2D(cells.astype(np.uint8), -1, kernel)
    np.logical_not(boundary_layer, out=boundary_layer)

    #remove any non-connected segments
    fill_value = 2
    fill_source = boundary_layer.shape[0]-1, 0 # (x, y) not (r, c)
    cv2.floodFill(boundary_layer, None, fill_source, fill_value)

    return boundary_layer != fill_value

def timeit_setup(method):
    cells = imread('biofilm.png') > 0
    narrow, boundary = narrow_focus(cells, 10)
    if method == 'new':
        return lambda n: diffuse(cells=narrow, boundary=boundary, num_steps=n)
    elif method == 'old':
        return lambda n: diffuse_old(cells=narrow, boundary=boundary, num_steps=n)

def calculate_convergence():
    cells = imread('biofilm.png') > 0
    narrow, boundary = narrow_focus(cells, 10)

    ds, uptakes = np.meshgrid(np.linspace(0.1, 2.0, 10), np.linspace(0.01, 0.5, 10))
    iters = np.empty_like(ds, dtype=int)
    for i in range(ds.shape[0]):
        print i
        for j in range(ds.shape[1]):
            iters[i, j] = diffuse_check(cells=narrow, boundary=boundary, d=ds[i,j], uptake=uptakes[i,j])[1]
    pcolor(ds, uptakes, iters)
    show()
    return iters
    #equilib = diffuse(cells=narrow, boundary=boundary, num_steps=5000)

    #iters = range(1, 500, 5)
    #diffs = [np.abs(diffuse(cells=narrow, boundary=boundary, num_steps=n)-equilib).sum()/narrow.size
    #         for n in iters]
    #plot(iters, diffs)
    #show()

def test(num_steps=1000, num_steps_old=1000):
    cells = imread('biofilm.png') > 0
    narrow, boundary = narrow_focus(cells, 10)

    u = diffuse(cells=narrow, boundary=boundary, num_steps=num_steps)
    u_old = diffuse_old(cells=narrow, boundary=boundary, num_steps=num_steps_old)

    figure()
    subplot(311); imshow(u)
    subplot(312); imshow(u_old)
    subplot(313); imshow(np.abs(u-u_old)); print np.abs(u-u_old).sum()
    show()

def clear_up_results():
    import specs, models
    to_delete = []
    for row in models.Result.table.iter_rows(True):
        try:
            specs.Spec.get(row['spec_uuid'])
        except KeyError:
            to_delete.append(row['uuid'])
    for i, uuid in enumerate(to_delete):
        print i
        models.Result(uuid=uuid).delete()

def do_analysis():
    import specs, analysis

    for i, spec in enumerate(specs.Spec.all()):
        if i < 3500: continue
        print i
        analysis.heights.get_by_spec(spec)
        analysis.perimeter.get_by_spec(spec)
        analysis.convex_hull_area.get_by_spec(spec)
        analysis.mass.get_by_spec(spec)
        analysis.light_exposure.get_by_spec(spec)
        del spec