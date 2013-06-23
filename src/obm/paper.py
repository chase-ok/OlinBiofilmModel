

import matplotlib
from matplotlib import pyplot as plt
import cv2
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'

def make_diffusion_data():
    import utils; utils.DEFAULT_H5_FILE = "new_diffusion.h5"
    import specs, models, analysis

    builder = specs.SpecBuilder()
    builder.add("boundary_layer", 10)
    builder.add("stop_on_no_growth", 300)
    builder.add("stop_on_time", 20000)
    builder.add("stop_on_mass", 2000)
    builder.add("light_penetration", 0, 8, 16)
    builder.add("diffusion_constant", *np.linspace(0.01, 1.0, 20))
    builder.add("uptake_rate", *np.linspace(0.01, 1.0, 20))
    builder.add("height", 40)
    builder.add("initial_cell_spacing", 2, 16)
    print builder.num_specs
    builder.build()

    for i, spec in enumerate(specs.Spec.all()):
        print i
        for j in range(1):
            print j,
            result = models.compute_probabilistic(spec)
            result.save()
        analysis.coverages.compute_by_spec(spec)

    print "Done!"

def do_diffusion_plots():
    import utils; utils.DEFAULT_H5_FILE = "data.h5"
    import specs, models, analysis

    for boundary_layer in [3,6,9,12,15]:
        for light in [1,2,4,6,10,16,24,32]:
            query = specs.make_query(boundary_layer='==%s'%boundary_layer,
                                     light_penetration='==%s'%light)

            plt.clf()
            analysis.heights.phase_diagram_2d(
                'diffusion_constant', 'uptake_rate',
                spec_query=query, num_cells=15, statistic='max',
                cmap=plt.get_cmap('hot'), vmin=0, vmax=64)
            save_plot(path='diffusion/heights/max/boundary{0}_light{1}'.format(boundary_layer, light),
                      xlabel='Diffusion Constant',
                      ylabel='Uptake Rate')

            plt.clf()
            analysis.convex_density.phase_diagram_2d(
                'diffusion_constant', 'uptake_rate',
                spec_query=query, num_cells=15, statistic='mean',
                cmap=plt.get_cmap('hot'), vmin=0, vmax=1)
            save_plot(path='diffusion/convex_density/boundary{0}_light{1}'.format(boundary_layer, light),
                      xlabel='Diffusion Constant',
                      ylabel='Uptake Rate')

def make_coverage_data():
    import utils; utils.DEFAULT_H5_FILE = "new_coverages.h5"
    import specs, models, analysis

    builder = specs.SpecBuilder()
    builder.add("boundary_layer", 10)
    builder.add("stop_on_no_growth", 300)
    builder.add("stop_on_time", 20000)
    builder.add("stop_on_mass", 1000, 1500, 2000, 2500)
    builder.add("light_penetration", 0, 4, 8, 12, 16)
    builder.add("diffusion_constant", 0.4)
    builder.add("uptake_rate", 0.7)
    builder.add("height", 40)
    builder.add("initial_cell_spacing", 16)
    print builder.num_specs
    builder.build()

    for i, spec in enumerate(specs.Spec.all()):
        print i
        for j in range(15):
            print j,
            result = models.compute_probabilistic(spec)
            result.save()
        analysis.coverages.compute_by_spec(spec)

    print "Done!"

def make_coverage_data2():
    import utils; utils.DEFAULT_H5_FILE = "new2_coverages.h5"
    import specs, models, analysis

    builder = specs.SpecBuilder()
    builder.add("boundary_layer", 10)
    builder.add("stop_on_no_growth", 300)
    builder.add("stop_on_time", 20000)
    builder.add("stop_on_mass", 1000, 2000)
    builder.add("light_penetration", 0, 8, 16)
    builder.add("diffusion_constant", 0.4)
    builder.add("uptake_rate", 0.7)
    builder.add("height", 40)
    builder.add("initial_cell_spacing", 2, 4, 6, 8, 10, 16, 24, 32, 64)
    print builder.num_specs
    builder.build()

    for i, spec in enumerate(specs.Spec.all()):
        print i
        for j in range(10):
            print j,
            result = models.compute_probabilistic(spec)
            result.save()
        analysis.coverages.compute_by_spec(spec)

    print "Done!"

def do_coverage_plots():
    import utils; utils.DEFAULT_H5_FILE = "new_coverages.h5"
    import specs, models, analysis

    plt.clf()
    for light in [0, 4, 8, 16]:
        query = specs.make_query(stop_on_mass='==2500', light_penetration='==%s'%light)
        analysis.coverages.average_curve_plot(spec_query=query, lw=3, 
                label='Light Penetration Depth = %s'%light)
    plt.legend()
    save_plot(path='coverages/average_test', xlabel='Depth', ylabel='Coverage')


def make_biomass_vs_light_data():
    import utils; utils.DEFAULT_H5_FILE = "new_growth_data.h5"
    import specs, models, analysis

    builder = specs.SpecBuilder()
    builder.add("boundary_layer", 10)
    builder.add("stop_on_no_growth", 1000)
    builder.add("stop_on_time", 300000)
    builder.add("light_penetration", 0, 8)
    builder.add("diffusion_constant", 0.4)
    builder.add("uptake_rate", 0.7)
    builder.add("stop_on_mass", *np.linspace(500, 3500, 200))
    builder.build()

    for i, spec in enumerate(specs.Spec.all()):
        print i
        result = models.compute_probabilistic(spec)
        result.save()
        analysis.light_exposure.compute_by_spec(spec)
        analysis.mass.compute_by_spec(spec)

    print "Done!"

def make_biomass_vs_light_data_spaced():
    import utils; utils.DEFAULT_H5_FILE = "new_growth_data_spaced.h5"
    import specs, models, analysis

    builder = specs.SpecBuilder()
    builder.add("boundary_layer", 10)
    builder.add("stop_on_no_growth", 1000)
    builder.add("stop_on_time", 200000)
    builder.add("light_penetration", 0, 8)
    builder.add("diffusion_constant", 0.4)
    builder.add("uptake_rate", 0.7)
    builder.add("stop_on_mass", *np.linspace(400, 2400, 200))
    builder.add("initial_cell_spacing", 16)
    builder.add("height", 40)
    builder.build()

    for i, spec in enumerate(specs.Spec.all()):
        print i
        result = models.compute_probabilistic(spec)
        result.save()
        analysis.light_exposure.compute_by_spec(spec)
        analysis.mass.compute_by_spec(spec)

    print "Done!"

def do_biomass_vs_light_plots():
    import utils; utils.DEFAULT_H5_FILE = "new_growth_data.h5"
    import specs, models, analysis
    from numpy.random import randint

    plt.clf()

    query = specs.make_query(light_penetration='==8', stop_on_mass='<4000')
    x8, y8 = analysis.mass.scatter_plot(analysis.light_exposure, spec_query=query, 
                                        marker='o', mfc='none', ms=8,
                                        label='Light Dependent (Penetration Depth = 8)')

    query = specs.make_query(light_penetration='==0', stop_on_mass='<4000')
    x0, y0 = analysis.mass.scatter_plot(analysis.light_exposure, spec_query=query, 
                                        mec='r', mfc='r', ms=4,
                                        label='Light Independent', marker='.')

    plt.semilogx()
    plt.xlim([500, 3500])
    powers = [2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
    plt.xticks([10**x for x in powers], ['$10^{%s}$'%x for x in powers])
    plt.legend(loc='lower right')
    save_plot(path='new_biomass_vs_light_exposure_all', 
              xlabel='Biomass', ylabel='Light Exposure')

    def dump_to_file(path, xs, ys):
        with open(path, "w") as f:
            f.write("biomass,light_exposure\n")
            for x, y in zip(xs, ys):
                f.write("%s,%s\n"%(x, y))
    dump_to_file("fig/new_light_indep_data.txt", x0, y0)
    dump_to_file("fig/new_light_dep_dep8_data.txt", x8, y8)

    x_range = [1500, 3500]
    points = np.linspace(*x_range)
    log_points = np.log10(points)
    plt.xlim(x_range)
    plt.ylim([580, 700])

    def do_fit(x, y, label, **plot_args):
        in_range = (x >= x_range[0])*(x <= x_range[1])
        log_x = np.log10(x[in_range])
        y = y[in_range]

        fit = np.polyfit(log_x, y, 2, full=True)
        p = np.poly1d(fit[0])
        r2 = 1 - fit[1]/(y.size*y.var())
        print label, p, r2
        plt.plot(points, p(log_points), label=label, **plot_args)
        return p

    p0 = do_fit(x0, y0, 'Light Independent', c='r', lw=3)
    p8 = do_fit(x8, y8, 'Light Dependent (Penetration Depth = 8)', ls='--', lw=3, c='k')
    
    save_plot(path='new_biomass_vs_light_exposure_upper', 
              xlabel='Biomass', ylabel='Light Exposure')

    def get_percent_diff(p1, p2):
        return np.abs((p1(log_points)-p2(log_points))/p1(log_points)).mean()
    real_diff = get_percent_diff(p0, p8)
    print "Real diff", real_diff

    pool = np.array(zip(x0, y0) + zip(x8, y8))
    def sample_percent_diff():
        def sample(size): return pool[randint(pool.shape[0], size=size), :]
        subpool0, subpool8 = sample(x0.size), sample(x8.size)
        def fit(subpool): return np.poly1d(np.polyfit(np.log10(subpool[:, 0]), subpool[:, 1], 2))
        p0, p8 = fit(subpool0), fit(subpool8)
        return get_percent_diff(p0, p8)

    #percent_diffs = [sample_percent_diff() for _ in range(100000)]
    #print "p-value", sum(1 for d in percent_diffs if d > real_diff)/float(len(percent_diffs))

def do_biomass_vs_light_plots_spaced():
    import utils; utils.DEFAULT_H5_FILE = "new_growth_data_spaced.h5"
    import specs, models, analysis
    from numpy.random import randint

    plt.clf()

    query = specs.make_query(light_penetration='==8', stop_on_mass='<4000')
    x8, y8 = analysis.mass.scatter_plot(analysis.light_exposure, spec_query=query, 
                                        marker='o', mfc='none', ms=8,
                                        label='Light Dependent (Penetration Depth = 8)')

    query = specs.make_query(light_penetration='==0', stop_on_mass='<4000')
    x0, y0 = analysis.mass.scatter_plot(analysis.light_exposure, spec_query=query, 
                                        mec='r', mfc='r', ms=4,
                                        label='Light Independent', marker='.')

    plt.semilogx()
    plt.xlim([400, 2400])
    powers = [2.6, 2.8, 3.0, 3.2, 3.4]
    plt.xticks([10**x for x in powers], ['$10^{%s}$'%x for x in powers])
    plt.legend(loc='lower right')
    save_plot(path='new_spaced_biomass_vs_light_exposure_all', 
              xlabel='Biomass', ylabel='Light Exposure')

    def dump_to_file(path, xs, ys):
        with open(path, "w") as f:
            f.write("biomass,light_exposure\n")
            for x, y in zip(xs, ys):
                f.write("%s,%s\n"%(x, y))
    dump_to_file("fig/new_spaced_light_indep_data.txt", x0, y0)
    dump_to_file("fig/new_spaced_light_dep_dep8_data.txt", x8, y8)

    x_range = [400, 2400]
    points = np.linspace(*x_range)
    log_points = np.log10(points)
    plt.xlim(x_range)
    plt.ylim([220, 670])

    def do_fit(x, y, label, **plot_args):
        in_range = (x >= x_range[0])*(x <= x_range[1])
        log_x = np.log10(x[in_range])
        y = y[in_range]

        fit = np.polyfit(log_x, y, 1, full=True)
        p = np.poly1d(fit[0])
        r2 = 1 - fit[1]/(y.size*y.var())
        print label, p, r2
        plt.plot(points, p(log_points), label=label, **plot_args)
        return p

    p0 = do_fit(x0, y0, 'Light Independent', c='r', lw=3)
    p8 = do_fit(x8, y8, 'Light Dependent (Penetration Depth = 8)', ls='--', lw=3, c='k')
    
    save_plot(path='new_spaced_biomass_vs_light_exposure_upper', 
              xlabel='Biomass', ylabel='Light Exposure')

    def get_percent_diff(p1, p2):
        return np.abs((p1(log_points)-p2(log_points))/p1(log_points)).mean()
    real_diff = get_percent_diff(p0, p8)
    print "Real diff", real_diff

    pool = np.array(zip(x0, y0) + zip(x8, y8))
    def sample_percent_diff():
        def sample(size): return pool[randint(pool.shape[0], size=size), :]
        subpool0, subpool8 = sample(x0.size), sample(x8.size)
        def fit(subpool): return np.poly1d(np.polyfit(np.log10(subpool[:, 0]), subpool[:, 1], 1))
        p0, p8 = fit(subpool0), fit(subpool8)
        return get_percent_diff(p0, p8)

    #percent_diffs = [sample_percent_diff() for _ in range(100000)]
    #print "p-value", sum(1 for d in percent_diffs if d > real_diff)/float(len(percent_diffs))

def save_plot(xlabel=None, ylabel=None, path=''):
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.title('')
    plt.savefig('fig/{0}.png'.format(path), dpi=600, 
                bbox_inches='tight', pad_inches=0.1)

def dump_pictures():
    import models, analysis
    for result in models.Result.all():
        #if analysis.mass.get_by_result(result) < 3800: continue
        cv2.imwrite("fig/images/" + result.uuid + ".png", result.int_image*255)
    print "done!"
