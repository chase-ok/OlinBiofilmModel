
CONFIG = {
    'h5_file':'data.h5',
    'fig_folder':'figs',
    'fig_prefix':'boundary10',
    'spec_query': None,
    'num_cells': 20,
    'cmap':'hot'
}

import utils
utils.set_h5(CONFIG['h5_file'])

import analysis as ana
import specs
import models
from matplotlib import pyplot as plt

def height_std(param1, param2):
    ana.height_std_field.phase_diagram_2d(param1, param2,
                                          **_phase_diagram_kwargs())
    _setup_figure(title='Std. Deviation of Height',
                  xlabel=_map_param_name(param1),
                  ylabel=_map_param_name(param2))
    name = 'height_std-{0}-vs-{1}'.format(param1, param2)
    _save_figure(name)

    max_std = ana.height_std_field.find_max_spec_id()
    spec = specs.get_spec(max_std)
    spec.dump(_make_path(name + '-max-spec.txt'))
    models.show_first_for_spec_id(max_std, show=False)
    _save_figure(name + "-max")

def main():
    height_std('diffusion_constant', 'uptake_rate')


_PARAM_NAMES = {
    'diffusion_constant':'Diffusion Constant',
    'uptake_rate':'Uptake Rate'
}
def _map_param_name(param):
    return _PARAM_NAMES.get(param, param)

def _make_path(ending):
    return "{0}/{1}-{2}"\
           .format(CONFIG['fig_folder'], CONFIG['fig_prefix'], ending)

def _save_figure(name):
    plt.savefig(_make_path(name + ".png"))
    plt.clf()

def _setup_figure(title=None, xlabel=None, ylabel=None, cmap=None):
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.set_cmap(cmap if cmap else CONFIG['cmap'])

def _phase_diagram_kwargs():
    return dict(show=False, 
                num_cells=CONFIG['num_cells'], 
                spec_query=CONFIG['spec_query'])

if __name__ == '__main__':
    main()