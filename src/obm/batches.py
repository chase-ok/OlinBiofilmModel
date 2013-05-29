
import utils
import specs
import models
import analysis as ana

DEFAULT_ANALYSES = [
    ana.compute_heights,
    ana.compute_height_dist,
    ana.compute_perimeter,
    ana.compute_coverages,
    ana.compute_overhangs,
    ana.compute_x_correlations,
    ana.compute_convex_hull_area
]

class Batch(object):

    def __init__(self, spec_uuids, 
                 num_models_per_spec=5, 
                 model_func=models.compute_probabilistic,
                 analyses=DEFAULT_ANALYSES,
                 show_progress=True):
        self.spec_uuids = spec_uuids
        self.num_models_per_spec = num_models_per_spec
        self.model_func = model_func
        self.analyses = analyses
        self.show_progress = show_progress

    def run(self):
        self._print("Starting...")
        for spec_num, uuid in enumerate(self.spec_uuids):
            self._print("Spec #{0}/{1} uuid={2} "\
                        .format(spec_num+1, len(self.spec_uuids), uuid))
            spec = specs.Spec.get(uuid)

            for i in range(self.num_models_per_spec):
                self._print("  Model #{0}".format(i+1))
                result = self.model_func(spec)
                result.save()

                for analysis in self.analyses:
                    self._print("    Analysis {0}".format(analysis.func_name))
                    analysis(result)
        self._print("Done!")

    def _print(self, message): 
        if self.show_progress: print message

def _bin(values, num_bins):
    bins = [[] for _ in range(num_bins)]
    for i, value in enumerate(values):
        bins[i%num_bins].append(value)
    return bins

def _default_file_names():
    i = 0
    while True:
        yield "{0}.specs".format(i)
        i += 1

def parcel_out_specs(num_files, file_name_gen=_default_file_names):
    all_uuids = [spec.uuid for spec in specs.Spec.all()]
    for bin, name in zip(_bin(all_uuids, num_files), file_name_gen()):
        with open(name, 'w') as f:
            for uuid in bin: f.write(uuid + "\n")

def run_batch_from_spec_file(path):
    uuids = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) != 0: uuids.append(line)
    batch = Batch(uuids)
    batch.run()

if __name__ == '__main__':
    import sys
    spec_file = sys.argv[1]
    h5_file = sys.argv[2]

    utils.DEFAULT_H5_FILE = h5_file
    run_batch_from_spec_file(spec_file)
