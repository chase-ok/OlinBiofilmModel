from celery import Celery, group
from obm import specs, models, utils
import numpy as np

class Config:
    BROKER_URL = 'amqp://guest@localhost//'
    CELERY_RESULT_BACKEND = "amqp"
    CELERY_TASK_SERIALIZER = 'pickle'
    CELERY_RESULT_SERIALIZER = 'pickle'

    CELERY_ROUTES = {'obm.compute.tasks.append_result': {'queue': 'hdf5'},
                     'obm.compute.tasks.create_specs': {'queue': 'hdf5'},
                     'obm.compute.tasks.map_models': {'queue': 'hdf5'},
                     'obm.compute.tasks.close_hdf5': {'queue': 'hdf5'},
                     'obm.compute.tasks.compute_model': {'queue': 'model'}}

celery = Celery('obm.compute.tasks')
celery.config_from_object(Config)

@celery.task
def create_specs():
    builder = specs.SpecBuilder()
    builder.add("boundary_layer", 10)
    builder.add("stop_on_no_growth", 500)
    builder.add("stop_on_time", 30000)
    builder.add("stop_on_mass", 2000)
    builder.add("light_penetration", 0, 8)
    builder.add("diffusion_constant", *np.linspace(0.01, 1.0, 40))
    builder.add("uptake_rate", *np.linspace(0.01, 1.0, 40))
    builder.add("height", 40)
    builder.add("initial_cell_spacing", 2)
    builder.build()

    return list(specs.Spec.all())

@celery.task
def map_models(specs, n):
    for spec in specs:
        for _ in range(n):
            (compute_model.s(spec) | append_result.s()).delay()

@celery.task(ignore_result=True)
def append_result(result):
    result.save()

@celery.task(ignore_result=True)
def close_hdf5(*_):
    utils.get_h5().hdf5.close()

@celery.task
def compute_model(spec):
    return models.compute_probabilistic(spec)

def run():
    workflow = create_specs.s() |\
               map_models.s(9)
    workflow.delay().get()
