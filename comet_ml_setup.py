from comet_ml import Experiment, ExistingExperiment

api_key = "foo"
experiment = None

def init():
    global experiment
    if experiment is None:
        experiment = Experiment(api_key)
    else:
        experiment = ExistingExperiment(api_key, experiment.id)
