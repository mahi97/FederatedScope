import evaluate


_GLUE_METRIC_CACHE = {}


def get_glue_metric(task_name):
    if task_name not in _GLUE_METRIC_CACHE:
        _GLUE_METRIC_CACHE[task_name] = evaluate.load('glue',
                                                     task_name,
                                                     trust_remote_code=True)
    return _GLUE_METRIC_CACHE[task_name]
