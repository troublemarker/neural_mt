import time
import contextlib
from collections import OrderedDict


field_statistics_container_dict = OrderedDict()

@contextlib.contextmanager
def build_field_statistics_container(name, new_container_dict=False):

    log_dict = OrderedDict()
    if new_container_dict:
        backup_field_statistics_container = field_statistics_container_dict.copy()
        field_statistics_container_dict.clear()
    field_statistics_container_dict[name] = log_dict

    yield log_dict

    del field_statistics_container_dict[name]

    if new_container_dict:
        field_statistics_container_dict.clear()
        field_statistics_container_dict = backup_field_statistics_container.copy()
        del backup_field_statistics_container










class Measure(object):

    def __init__(self):
        self.value = 0

    def update_value(self, val):
        self.value += val

    def average(self):
        raise NotImplementedError


class TimeMeasure(Measure):

    def __init__(self):
        super(TimeMeasure, self).__init__()
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def average(self):
        return self.value / (self.end_time - self.start_time)


class CountMeasure(Measure):
    def __init__(self):
        super(CountMeasure, self).__init__()
        self.count = 0

    def update_value(self, val):
        self.count += 1
        self.value += val

    def average(self):
        return self.value / self.count