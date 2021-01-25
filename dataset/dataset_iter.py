import itertools
import math
import operator
import queue
from threading import Thread

import numpy as np
import torch
from torch.utils.data import DataLoader



def data_generater(iterable, chunk_size):
    chunk = []
    for sample in iterable:
        if len(chunk) < chunk_size:
            chunk.append(sample)
        yield chunk
        chunk = []
    if len(chunk) > 0:
        yield chunk


class EpochIterator(object):

    epoch = 0

    def __init__(self, dataset, batches, num_shards, shard_id, fill_value=None):
        EpochIterator.epoch += 1

        self.shard_len = math.ceil(len(batches) / num_shards)
        self.dataset = dataset
        self.batches = batches
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.fill_value = None
        self.itr = self.get_iterator()

    def get_iterator(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self.batches)

        batches = list(itertools.islice(self.batches, start=self.shard_id, step=self.num_shards, stop=len(self.batches)))
        itr = DataLoader(self.dataset, collate_fn=self.dataset.collater, batch_sampler=batches)

        return itr

    def __len__(self):
        return self.shard_len


class BackgroundConsumer(Thread):

    def __init__(self, queue, source):
        super(BackgroundConsumer, self).__init__()
        self.queue = queue
        self.source = source

    def run(self) -> None:
        for item in self.source:
            self.queue.put(item)


class BaseIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.iterable)


class BufferedIterator(BaseIterator):
    def __init__(self, size, iterable):
        super(BufferedIterator, self).__init__(iterable)
        self._queue = queue.Queue(size)

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(self._queue, self.iterable)
        self._consumer.start()

    def __next__(self):
        if self._consumer is None:
            self._create_consumer()

        item = self._queue.get()
        return item


class ChunkIterator(BaseIterator):
    def __init__(self, iterable, chunk_size):
        super(ChunkIterator, self).__init__(iterable)
        self.chunk_size = chunk_size
        self.data_gen = data_generater(iterable, chunk_size)

    def __next__(self):
        return next(self.data_gen)


class CountIterator(BaseIterator):
    def __init__(self, iterable):
        super(CountIterator, self).__init__(iterable)
        self.count = 1

    def __next__(self):
        self.count += 1
        return next(self.iterable)

    def has_next(self):
        return self.count < len(self)







