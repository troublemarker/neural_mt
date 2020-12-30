import os

import torch
import numpy as np

from torch.utils.data import Dataset

from dataset_utils import get_offset_file, safe_read, get_index_file

nptype_code = {np.int32: 1}


class SingleDataset(Dataset):

    def __init__(self, dataset_dir):
        self.sizes, self.pointers = self._load_index(get_offset_file(dataset_dir))
        self.data = self._load_data(get_index_file(dataset_dir))

    def _load_index(self, index_path):
        with open(index_path, 'rb') as fr:
            buffer = fr.read()
            size = np.frombuffer(buffer, dtype=np.int32)
            pointers = np.zeros(len(size), dtype=np.int32)
            for i in range(1, len(size)):
                pointers[i] = size[i-1] * np.int32().itemsize
        return size, pointers

    def _load_data(self, data_path):
        mmap = np.memmap(data_path, mode='r', dtype=np.int32)
        return mmap

    def __getitem__(self, index):
        data = np.frombuffer(self.data, dtype=np.int32, offset=self.pointers[index], count=self.sizes[index])
        return torch.from_numpy(data)

    def batch_by_size(self, max_token):
        sorted_idx = np.argsort(self.sizes)
        batches, batch, token_num = [], [], 0
        for idx in sorted_idx:
            token_num += self.sizes[idx]
            if token_num > max_token:
                batches.append(batch)
                batch, token_num = [], 0
            else:
                batch.append(idx)

        return batches


class PairDataset(Dataset):
    def __init__(self, src_dataset, trg_dataset):
        self.src_dataset = src_dataset
        self.trg_dataset = trg_dataset

    def __getitem__(self, index):
        return self.src_dataset[index], self.trg_dataset[index]

    def batch_by_size(self, max_token):

        sorted_idx = self.ordered_indices()
        batches, batch, token_num = [], [], 0
        for idx in sorted_idx:
            token_num += max(self.src_dataset.sizes[idx], self.trg_dataset.sizes[idx])
            if token_num > max_token:
                batches.append(batch)
                batch, token_num = [], 0
            else:
                batch.append(idx)

        return batches

    def ordered_indices(self):
        indices = np.argsort(self.trg_dataset.sizes, kind="mergesort")
        indices = np.argsort(self.src_dataset.sizes[indices], kind="mergesort")
        return indices


class BinaryDatasetBuilder(object):
    def __init__(self, filename):
        self.filepath = filename
        self.fp = open(filename, 'wb')
        self.sizes = []

    def write(self, encoded_line):
        self.fp.write(encoded_line.tobytes())
        self.sizes.append(encoded_line.size)

    def finalize(self):
        self.fp.close()

        offset_file_path = self.filepath.replace("index", "offset")
        with open(offset_file_path, 'wb') as fp:
            fp.write(np.array(self.sizes, dtype=np.int32).tobytes())


def write_encoded_sentence(input_file_path, start_pos, end_pos, worker_id, dictionary):
    base_dir = os.path.dirname(input_file_path)

    with open(input_file_path, 'r', encoding='utf-8') as fr:
        dataset = BinaryDatasetBuilder(get_index_file(base_dir, worker_id))
        fr.seek(start_pos)
        _, line = safe_read(fr)
        while line:
            if end_pos < fr.tell() < end_pos + 2**32:
                break
            line = line.strip().split()
            encoded_line = dictionary.encode_sentence(line)
            dataset.write(encoded_line)
            line = fr.readline()
        dataset.finalize()








