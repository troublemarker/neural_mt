import os

import torch
import numpy as np
from torch.utils.data import Dataset

from dataset.dataset_utils import safe_read, get_index_file, collate_tokens

nptype_code = {np.int32: 1}

# bin是它的数据文件，idx是它存放size大小的文件


class SingleDataset(Dataset):

    def __init__(self, split_path):
        self.sizes, self.pointers = self._load_index(split_path + ".idx")
        self.data = self._load_data(split_path + ".bin")

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
    def __init__(self, src_dataset, trg_dataset, src_dict, trg_dict):
        self.src_dataset = src_dataset
        self.trg_dataset = trg_dataset
        self.src_dict = src_dict
        self.trg_dict = trg_dict

    def __getitem__(self, index):

        return {"id": index, "src_item": self.src_dataset[index], "trg_item": self.trg_dataset[index]}

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
        # if self.shuffle:
        #     indices = np.random.permutation(len(self)).astype(np.int64)
        # else:
        #     indices = np.arange(len(self), dtype=np.int64)
        # if self.buckets is None:
        #     # sort by target length, then source length
        #     if self.tgt_sizes is not None:
        #         indices = indices[
        #             np.argsort(self.tgt_sizes[indices], kind='mergesort')
        #         ]
        #     return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        indices = np.argsort(self.trg_dataset.sizes, kind="mergesort")
        indices = np.argsort(self.src_dataset.sizes[indices], kind="mergesort")
        return indices

    def collater(self, samples, padding_idx, eos_idx):
        # 原代码中对长度进行降序排列
        id = torch.LongTensor([s['id'] for s in samples])
        src_padding_batches = collate_tokens(samples["src_item"], self.src_dict.padding_idx, self.src_dict.bos_idx)
        trg_padding_batches = collate_tokens(samples["trg_item"], self.trg_dict.padding_idx, self.trg_dict.bos_idx)
        trg_input_padding_batches = collate_tokens(samples["trg_item"],
                                                   self.trg_dict.padding_idx,
                                                   self.trg_dict.bos_idx)
        n_tokens = trg_padding_batches.ne(padding_idx).sum()

        return {'id': id,
                'n_tokens': n_tokens,
                'n_sentences': len(samples),
                'src_batch': src_padding_batches,
                'trg_batch': trg_padding_batches,
                'trg_input_batch': trg_input_padding_batches}


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








