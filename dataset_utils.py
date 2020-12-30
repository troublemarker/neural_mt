import os

import torch


def get_index_file(base_dir, num=0):
    return os.path.join(base_dir, "%d_index.bin" % num)


def get_offset_file(base_dir, num=0):
    return os.path.join(base_dir, "%d_offset.bin" % num)


def safe_read(f):
    pos = f.tell()
    while True:
        try:
            line = f.readline()
            break
        except UnicodeError:
            pos -= 1
            f.seek(pos)
    return pos, line


def find_offsets(filename, chunks):
    offsets = [0] * (chunks + 1)
    with open(filename, 'r', encoding='utf-8') as f:
        size = os.fstat(f.fileno()).st_size
        offsets[-1] = size
        chunk_size = size // chunks
        for i in range(0, chunks):
            f.seek(i * chunk_size)
            pos, _ = safe_read(f)
            offsets[i] = pos
    return offsets


def collate(samples):
    max_length = max([sample.size() for sample in samples])
    padding_samples = torch.zeros((len(samples), max_length[0]), dtype=torch.int32)
    for idx, sample in enumerate(samples):
        padding_samples[idx][:len(sample)].copy_(sample)
    return padding_samples
