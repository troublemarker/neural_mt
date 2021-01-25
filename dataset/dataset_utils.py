import os

import torch


def collate_tokens(samples, padding_idx, bos_idx, insert_bos_to_beginning=False):
    max_len = max([sample.size() for sample in samples])
    collated_batches = torch.zeros([len(samples), max_len]) * padding_idx

    for i, sample in enumerate(samples):
        collated_batches[i, :len(sample)] = sample
        if insert_bos_to_beginning:
            collated_batches[i, 1:len(sample)] = sample[:-1]
        else:
            collated_batches[i, :len(sample)] = sample

    if insert_bos_to_beginning:
        collated_batches[:, 0] = bos_idx

    return collated_batches


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


def infer_language_pair(path):
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')



