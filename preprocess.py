import argparse
import os
import shutil
from multiprocessing import Pool

from dataset import write_encoded_sentence
from dataset_utils import find_offsets, get_offset_file, get_index_file
from dictionary import Dictionary
from options import add_preprocess_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_preprocess_args(parser)
    args = parser.parse_args()

    dictionary = Dictionary.build_from_corpus(args.corpus-path)

    workers = 4 or os.cpu_count()
    offsets = find_offsets(args.corpus-path, workers)

    pool = Pool(processes=workers)
    for i in range(1, len(offsets)):
        pool.apply_async(write_encoded_sentence, args=(args.corpus-path, offsets[i - 1], offsets[i], i, dictionary))
    pool.close()
    pool.join()

    base_dir = os.path.dirname(args.corpus-path)
    merged_offset_file = get_offset_file(base_dir, 0)
    merged_idx_file = get_index_file(base_dir, 0)

    with open(merged_idx_file, 'wb') as f_idx_dst, open(merged_offset_file, 'wb') as f_offset_dst:
        for i in range(1, len(offsets)):
            idx_file = get_index_file(base_dir, i)
            offset_file = get_offset_file(base_dir, i)
            with open(idx_file, 'rb') as f_idx_src, open(offset_file, 'rb') as f_offset_src:
                shutil.copyfileobj(f_idx_src, f_idx_dst)
                shutil.copyfileobj(f_offset_src, f_offset_dst)