import os
import shutil
from multiprocessing import Pool

from torch.utils.data import DataLoader
from dataset import write_encoded_sentence, SingleDataset
from dataset_utils import find_offsets, get_offset_file, get_index_file, collate
from dictionary import Dictionary


file_path = r"C:\Users\53130\Downloads\en.txt"

if __name__ == '__main__':


    base_dir = os.path.dirname(file_path)
    dataset = SingleDataset(base_dir)
    batches = dataset.batch_by_size(4096)

    data_loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate)
    print(len(data_loader))
    for i, data in enumerate(data_loader):
        print(i, len(data))


    # TODO: 1 fairseq看数据预处理和数据加载还有哪些要注意的参数和流程 2 开始写模型 3 有些代码可以优化 4 从一开始就要有宏观规划
    # TODO：5 加载两个数据集的代码还没完成



