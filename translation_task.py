from dataset import SingleDataset, PairDataset


class Translation(object):

    def __init__(self):
        self.datasets = {}

    def load_dataset(self, split, path):
        # 根据split找到路径
        src_dataset = SingleDataset(path)
        trg_dataset = SingleDataset(path)
        pair_dataset = PairDataset(src_dataset, trg_dataset)
        self.datasets[split] = pair_dataset

    def build_model(self):
        pass

