import os
from collections import Counter
from typing import List

import numpy as np

# 字典应该有哪些方法和属性？
# 属性应该包括：token，index字典，特殊字符
# 方法应该包括：编码句子，保存字典文件
# 类方法应该包括: 读取字典文件，新建立字典文件
# 初始化方法如果给路径，那以后从文件里读取字典应该如何初始化？
# 在read and write里处理同样的逻辑，得出symbol和counts，传给构造方法，构造方法里再建立字典
class Dictionary(object):

    def __init__(self, symbol_count, bos='<s>', eos='</s>', unk='<unk>', pad='<pad>'):
        self.symbol_counts = symbol_count
        self.num_dict = len(symbol_count)
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.special_symbols = [self.bos, self.eos, self.unk, self.pad]
        self.n_special = len(self.special_symbols)

        self.token2idx = self._build_dict()

    def add_symbol(self, symbol):
        self.symbol_counts[symbol] += 1

    @property
    def padding_idx(self):
        return self.token2idx[self.pad]

    @property
    def bos_idx(self):
        return self.token2idx[self.bos]

    @property
    def token_num(self):
        return len(self.token2idx)

    def _build_dict(self):
        token2idx = {}
        for i, symbol in enumerate(self.special_symbols):
            token2idx[symbol] = len(token2idx)

        sorted_symbols = self.symbol_counts.most_common()
        for symbol, count in sorted_symbols:
            token2idx[symbol] = len(token2idx)
        return token2idx

    def encode_sentence(self, sentence: List):
        """
        :param sentence: tokenized sentence
        """
        encoded_sent = np.empty(len(sentence), dtype=np.int32)

        for i, token in enumerate(sentence):
            if token in self.symbols:
                encoded_sent[i] = self.token2idx[token]
            else:
                encoded_sent[i] = self.token2idx[self.unk]
        return encoded_sent

    def save(self, data_dir, lang_prefix):
        dest_path = data_dir + "dict." + lang_prefix
        os.mkdir(dest_path)
        with open(dest_path, 'w', encoding='utf-8') as f:
            for sym, count in self.token2idx.items():
                if sym in self.special_symbols:
                    continue
                f.write(sym + " " + count + "\n")

    @classmethod
    def build_from_corpus(cls, file_path):
        symbol_counts = Counter()
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                symbol_counts.update(line)
        return cls(symbol_counts)

    @classmethod
    def build_from_dict_file(cls, dict_file):
        with open(dict_file, 'r', encoding="utf-8") as f:
            token2idx = Counter()
            for line in f:
                line = line.strip().split()
                token2idx[line[0]] = int(line[1])
        return cls(token2idx)

    def __len__(self):
        return self.num_dict



