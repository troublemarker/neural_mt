import os

from torch import nn

from dataset.dataset import SingleDataset, PairDataset
from dictionary import Dictionary
from model import Transformer
from label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from optimizer.optim import CustomAdam

dict_path = "dict.{}.txt"
subset_path = "{}.{}-{}.{}"


class Translation(object):

    def __init__(self, args):
        super(Translation, self).__init__()
        self.datasets = {}
        self.data_dir = args.data_dir

        self.src_lang, self.trg_lang = dataset_utils.infer_language_pair(args.data_dir)

        src_dict_path = os.path.join(args.data_dir, dict_path.format(self.src_lang))
        trg_dict_path = os.path.join(args.data_dir, dict_path.format(self.trg_lang))
        self.src_dict = Dictionary.build_from_dict_file(src_dict_path)
        self.trg_dict = Dictionary.build_from_dict_file(trg_dict_path)

        self.model = None
        self.criterion = None
        self.optimizer = None

    def load_dataset(self, split):
        # 根据split找到路径
        src_split_path = os.path.join(self.data_dir,
                                      subset_path.format(split, self.src_lang, self.trg_lang, self.src_lang))
        trg_split_path = os.path.join(self.data_dir,
                                      subset_path.format(split, self.src_lang, self.trg_lang, self.trg_lang))

        src_dataset = SingleDataset(src_split_path)
        trg_dataset = SingleDataset(trg_split_path)
        pair_dataset = PairDataset(src_dataset, trg_dataset)
        self.datasets[split] = pair_dataset

    def build_model(self, args):
        encoder_embed_tokens = nn.Embedding(self.src_dict.token_num,
                                            args.encoder_embed_dim,
                                            padding_idx=self.src_dict.padding_idx)
        if args.share_all_embeddings:
            decoder_embed_tokens = encoder_embed_tokens
        else:
            decoder_embed_tokens = nn.Embedding(self.trg_dict.token_num,
                                                args.decoder_embed_dim,
                                                padding_idx=self.trg_dict.padding_idx)
        self.model = Transformer(args, self.src_dict, self.trg_dict)

    def build_criterion(self, label_smooth):
        self.criterion = LabelSmoothedCrossEntropyCriterion(label_smooth)

    def build_optimizer(self):
        if self.model is None:
            print("should build model first!")
        else:
            self.optimizer = CustomAdam(self.model.parameters(), lr=self.args.lr, betas=self.args.betas)






def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)


def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


def transformer_wmt_en_de(args):
    base_architecture(args)