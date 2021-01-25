import argparse

import torch

def args_postprocess(args):
    assert (args.max_tokens is not None or args.max_sentences is not None), \
        "must specify one of max_tokens and max_sentences"

def add_train_args(parser):
    pass

def add_general_args(parser):
    parser.add_argument('data-dir', help="")

def add_preprocess_args(parser):

    parser.add_argument('--corpus-path', '-i', help="")
    parser.add_argument('--output-dir', '-o', help="")


def add_dataset_args(parser, mode):
    parser.add_argument('--data-buffer-size', default=10, type=int, metavar='N',
                        help='number of batches to preload')
    parser.add_argument('--max-tokens', type=int, metavar='N',
                        help='maximum number of tokens in a batch')
    parser.add_argument('--required-batch-size-multiple', default=8, type=int, metavar='N',
                        help='batch size will either be less than this value, '
                             'or a multiple of this value')
    parser.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    if mode == 'train':
        parser.add_argument('--validate-interval', type=int, default=1, metavar='N',
                           help='validate every N epochs')
        parser.add_argument('--validate-interval-updates', type=int, default=0, metavar='N',
                           help='validate every N updates')



def add_transformer_args(parser):
    parser.add_argument('--dropout', type=float, help="")
    parser.add_argument('--activation-dropout', type=float, help="")
    parser.add_argument('--attention-dropout', type=float, help="")

    parser.add_argument('--encoder-embed-dim', type=int, help="")
    parser.add_argument('--encoder-ffn-embed-dim', type=int, help="")
    parser.add_argument('--encoder-layers', type=int, metavar='N',
                        help='num encoder layers')
    parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                        help='num encoder attention heads')
    parser.add_argument('--encoder-normalize-before', action='store_true',
                        help='apply layernorm before each encoder block')

    parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension')
    parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers', type=int, metavar='N',
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                        help='num decoder attention heads')
    parser.add_argument('--decoder-normalize-before', action='store_true',
                        help='apply layernorm before each decoder block')
    parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                        help='decoder output dimension (extra linear layer '
                             'if different from decoder embed dim')


def add_distribute_args(parser):
    parser.add_argument('--distributed-world-size', type=int, metavar='N',
                        default=torch.cuda.device_count(),
                        help='total number of GPUs (default: all visible GPUs)')


def add_optimization_args(parser):

    parser.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    parser.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    parser.add_argument('--stop-time-hours', default=0, type=float, metavar='N',
                       help='force stop training after specified cumulative time (if >0)')
    parser.add_argument('--clip-norm', default=0.0, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    parser.add_argument('--update-freq', default='1', metavar='N1,N2,...,N_K',
                       help='update parameters every N_i batches, when in epoch i')
    parser.add_argument('--lr', '--learning-rate', default='0.25', type=eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    parser.add_argument('--min-lr', default=-1, type=float, metavar='LR',
                       help='stop training when the learning rate reaches this minimum')
    parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                        help='warmup the learning rate linearly for the first N updates')
    parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                        help='initial learning rate during warmup phase; default is args.lr')