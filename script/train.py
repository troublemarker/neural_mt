import argparse
import random
import logging

import numpy as np
import torch
import torch.distributed as dist

from translation_task import Translation
from trainer import Trainer


logger = logging.getLogger(__name__)


def main(i, args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device_id = i
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    init_method = "tcp://127.0.0.1:{}".format(random.randint(10000, 20000))
    dist.init_process_group(backend='NCCl',
                            init_method=init_method,
                            world_size=args.distributed_world_size,
                            rank=i)
    logger.info("distributed init rank {}: {}".format(dist.get_rank(), init_method))


    # 定义的数据集格式为 "{split}.{src}-{trg}.{src}.{bin}"
    # 字典的定义格式为“{dict}.{src}.txt”
    # 搞清楚它定义的bin和idx到底是什么文件后缀
    trans_task = Translation(args)
    trans_task.load_dataset('split')
    trans_task.build_model(args)
    trans_task.build_criterion(args)

    lr = trans_task.optimizer.param_groups[0]['lr']

    # validation_every_num_updates更新次数验证频率，validation_every_num_epoch更新轮数验证频率，
    # validation_after_num_updates多少次更新后验证，log_every_num_updates记录间隔，
    # save_every_num_updates更新轮数保存间隔，save_every_num_epoch更新次数保存间隔，
    # updating_data_chunk_size
    # 清空cuda缓存步骤

    trainer = Trainer()
    while lr > args.min_lr and trainer.epoch <= args.max_epoch:
        should_stop, lr = trainer.train_epoch()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    args_postprocess(args)

    torch.multiprocessing.spawn(fn=main,
                                args=(args,),
                                nprocs=args.distributed_world_size)