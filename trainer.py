import time
from collections import OrderedDict

import torch
import torch.distributed as dist

from dataset.dataset_iter import EpochIterator, ChunkIterator, BufferedIterator, CountIterator
from meters import TimeMeasure, CountMeasure
import meters


class Trainer(object):

    def __init__(self, args, task, model, criterion, optimizer):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.task = task
        self.model = model.to(device = self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_parallel_word_size = args.data_parallel_word_size
        self.args = args
        self.epoch = 1
        # {"epoch": 3, "update": 2.174, "loss": "9.368", "nll_loss": "8.846",
        #  # "ppl": "460.11", "wps": "14185.4", "ups": "0.5", "wpb": "28201.6", "bsz": "1117.8", "num_updates": "300",
        #  # "lr": "3.75925e-05", "gnorm": "1.724", "train_wall": "93", "wall": "0"}
        self.time_measure_statistics = ['wps', 'ups', 'train_wall', 'wall']
        self.count_measure_statistics = ['epoch', 'loss', 'nll_loss', 'ppl', 'update_percent',
                                         'wpb', 'bsz', 'num_updates', 'lr', 'gnorm']

        self.start_time = 0
        self.previous_training_time = 0
        self.cumulative_training_time = 0

    def get_train_iterator(self):
        epoch_itr = EpochIterator(self.task.dataset['train'])
        if self.args.prefetch:
            epoch_itr = BufferedIterator()
        if epoch_itr.epoch <= len(args.updating_data_chunk_size):
            data_chunk_size = args.updating_data_chunk_size[epoch_itr.epoch - 1]
        else:
            data_chunk_size = args.updating_data_chunk_size[-1]
        epoch_itr = ChunkIterator(epoch_itr, data_chunk_size)
        epoch_itr = CountIterator(epoch_itr)
        return epoch_itr

    def update_statistical_state_dict(self, state_dict, sync_logging_output_dict, grad_norm, epoch_iter, num_updates):

        state_dict["epoch"] = epoch_iter.count
        state_dict["update"] = epoch_iter.count + num_updates / len(epoch_iter)
        for key, val in sync_logging_output_dict.keys():
            state_dict[key].update_value(val)

        return state_dict

    def train_epoch(self):
        # outer loop
        with meters.build_field_statistics_hierarchical_container("train_epoch") as epoch_field_container:
            epoch_itr = self.get_train_iterator()
            num_updates = 0
            previous_num_updates = 0
            for i, samples in enumerate(epoch_itr):
                self.model.train()
                self.criterion.train()
                self.optimizer.zero_grad()
                logging_output, num_tokens = [], 0
                start_time = time.time()

                # inner loop
                with meters.build_field_statistics_hierarchical_container("train_step") as train_step_field_container:
                    for j, sample in enumerate(samples):
                        logits = self.model(sample)
                        loss, nll_loss = self.criterion(logits)
                        output_dict = {
                            'loss': loss.data,
                            'nll_loss': nll_loss.data,
                            'n_tokens': sample['n_tokens'],
                            'n_sentences': sample['target'].size(0)}
                        logging_output.append(output_dict)
                        num_tokens += sample['n_tokens']
                        loss.backward()

                    # sync inner loop result
                    every_update_train_time = time.time() - start_time
                    sync_logging_output_dict, sync_train_time = self.get_state_sum(logging_output, every_update_train_time)
                    average_every_update_train_time = sync_train_time / self.data_parallel_world_size

                    # optimize
                    self.optimizer.multiply_grads(self.data_parallel_world_size / num_tokens)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                    self.optimizer.step()
                    num_updates += 1

                    # {"epoch": 3, "update": 2.174, "loss": "9.368", "nll_loss": "8.846",
                    #  # "ppl": "460.11", "wps": "14185.4", "ups": "0.5", "wpb": "28201.6", "bsz": "1117.8", "num_updates": "300",
                    #  # "lr": "3.75925e-05", "gnorm": "1.724", "train_wall": "93", "wall": "0"}
                    # merge log_interval result and log to console
                    if num_updates % self.args.log_interval == 0:
                        state_dict = self.update_statistical_state_dict(state_dict,
                                                                        sync_logging_output_dict,
                                                                        grad_norm,
                                                                        epoch_itr,
                                                                        num_updates)
                    # save and validate condition
                    # 验证保存，最大的训练轮数和训练更新次数只能选择一个，逻辑捋清楚
                    # 终止条件：学习率到达某一特定值，最大训练轮数，最大训练更新数是非常不准的，一方面是更新计数可以作为参数自己调，比如
                    # 每隔几个batch更新一次参数，而batch的大小又可以作为参数自己调，看看fairseq到底是怎么处理的吧
                    # 到了一轮训练结束且训练轮数是要保存间隔的整数倍，更新次数大于等于最大更新次数
                    # 保存和停止条件分别判断
                    # 按优先级条件，最重要的是每一轮或者每隔一定的迭代次数进行验证，至于要不要保存可以灵活处理，如果在验证集上几次验证
                    # 都没有改变就可以终止了，
                    end_of_epoch = not epoch_itr.has_next()

                    do_validate = (end_of_epoch and epoch_itr.epoch % self.args.validate_epoch_interval == 0) or (
                            num_updates > 0
                            and self.args.validate_interval_updates > 0
                            and num_updates % self.args.validate_update_interval == 0
                            and self.args.validate_after_updates == 0)

                    do_save = do_validate and (end_of_epoch and epoch_itr.epoch % self.args.save_epoch_interval == 0) or (
                                num_updates > 0
                                and self.args.save_interval_updates > 0
                                and num_updates % self.args.save_update_interval == 0)


                    should_stop = num_updates >= self.args.max_update or epoch_itr.epoch >= self.args.max_epoch

    def validate(self):
        valid_itr = EpochIterator(self.task.dataset['valid'], shuffle=False)
        self.model.eval()

        logging_output, num_tokens, valid_start_time = [], 0, time.time()
        with meters.build_field_statistics_hierarchical_container("valid") as valid_field_container:
            for i, sample in enumerate(valid_itr):
                logits = self.model(sample)
                loss, nll_loss = self.criterion(logits)
                output_dict = {
                    'loss': loss.data,
                    'nll_loss': nll_loss.data,
                    'n_tokens': sample['n_tokens'],
                    'n_sentences': sample['target'].size(0)}
                logging_output.append(output_dict)
                num_tokens += sample['n_tokens']


            valid_time = time.time() - valid_start_time
            sync_logging_output_dict, sync_valid_time = self.get_state_sum(logging_output, valid_time)
            average_valid_time = sync_valid_time / self.data_parallel_world_size


    def get_state_sum(self, logging_output, train_time):
        stat_sum = OrderedDict({'train_time': train_time})

        for key in logging_output[0].keys():
            stat_sum[key] = sum([output_dict[key] for output_dict in logging_output])

        cpu_data = OrderedDict()
        gpu_data = OrderedDict()

        for k, v in stat_sum.items():
            if torch.is_tensor(v) and v.device.type == 'cuda':
                gpu_data[k] = v
            else:
                cpu_data[k] = torch.tensor(v, dtype=torch.double)

        sync_cpu_data = dist.all_reduce(torch.stack(cpu_data.values()))
        sync_gpu_data = dist.all_reduce(torch.stack(gpu_data.values()))

        sync_state = OrderedDict()
        for key in stat_sum.keys():
            if key in sync_cpu_data:
                sync_state[key] = sync_cpu_data[key]
            else:
                sync_state[key] = sync_gpu_data[key]

        sync_train_time = sync_state.pop('train_time')

        return sync_state, sync_train_time









