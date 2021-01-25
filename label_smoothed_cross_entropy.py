import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothedCrossEntropyCriterion(nn.Module):

    def __init__(self, label_smoothing):
        self.eps = label_smoothing

    def forward(self, logits, targets, padding_idx):
        """
        :param logits: [B, T, C]
        :param targets: [B, T]
        :param padding_idx:
        """
        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = targets.view(-1, 1)

        # TODO：lprobs在推断的时候也要输出进行beam_search排序，所以要返回
        nll_loss = -lprobs.gather(dim=-1, index=targets).squeeze()
        smooth_loss = -lprobs.sum(dim=-1, keepdim=False)
        padding_mask = targets.eq(padding_idx)
        nll_loss.masked_fill_(padding_mask, 0.)
        smooth_loss.masked_fill_(padding_mask, 0.)
        loss = (1 - self.eps) * nll_loss.sum() + self.eps / lprobs.size(-1) * smooth_loss.sum()

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'n_tokens': sample['ntokens'],
            'n_sentences': sample['target'].size(0),
        }

        return logging_output



