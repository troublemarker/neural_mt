import torch


# 输入：[batch_size beam_size]
# 1 经过embedding，transformer decoder 输出logits [batch_size beam_size token_score]
# 2 对每个batch的每个beam所有token生成的分数进行排序，选出最高的beam_size个[batch_size beam_size beam_size_score]，分别记录其分数
#   和index，加上之前的[batch_size beam_size]存放的分数，[batch_size beam_size beam_size_score]，index也拼上之前的
#   [batch_size beam_size token_index]
# 3 在beam轴上进行排序，选出最高的beam_size个token，记录其index
#   具体地，变换成[batch_size beam_size * token_score]数组，排序返回其前beam_size个index，index对token_size取整是beam_index，
#   index对token_size取余是token_index，从[batch_size beam_size beam_size_score]按index选择出来[batch_size beam_size]分数，
#   从[batch_size beam_size token_index]选择出新的句子。
#   检查最后一个token是否为eos，如果是，记录它所在的batch和beam位置，下次再生成
#   下次再生成的时候还可以继续输入模型，但是预测出的[batch_size beam_size token_index]选出最高的beam_size个后，再进行排序就要考虑EOS的
#   情况了，先把EOS位置的分数置为负无穷，而后再进行排序，后面操作同理，
# 4 遇到EOS字符的句子，逻辑上剩下beam_size-1个句子继续搜索，排序；至于程序上如何处理再思考。
# 5 所有的句子都结束了，将分数最高的句子输出。
torch.tensor().size()

class BeamSearch(object):
    def __init__(self, task, args):
        self.task = task
        self.args = args
        self.beam_size = self.args.beam_size
        self.padding_idx = self.task.trg_dict.padding_idx

    def search(self, src_tokens):
        batch_size = src_tokens.size()[0]
        token_num = src_tokens.size()[1]
        seq_lprobs = torch.zeros([batch_size, self.beam_size], dtype=torch.float32)
        seq_eos_flag = torch.zeros([batch_size, self.beam_size])
        target_sequence = torch.ones([batch_size, self.beam_size, 1], dtype=torch.float32) * self.padding_idx

        logits = self.task.model(src_tokens)
        lprobs = self.taks.criterion(logits)  # [batch_size beam_size token_num]
        top_k_lprobs_token_index = lprobs.argsort(dim=-1, descending=True)[:, :, self.beam_size]
        top_k_lprobs = lprobs.sort(dim=-1, descending=True)[:, :, self.beam_size]

        # seq_lprobs
        seq_lprobs = seq_lprobs.unsqueeze(dim=-1) + top_k_lprobs
        # target_sequence = target_sequence.unsqueeze(dim=-1) + top_k_lprobs_token_index
        seq_lprobs.masked_fill_(seq_eos_flag, -float('inf'))
        top_beam_size_index = seq_lprobs.view(batch_size, -1).argsort(dim=-1, descending=True)[:, self.beam_size]
        beam_index = top_beam_size_index // token_num
        token_index = top_beam_size_index % token_num

        # TODO: acordding to beam_index and token_index, select from lprobs and seq_lprobs to get [batch_size beam_size 1].
        # TODO: concat with target_sequence
        torch.index_select(lprobs, 0, indices)
        target_sequence[:, [beam_index], :] +


        # 要把beam和token能选出来






