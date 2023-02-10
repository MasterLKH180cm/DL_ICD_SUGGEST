import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size):
        """Define layers for a vanilla rnn encoder"""
        super(VanillaEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size,scale_grad_by_freq =True)
        self.gru = nn.GRU(embedding_size, output_size,num_layers=1)
        self.out_1 = nn.Linear(output_size, output_size)
        self.act_1 = nn.PReLU()
        self.out_2 = nn.Linear(output_size, output_size)
        self.act_2 = nn.PReLU()

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded,
                                      input_lengths,
                                      enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed, hidden)
        hidden = self.act_1(self.out_1(hidden))
        hidden = self.act_2(self.out_2(hidden))
        outputs, output_lengths = pad_packed_sequence(packed_outputs)
        return outputs, hidden

    def forward_a_sentence(self, inputs, hidden=None):
        """Deprecated, forward 'one' sentence at a time which is bad for gpu utilization"""
        embedded = self.embedding(inputs)
        outputs, hidden = self.gru(embedded, hidden)
        hidden = self.act_1(self.out_1(hidden))
        hidden = self.act_2(self.out_2(hidden))
        return outputs, hidden
