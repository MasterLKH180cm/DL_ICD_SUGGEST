import random

import torch
import torch.nn as nn
from torch.autograd import Variable


class VanillaDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_length,
                 teacher_forcing_ratio, sos_id, use_cuda):
        """Define layers for a vanilla rnn decoder"""
        super(VanillaDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.in_1 = nn.Linear(hidden_size, hidden_size)
        self.act_1 = nn.PReLU()
        self.in_2 = nn.Linear(hidden_size, hidden_size)
        self.act_2 = nn.PReLU()
        self.embedding = nn.Embedding(output_size, hidden_size,scale_grad_by_freq =True)
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)
        # self.Sigmoid = nn.Sigmoid()

        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_id = sos_id
        self.use_cuda = use_cuda

    # def forward_step(self, inputs, hidden):
    #     # inputs: (time_steps=1, batch_size)
    #     batch_size = inputs.size(1)
    #     embedded = self.embedding(inputs)
    #     embedded.view(1, batch_size, self.hidden_size)  # S = T(1) x B x N
    #     rnn_output, hidden = self.gru(embedded, hidden)  # S = T(1) x B x H
    #     rnn_output = rnn_output.squeeze(0)  # squeeze the time dimension
    #     output = self.log_softmax(self.out(rnn_output))  # S = B x O
    #     return output, hidden

    def forward(self, context_hidden, context_output):

        # Prepare variable for decoder on time_step_0
        batch_size = context_hidden.size(1)

        # Pass the context vector
        context_output = self.act_1(self.in_1(context_output))
        context_output = self.act_2(self.in_2(context_output))
        # Unfold the decoder RNN on the time dimension
        
        context_output, context_hidden = self.gru(context_output, context_hidden)  # S = T(1) x B x H
        context_output = context_output.sum(0).squeeze(0)  # squeeze the time dimension
        context_output = self.out(context_output)  # S = B x O
        

        return context_output, context_hidden

    def evaluate(self, context_hidden, context_output):
        batch_size = context_hidden.size(1)  # get the batch size
        # Pass the context vector
        context_output = self.act_1(self.in_1(context_output))
        context_output = self.act_2(self.in_2(context_output))
        # Unfold the decoder RNN on the time dimension
        
        context_output, context_hidden = self.gru(context_output, context_hidden)  # S = T(1) x B x H
        context_output = context_output.sum(0).squeeze(0)  # squeeze the time dimension
        context_output = self.out(context_output)  # S = B x O
        

        return context_output

    def _decode_to_index(self, decoder_output):
        """
        evaluate on the logits, get the index of top1
        :param decoder_output: S = B x V or T x V
        """
        value, index = torch.topk(decoder_output, 1)
        index = index.transpose(0,
                                1)  # S = 1 x B, 1 is the index of top1 class
        if self.use_cuda:
            index = index.cuda()
        return index

    def _decode_to_indices(self, decoder_outputs):
        """
        Evaluate on the decoder outputs(logits), find the top 1 indices.
        Please confirm that the model is on evaluation mode if dropout/batch_norm layers have been added
        :param decoder_outputs: the output sequence from decoder, shape = T x B x V
        """
        decoded_indices = []
        batch_size = decoder_outputs.size(1)
        decoder_outputs = decoder_outputs.transpose(0, 1)  # S = B x T x V

        for b in range(batch_size):
            top_ids = self._decode_to_index(decoder_outputs[b])
            decoded_indices.append(top_ids.data[0].cpu().numpy())
        return decoded_indices
