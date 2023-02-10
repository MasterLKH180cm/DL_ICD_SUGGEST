import random

import torch
import torch.nn as nn
from torch.autograd import Variable


class Seq2Seq(nn.Module):

    def __init__(self, encoder_1, encoder_2, encoder_3, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2
        self.encoder_3 = encoder_3
        self.decoder = decoder

    def forward(self, inputs_1, inputs_2, inputs_3, targets):
        input_vars, input_lengths = inputs_1
        encoder_outputs, encoder_hidden_1 = self.encoder_1.forward(
            input_vars, input_lengths)
        input_vars, input_lengths = inputs_2
        encoder_outputs, encoder_hidden_2 = self.encoder_2.forward(
            input_vars, input_lengths)
        input_vars, input_lengths = inputs_3
        encoder_outputs, encoder_hidden_3 = self.encoder_3.forward(
            input_vars, input_lengths)
        # print(inputs_1[0].shape, inputs_2[0].shape, inputs_3[0].shape)
        encoder_hidden = encoder_hidden_1 + encoder_hidden_2 + encoder_hidden_3
        # encoder_hidden = torch.cat((encoder_hidden_1,encoder_hidden_2, encoder_hidden_3), dim=2)
        # print(encoder_hidden_1.shape, encoder_hidden_2.shape,
        #       encoder_hidden_3.shape)
        decoder_outputs, decoder_hidden = self.decoder.forward(
            context_vector=encoder_hidden, targets=targets)
        # print(decoder_outputs.shape)
        return decoder_outputs, decoder_hidden

    def evaluate(self, inputs_1, inputs_2, inputs_3):
        input_vars, input_lengths = inputs_1
        encoder_outputs, encoder_hidden_1 = self.encoder_1(
            input_vars, input_lengths)
        input_vars, input_lengths = inputs_2
        encoder_outputs, encoder_hidden_2 = self.encoder_2(
            input_vars, input_lengths)
        input_vars, input_lengths = inputs_3
        encoder_outputs, encoder_hidden_3 = self.encoder_3(
            input_vars, input_lengths)
        # print(inputs_1[0].shape, inputs_2[0].shape, inputs_3[0].shape)
        encoder_hidden = encoder_hidden_1 + encoder_hidden_2 + encoder_hidden_3
        # encoder_hidden = torch.cat((encoder_hidden_1,encoder_hidden_2, encoder_hidden_3), dim=2)
        decoded_sentence = self.decoder.evaluate(context_vector=encoder_hidden)
        return decoded_sentence
