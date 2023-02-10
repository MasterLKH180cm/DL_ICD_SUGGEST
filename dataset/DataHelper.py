import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


class Vocabulary(object):

    def __init__(self):
        self.code2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2code = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.PAD_ID = 2
        self.SOS_ID = 0
        self.EOS_ID = 1
        self.UNK_ID = 3
        self.max_length = 10
        self.vocab_size = 4
        self.code_list = []
    def build_vocab(self, data_path):
        """Construct the relation between codes and indices"""
        with open(data_path, 'rb') as dataset:
            dataset = pd.read_excel(dataset, header=None,
                                    dtype=str)[0].to_list()
            for code in dataset:
                # print(code)
                if data_path=='dataset/specific_atc_list.xlsx':
                    code = code[:5]
                self.code2idx[code] = self.vocab_size
                self.idx2code[self.vocab_size] = code
                self.vocab_size += 1
                # print(self.code2idx, self.idx2code)

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=True):
        """Transform a code sequence to index sequence
            :param sequence: a string composed with codes
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.code2idx['SOS']] if add_sos else []
        for code in sequence:
            if code not in self.code2idx:
                # index_sequence.append((self.code2idx['UNK']))
                continue
            else:
                index_sequence.append(self.code2idx[code])

        if add_eos:
            index_sequence.append(self.code2idx['EOS'])
        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = []
        for idx in indices:
            code = self.idx2code[idx]
            if code == "EOS":
                # break
                continue
            else:
                sequence += [code]
        return sequence

    # def split_sequence(self, sequence):
    #     """Vary from languages and tasks. In our task, we simply return codes in given sentence
    #     For example:
    #         Input : alphabet
    #         Return: [a, l, p, h, a, b, e, t]
    #     """
    #     return [code for code in sequence]

    def __str__(self):
        str = "Vocab information:\n"
        for idx, code in self.idx2code.items():
            str += "code: %s Index: %d\n" % (code, idx)
        return str

    def pad_sequence(self, sequence, max_length):
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence

    def evaluation_batch(self, codes):
        """
        Prepare a batch of var for evaluating
        :param codes: a list, store the testing data 
        :return: evaluation_batch
        """
        evaluation_batch = []

        for code in codes:
            indices_seq = self.vocab.sequence_to_indices(code, add_eos=True)
            evaluation_batch.append([indices_seq])

        seq_pairs = sorted(evaluation_batch,
                           key=lambda seqs: len(seqs[0]),
                           reverse=True)
        input_seqs = [pair[0] for pair in seq_pairs]
        input_lengths = [len(s) for s in input_seqs]
        in_max = input_lengths[0]
        input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(
            0, 1)  # time * batch

        if self.use_cuda:
            input_var = input_var.cuda()

        return input_var, input_lengths


class DataTransformer(object):

    def __init__(self, path, use_cuda):
        self.indices_sequences = []
        self.use_cuda = use_cuda

        # Load and build the vocab
        self.vocab = Vocabulary()
        self.vocab.build_vocab(path)
        self.PAD_ID = self.vocab.code2idx["PAD"]
        self.SOS_ID = self.vocab.code2idx["SOS"]
        self.vocab_size = self.vocab.vocab_size
        self.max_length = self.vocab.max_length

        self._build_training_set(path)

    def _build_training_set(self, path):
        # Change sentences to indices, and append <EOS> at the end of all pairs
        for code in self.vocab.code_list:
            indices_seq = self.vocab.sequence_to_indices(code, add_eos=True)
            # input and target are the same in auto-encoder
            self.indices_sequences.append([indices_seq, indices_seq[:]])

    def mini_batches(self, batch_size):
        input_batches = []
        target_batches = []

        np.random.shuffle(self.indices_sequences)
        mini_batches = [
            self.indices_sequences[k:k + batch_size]
            for k in range(0, len(self.indices_sequences), batch_size)
        ]

        for batch in mini_batches:
            seq_pairs = sorted(batch,
                               key=lambda seqs: len(seqs[0]),
                               reverse=True)  # sorted by input_lengths
            input_seqs = [pair[0] for pair in seq_pairs]
            target_seqs = [pair[1] for pair in seq_pairs]

            input_lengths = [len(s) for s in input_seqs]
            in_max = input_lengths[0]
            input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]

            target_lengths = [len(s) for s in target_seqs]
            out_max = target_lengths[0]
            target_padded = [
                self.pad_sequence(s, out_max) for s in target_seqs
            ]

            input_var = Variable(torch.LongTensor(input_padded)).transpose(
                0, 1)  # time * batch
            target_var = Variable(torch.LongTensor(target_padded)).transpose(
                0, 1)  # time * batch

            if self.use_cuda:
                input_var = input_var.cuda()
                target_var = target_var.cuda()

            yield (input_var, input_lengths), (target_var, target_lengths)

    def pad_sequence(self, sequence, max_length):
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence

    def evaluation_batch(self, codes):
        """
        Prepare a batch of var for evaluating
        :param codes: a list, store the testing data 
        :return: evaluation_batch
        """
        evaluation_batch = []

        for code in codes:
            indices_seq = self.vocab.sequence_to_indices(code, add_eos=True)
            evaluation_batch.append([indices_seq])

        seq_pairs = sorted(evaluation_batch,
                           key=lambda seqs: len(seqs[0]),
                           reverse=True)
        input_seqs = [pair[0] for pair in seq_pairs]
        input_lengths = [len(s) for s in input_seqs]
        in_max = input_lengths[0]
        input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(
            0, 1)  # time * batch

        if self.use_cuda:
            input_var = input_var.cuda()

        return input_var, input_lengths


if __name__ == '__main__':
    vocab = Vocabulary()
    vocab.build_vocab('atc_list.xlsx')
    print(vocab)

    test = ['S01XA95', 'C09AA05', 'L02BB01']
    print("Sequence before transformed:", test)
    ids = vocab.sequence_to_indices(test)
    print("Indices sequence:", ids)
    sent = vocab.indices_to_sequence(ids)
    print("Sequence after transformed:", sent)

    data_transformer = DataTransformer('atc_list.xlsx', use_cuda=False)

    for ib, tb in data_transformer.mini_batches(batch_size=3):
        print("B0-0")
        print(ib, tb)
        break
