import pdb
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
from optimizer import optimizer, rangerlars
from config import config
from dataset.DataHelper import Vocabulary
from model.Decoder import VanillaDecoder
from model.Encoder import VanillaEncoder
from model.Seq2Seq import Seq2Seq


class Trainer(object):

    def __init__(self,
                 model,
                 atc_vocab,
                 pro_vocab,
                 lab_vocab,
                 diag_vocab,
                 train_loader,
                 test_loader,
                 learning_rate,
                 use_cuda,
                 checkpoint_name=config.checkpoint_name,
                 teacher_forcing_ratio=config.teacher_forcing_ratio):

        self.model = model
        # record some information about dataset
        self.atc_vocab = atc_vocab
        self.pro_vocab = pro_vocab
        self.lab_vocab = lab_vocab

        self.diag_vocab = diag_vocab
        self.vocab_size = self.diag_vocab.vocab_size
        self.PAD_ID = self.diag_vocab.PAD_ID
        self.use_cuda = use_cuda
        # data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        # record some information about dataset

        self.use_cuda = use_cuda
        self.epoch = 0
        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer = optimizer.Ranger(self.model.parameters(),
                                          lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.PAD_ID,
                                          reduction='mean')

        self.checkpoint_name = checkpoint_name
        self.eval_decode = []
        self.eval_atc = []
        self.eval_pro = []
        self.eval_lab = []
        self.eval_ans = []
        self.tpr = []
        self.tprs = []
        self.acc = []
        self.accs = []

    def train(self, num_epochs, pretrained=False):

        if pretrained:
            self.load_model()

        step = 0
        print(num_epochs)
        self.model.train()
        for self.epoch in range(0, num_epochs):
            # mini_batches = self.train_loader(batch_size=batch_size)
            print(self.epoch)
            self.model.train()
            for iters, (atc, pro, lab, diag) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                atc = (atc[0].cuda(), atc[1])
                pro = (pro[0].cuda(), pro[1])
                lab = (lab[0].cuda(), lab[1])
                diag = (diag[0].cuda(), diag[1])
                decoder_outputs, decoder_hidden = self.model(
                    atc, pro, lab, diag)

                # calculate the loss and back prop.

                cur_loss = self.get_loss(decoder_outputs, diag[0])
                # print('epoch: {}, iters: {}, loss: {}'.format(
                #     self.epoch, iters, cur_loss))
                # logging
                step += 1
                if step % 50 == 0:
                    print("Step:", step, "char-loss: ", cur_loss.data)
                    # self.save_model('step' + str(step))
                cur_loss.backward()

                # optimize
                self.optimizer.step()
            self.epoch += 1

            self.save_model('epoch' + str(self.epoch))
            self.model.eval()
            with torch.no_grad():
                for iters, (atc, pro, lab, diag) in enumerate(self.test_loader):
                    atc = (atc[0].cuda(), atc[1])
                    pro = (pro[0].cuda(), pro[1])
                    lab = (lab[0].cuda(), lab[1])
                    diag = (diag[0].cuda(), diag[1])
                    diag = diag[0].T.tolist()
                    decoded_sentence = self.model.evaluate(atc, pro, lab)
                    # print(decoded_sentence, diag)
                    self.eval_decode += decoded_sentence
                    self.eval_ans += diag
                    
            self.eval_tpr()
            self.eval_acc()
            self.tprs += [np.mean(self.tpr)]
            self.accs += [np.mean(self.acc)]
            print('tprs: ', self.tprs)
            print('accs: ', self.accs)
            self.tpr.clear()
            self.acc.clear()
            self.eval_decode.clear()
            self.eval_ans.clear()
            pd.DataFrame({
                'tpr': self.tprs,
                'acc': self.accs
            }).to_excel(r'./tpr_acc.xlsx', index=False)

    def masked_nllloss(self):
        # Deprecated in PyTorch 2.0, can be replaced by ignore_index
        # define the masked NLLoss
        weight = torch.ones(self.vocab_size)
        weight[self.PAD_ID] = 0
        if self.use_cuda:
            weight = weight
        return torch.nn.NLLLoss(weight=weight)

    def get_loss(self, decoder_outputs, targets):
        b = decoder_outputs.size(1)
        t = decoder_outputs.size(0)
        targets = targets.contiguous().view(-1)  # S = (B*T)
        decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (B*T) x V
        return self.criterion(decoder_outputs, targets)

    def save_model(self, label):
        torch.save(self.model.state_dict(),
                   './model/' + label + '_' + self.checkpoint_name)
        print("Model has been saved as " + label + '_' + self.checkpoint_name)

    def load_model(self, name):
        self.model.load_state_dict(
            torch.load(name, map_location='cpu'))
        print("Pretrained model has been loaded.\n")

    def tensorboard_log(self):
        pass

    def evaluate(self, words):
        # make sure that words is list
        if type(words) is not list:
            words = [words]
        self.model.eval()
        # transform word to index-sequence
        eval_var = self.diag_vocab.evaluation_batch(words=words)
        decoded_indices = self.model.evaluate(eval_var)
        results = []
        for indices in decoded_indices:
            results.append(self.diag_vocab.indices_to_sequence(indices))
        
        return results

    def eval_tpr(self):

        for a, b in zip(self.eval_decode, self.eval_ans):
            # print(a, b)
            # print(self.diag_vocab.indices_to_sequence(a.tolist()), self.diag_vocab.indices_to_sequence(b))
            a, b = set(a.tolist()), set(b)
            a.discard(0)
            a.discard(1)
            a.discard(2)
            a.discard(3)
            b.discard(0)
            b.discard(1)
            b.discard(2)
            b.discard(3)
            # print(a, b)
            self.tpr += [len(a.intersection(b)) / len(b)]
    def eval_acc(self):

        for a, b in zip(self.eval_decode, self.eval_ans):
            a, b = set(a.tolist()), set(b)
            a.discard(0)
            a.discard(1)
            a.discard(2)
            a.discard(3)
            b.discard(0)
            b.discard(1)
            b.discard(2)
            b.discard(3)
            # print(a, b)
            self.acc += [len(a.intersection(b)) > 0]
    def idx2code(self):
        atcs, pros, labs, diags, outputs = [], [], [], [], []
        for a, b, c, d, e in zip(self.eval_decode, self.eval_ans, self.eval_atc, self.eval_pro, self.eval_lab):
            outputs += [self.diag_vocab.indices_to_sequence(a.tolist())]
            diags += [self.diag_vocab.indices_to_sequence(b)]
            atcs += [self.atc_vocab.indices_to_sequence(c)]
            pros += [self.pro_vocab.indices_to_sequence(d)]
            labs += [self.lab_vocab.indices_to_sequence(e)]
            
        return outputs, diags,atcs, pros, labs
    def test(self, model_name):
        self.load_model(model_name)
        self.model.eval()
        with torch.no_grad():
            atcs, pros, labs, diags, outputs = [], [], [], [], []
            for iters, (atc, pro, lab, diag) in enumerate(self.test_loader):
                atc = (atc[0].cuda(), atc[1])
                pro = (pro[0].cuda(), pro[1])
                lab = (lab[0].cuda(), lab[1])
                diag = (diag[0].cuda(), diag[1])
                diag = diag[0].T.tolist()
                decoded_sentence = self.model.evaluate(atc, pro, lab)
                atc = atc[0].cpu().T.tolist()
                pro = pro[0].cpu().T.tolist()
                lab = lab[0].cpu().T.tolist()
                # print(decoded_sentence, diag, atc, pro, lab)
                self.eval_decode += decoded_sentence
                self.eval_ans += diag
                self.eval_atc += atc
                self.eval_pro += pro
                self.eval_lab += lab
                a, b, c, d, e = self.idx2code()
                outputs += a
                diags += b
                atcs += c
                pros += d
                labs += e

            self.eval_decode.clear()
            self.eval_ans.clear()
            self.eval_atc.clear()
            self.eval_pro.clear()
            self.eval_lab.clear()
            pd.DataFrame({
                'atc': atcs,
                'pro': pros,
                'lab': labs,
                'diag': diags,
                'output': outputs,
            }).to_excel(r'./test_result.xlsx', index=False)


class DataSet(Dataset):

    def __init__(self, data, atc_vocab, pro_vocab, lab_vocab, diag_vocab):
        self.data = data
        self.atc_vocab = atc_vocab
        self.pro_vocab = pro_vocab
        self.lab_vocab = lab_vocab
        self.diag_vocab = diag_vocab

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return (self.atc_vocab.sequence_to_indices(
            self.data.loc[idx, 'Code_atc'].split(',')),
                self.pro_vocab.sequence_to_indices(
                    self.data.loc[idx, 'Code_pro'].split(',')),
                self.lab_vocab.sequence_to_indices(
                    self.data.loc[idx, 'Code_lab'].split(',')),
                self.diag_vocab.sequence_to_indices(
                    self.data.loc[idx, 'Code_diag'].split(',')))


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    length_atc = []
    length_lab = []
    length_pro = []
    length_diag = []

    batch_atc = []
    batch_lab = []
    batch_pro = []
    batch_diag = []

    for data in batch:
        length_atc += [len(data[0])]
        batch_atc += [torch.tensor(data[0], dtype=torch.int32)]
        length_pro += [len(data[1])]
        batch_pro += [torch.tensor(data[1], dtype=torch.int32)]
        length_lab += [len(data[2])]
        batch_lab += [torch.tensor(data[2], dtype=torch.int32)]
        length_diag += [len(data[3])]
        batch_diag += [torch.tensor(data[3], dtype=torch.int64)]
    # get sequence lengths
    length_atc = torch.tensor(length_atc, dtype=torch.int32)
    length_pro = torch.tensor(length_pro, dtype=torch.int32)
    length_lab = torch.tensor(length_lab, dtype=torch.int32)
    length_diag = torch.tensor(length_diag, dtype=torch.int32)
    # padd
    batch_atc = torch.nn.utils.rnn.pad_sequence(batch_atc,
                                                batch_first=False,
                                                padding_value=1)
    batch_pro = torch.nn.utils.rnn.pad_sequence(batch_pro,
                                                batch_first=False,
                                                padding_value=1)
    batch_lab = torch.nn.utils.rnn.pad_sequence(batch_lab,
                                                batch_first=False,
                                                padding_value=1)
    batch_diag = torch.nn.utils.rnn.pad_sequence(batch_diag,
                                                 batch_first=False,
                                                 padding_value=1)
    return ((batch_atc, length_atc), (batch_pro, length_pro),
            (batch_lab, length_lab), (batch_diag, length_diag))


def main():

    # build DataTransformers
    atc_vocab = Vocabulary()
    atc_vocab.build_vocab(config.atc_path)
    pro_vocab = Vocabulary()
    pro_vocab.build_vocab(config.pro_path)
    lab_vocab = Vocabulary()
    lab_vocab.build_vocab(config.lab_path)
    diag_vocab = Vocabulary()
    diag_vocab.build_vocab(config.diag_path)

    # load raw data
    with open(config.data_path, 'rb') as f:
        data = pd.read_excel(f)
        # data = data.apply(lambda s: s.fillna({i: [] for i in data.index}))

    if config.isSplit:
        ids = data.ID.drop_duplicates()
        x = np.random.permutation(len(ids))

        train = x < 0.8 * len(ids)
        test = ~train

        train_id = ids.loc[train]
        test_id = ids.loc[test]

        train_data = data.merge(train_id, on="ID")
        test_data = data.merge(test_id, on="ID")

        train_dataset = DataSet(data=train_data,
                                atc_vocab=atc_vocab,
                                pro_vocab=pro_vocab,
                                lab_vocab=lab_vocab,
                                diag_vocab=diag_vocab)
        test_dataset = DataSet(data=test_data,
                               atc_vocab=atc_vocab,
                               pro_vocab=pro_vocab,
                               lab_vocab=lab_vocab,
                               diag_vocab=diag_vocab)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            num_workers=config.workers,
            collate_fn=collate_fn_padd,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config.batch_size,
                                                  num_workers=config.workers,
                                                  collate_fn=collate_fn_padd,
                                                  shuffle=False)

    # define our models
    vanilla_encoder_1 = VanillaEncoder(
        vocab_size=atc_vocab.vocab_size,
        embedding_size=config.encoder_embedding_size,
        output_size=config.encoder_output_size)
    vanilla_encoder_2 = VanillaEncoder(
        vocab_size=pro_vocab.vocab_size,
        embedding_size=config.encoder_embedding_size,
        output_size=config.encoder_output_size)
    vanilla_encoder_3 = VanillaEncoder(
        vocab_size=lab_vocab.vocab_size,
        embedding_size=config.encoder_embedding_size,
        output_size=config.encoder_output_size)

    vanilla_decoder = VanillaDecoder(
        hidden_size=config.decoder_hidden_size,
        output_size=diag_vocab.vocab_size,
        max_length=diag_vocab.max_length,
        teacher_forcing_ratio=config.teacher_forcing_ratio,
        sos_id=diag_vocab.SOS_ID,
        use_cuda=config.use_cuda)
    if config.use_cuda:
        vanilla_encoder_1 = vanilla_encoder_1.cuda()
        vanilla_encoder_2 = vanilla_encoder_2.cuda()
        vanilla_encoder_3 = vanilla_encoder_3.cuda()
        vanilla_decoder = vanilla_decoder.cuda()

    seq2seq = Seq2Seq(encoder_1=vanilla_encoder_1,
                      encoder_2=vanilla_encoder_2,
                      encoder_3=vanilla_encoder_3,
                      decoder=vanilla_decoder).cuda()

    trainer = Trainer(seq2seq, atc_vocab, pro_vocab, lab_vocab, diag_vocab, train_loader,
                      test_loader, config.learning_rate, config.use_cuda)
    trainer.train(num_epochs=config.num_epochs, pretrained=False)
    trainer.test('./model/epoch{}_auto_encoder.pt'.format(np.argmax(trainer.accs)+1))

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
