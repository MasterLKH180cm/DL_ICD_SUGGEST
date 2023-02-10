import pdb
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, dataloader
from optimizer import optimizer, rangerlars, ranger_gc
from loss import loss
from config import config
from dataset.DataHelper import Vocabulary
from model.Decoder import VanillaDecoder
from model.Encoder import VanillaEncoder
from model.Seq2Seq import Seq2Seq
from sklearn.metrics import coverage_error
from sklearn.preprocessing import MultiLabelBinarizer


class Trainer(object):

    def __init__(self,
                 model,
                 atc_vocab,
                 pro_vocab,
                 lab_vocab,
                 diag_vocab,
                 mlb,
                 train_loader,
                 test_loader,
                 learning_rate,
                 pos_weight,
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
        
        
        self.mlb = mlb
        
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
        self.optimizer = ranger_gc.Ranger_GC(self.model.parameters(),
                                          lr=learning_rate)
        # self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5)
        # self.criterion = torch.nn.BCELoss(reduction='mean')
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

        self.checkpoint_name = checkpoint_name
        self.eval_decode = []
        self.eval_atc = []
        self.eval_pro = []
        self.eval_lab = []
        self.eval_ans = []
        self.coverage_error = []
        self.coverage_errors = []
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
            
            print(self.epoch+1)
            self.model.train()
            for iters, (atc, pro, lab, diag) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                atc = (atc[0].cuda(), atc[1])
                pro = (pro[0].cuda(), pro[1])
                lab = (lab[0].cuda(), lab[1])
                diag = diag[0].squeeze(0).cuda()
                decoder_outputs, decoder_hidden = self.model(atc, pro, lab)
                # calculate the loss and back prop.
                
                cur_loss = self.get_loss(decoder_outputs, diag)
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
            # self.lr_scheduler.step()
            self.epoch += 1

            self.save_model('epoch' + str(self.epoch))
            self.model.eval()
            with torch.no_grad():
                for iters, (atc, pro, lab,
                            diag) in enumerate(self.test_loader):
                    atc = (atc[0].cuda(), atc[1])
                    pro = (pro[0].cuda(), pro[1])
                    lab = (lab[0].cuda(), lab[1])
                    diag = diag[0].squeeze(0).cuda()
                    # diag = torch.tensor(self.mlb.transform(np.array(diag[0])), dtype=torch.float).cuda()
                    decoded_sentence = self.model.evaluate(atc, pro, lab)
                    # print(diag.cpu().tolist(),  decoded_sentence.shape)  

                    self.eval_decode += decoded_sentence.cpu().tolist()
                    self.eval_ans += diag.cpu().tolist()     
            self.eval_ans_idx = self.mlb.inverse_transform(np.array(self.eval_ans))
            
            self.eval_coverage_error()
            # self.eval_tpr()
            # self.eval_acc()
            self.tpr_acc()
            self.coverage_errors += [np.mean(self.coverage_error)]
            self.tprs += [np.mean(self.tpr)]
            self.accs += [np.mean(self.acc)]
            print('tprs: ', self.tprs)
            print('accs: ', self.accs)
            print('coverage_errors: ', self.coverage_errors)
            self.tpr.clear()
            self.acc.clear()
            self.coverage_error.clear()
            self.eval_decode.clear()
            self.eval_ans.clear()
            pd.DataFrame({
                'tpr': self.tprs,
                'acc': self.accs,
                'coverage_error': self.coverage_errors
            }).to_excel(r'./metric.xlsx', index=False)

    def get_loss(self, decoder_outputs, targets):
        return self.criterion(decoder_outputs, targets)

    def save_model(self, label):
        torch.save(self.model.state_dict(),
                   './model/' + label + '_' + self.checkpoint_name)
        print("Model has been saved as " + label + '_' + self.checkpoint_name)

    def load_model(self, name):
        self.model.load_state_dict(torch.load(name, map_location='cpu'))
        print("Pretrained model has been loaded.\n")

    
    def eval_coverage_error(self):
        self.coverage_error += [
            coverage_error(self.eval_ans, self.eval_decode)
        ]

    def tpr_acc(self):
        for a, b in zip(self.eval_decode, self.eval_ans_idx):
            
            v, i = torch.tensor(a).topk(20)
            a, b = set(self.diag_vocab.indices_to_sequence(
                i.cpu().tolist())), set(self.diag_vocab.indices_to_sequence(list(b)))
            a.discard('SOS')
            a.discard('EOS')
            a.discard('PAD')
            a.discard('UNK')
            b.discard('SOS')
            b.discard('EOS')
            b.discard('PAD')
            b.discard('UNK')
            
            self.tpr += [len(a.intersection(b)) / len(b)]
            self.acc += [len(a.intersection(b)) > 0]
    

    def idx2code(self):
        atcs, pros, labs, diags, outputs = [], [], [], [], []
        for a, b, c, d, e in zip(self.eval_decode, self.eval_ans,
                                 self.eval_atc, self.eval_pro, self.eval_lab):
            v, i = a.topk(20)

            outputs += [self.diag_vocab.indices_to_sequence(i.cpu().tolist())]
            diags += [self.diag_vocab.indices_to_sequence(b)]
            atcs += [self.atc_vocab.indices_to_sequence(c)]
            pros += [self.pro_vocab.indices_to_sequence(d)]
            labs += [self.lab_vocab.indices_to_sequence(e)]

        return outputs, diags, atcs, pros, labs

    def test(self, model_name):
        self.load_model(model_name)
        self.model.eval()
        
        with torch.no_grad():
            atcs, pros, labs, diags, outputs = [], [], [], [], []
            for iters, (atc, pro, lab, diag) in enumerate(self.test_loader):
                self.eval_decode.clear()
                self.eval_ans.clear()
                self.eval_atc.clear()
                self.eval_pro.clear()
                self.eval_lab.clear()
                atc = (atc[0].cuda(), atc[1])
                pro = (pro[0].cuda(), pro[1])
                lab = (lab[0].cuda(), lab[1])
                diag = diag[0].squeeze(0).cuda()
                # diag = torch.tensor(self.mlb.transform(np.array(diag[0])), dtype=torch.float).cuda()
                decoded_sentence = self.model.evaluate(atc, pro, lab)
                atc = atc[0].cpu().T.tolist()
                pro = pro[0].cpu().T.tolist()
                lab = lab[0].cpu().T.tolist()
                
                self.eval_decode += decoded_sentence
                self.eval_ans += self.mlb.inverse_transform(diag.cpu())
                self.eval_atc += atc
                self.eval_pro += pro
                self.eval_lab += lab
                a, b, c, d, e = self.idx2code()
                outputs += a
                diags += b
                atcs += c
                pros += d
                labs += e

            
            pd.DataFrame({
                'atc': atcs,
                'pro': pros,
                'lab': labs,
                'diag': diags,
                'output': outputs,
            }).to_excel(r'./test_result.xlsx', index=False)


class DataSet(Dataset):

    def __init__(self, data, atc_vocab, pro_vocab, lab_vocab, diag_vocab, mlb):
        self.data = data
        self.atc_vocab = atc_vocab
        self.pro_vocab = pro_vocab
        self.lab_vocab = lab_vocab
        self.diag_vocab = diag_vocab
        self.mlb = mlb

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        # print(self.mlb.classes_, self.diag_vocab.sequence_to_indices(
        #             self.data.loc[idx, 'Code_diag'].split(',')), self.mlb.transform([self.diag_vocab.sequence_to_indices(
        #             self.data.loc[idx, 'Code_diag'].split(','))]))
        return (self.atc_vocab.sequence_to_indices(
            self.data.loc[idx, 'Code_atc'].split(',')),
                self.pro_vocab.sequence_to_indices(
                    self.data.loc[idx, 'Code_pro'].split(',')),
                self.lab_vocab.sequence_to_indices(
                    self.data.loc[idx, 'Code_lab'].split(',')),
                self.mlb.transform([self.diag_vocab.sequence_to_indices(
                    self.data.loc[idx, 'Code_diag'].split(','))]))


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
        batch_diag += [torch.tensor(data[3], dtype=torch.float)]
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
    # batch_diag = torch.nn.utils.rnn.pad_sequence(batch_diag,
    #                                              batch_first=False,
    #                                              padding_value=1)
    # print(batch_diag)
    batch_diag = torch.cat(batch_diag, dim=0)
    # print(batch_diag)
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
    mlb = MultiLabelBinarizer()
    mlb.fit([diag_vocab.code2idx.values()])
    
    with open(r'./dataset/CC_count.xlsx', 'rb') as f:
        CC_count = pd.read_excel(f).groupby(by='Code').sum()
        CC_count['count'] = (CC_count['count'].sum()-CC_count['count'])/CC_count['count']
        CC_count['count'] = (CC_count['count'] - CC_count['count'].min())/(CC_count['count'].max()-CC_count['count'].min())*5+0.1
        # CC_count.loc[CC_count['count']>5, 'count'] = 5
    pos_weight = pd.DataFrame({'Code': diag_vocab.code2idx.keys()})
    # print(pos_weight, CC_count)
    pos_weight = pos_weight.merge(CC_count, on='Code', how='left').fillna(1)
    # print(pos_weight, CC_count)
    pos_weight = torch.Tensor(pos_weight['count']).cuda()
    # load raw data
    with open(config.data_path, 'rb') as f:
        data = pd.read_excel(f)
        data['Code_atc'] = data['Code_atc'].apply(lambda x: x.split(','))
        data = data.explode('Code_atc')
        data['Code_atc'] = data['Code_atc'].apply(lambda x: x[:5])
        data['Code_atc'] = data.groupby(by=['ID', 'Code_pro', 'Code_lab', 'Code_diag']).transform(lambda x: ','.join(x))
        data = data.drop_duplicates()
        # print(data)

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
                                diag_vocab=diag_vocab,
                                mlb=mlb)
        test_dataset = DataSet(data=test_data,
                               atc_vocab=atc_vocab,
                               pro_vocab=pro_vocab,
                               lab_vocab=lab_vocab,
                               diag_vocab=diag_vocab,
                               mlb=mlb)

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

    trainer = Trainer(seq2seq, atc_vocab, pro_vocab, lab_vocab, diag_vocab,
                      mlb, train_loader, test_loader, config.learning_rate,
                      pos_weight, config.use_cuda)
    trainer.train(num_epochs=config.num_epochs, pretrained=False)
    trainer.test('./model/epoch{}_auto_encoder.pt'.format(
        np.argmin(trainer.coverage_errors) + 1))


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
