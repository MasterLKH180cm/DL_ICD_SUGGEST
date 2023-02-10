from model.Encoder import VanillaEncoder
from model.Decoder import VanillaDecoder
from model.Seq2Seq import Seq2Seq
from dataset.DataHelper import Vocabulary
from train import Trainer, DataSet, collate_fn_padd
from config import config
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
import pdb
import random
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch

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
    trainer.test()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)