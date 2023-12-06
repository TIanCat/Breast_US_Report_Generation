'''
This script handles the training process.
'''

import os
import sys
import math
import time
import glob
import json
import torch
import logging
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import transformer.Constants as Constants
from transformer.Translator_cat import Translator
from transformer.Models_cat import CNNTransformer
from dataset import mydataset_cnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('-image_folder', default="data/image")   # image data
parser.add_argument('-train_path', default="data/split/train.json")  
parser.add_argument('-val_path', default="data/split/val.json")     
parser.add_argument('-test_path', default="data/split/test.json")     
parser.add_argument('-src_stoi_path', default="data/text_map/findings_stoi_min10.json")     # stoi data
parser.add_argument('-trg_stoi_path', default="data/text_map/impression_stoi_min10.json")     # stoi data 

parser.add_argument('-pretrained', default= None)  # pretrained weight

parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-start_epoch', type=int, default=0)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-wd', type=float, default=0.00001)
parser.add_argument('-src_max_len', type=int, default=450)
parser.add_argument('-trg_max_len', type=int, default=150)

parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_inner_hid', type=int, default=2048)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)
parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-en_n_layers', type=int, default=3)
parser.add_argument('-de_n_layers', type=int, default=3)

parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-gpu_id', type=int, default=1, help='gpu device id')

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-proj_share_weight', action='store_true')
parser.add_argument('-scale_emb_or_prj', type=str, default='prj')
parser.add_argument('-label_smoothing', action='store_true')


parser.add_argument('-save', type=str, default='EXP', help='experiment name')
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
opt = parser.parse_args()

opt.d_word_vec = opt.d_model
opt.proj_share_weight = True
opt.track_bn = True
opt.label_smoothing = True

opt.use_local = False



def main(opt):

    # exp directory: EXA + datetime
    if not os.path.exists(opt.save):
        opt.save = '{}-{}'.format(opt.save, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(opt.save)
    
    # save code and ckpt
    create_exp_dir(opt.save, scripts_to_save=glob.glob('*.py'))
    opt.checkpoint_dir = os.path.join(opt.save,'checkpoint')
    create_exp_dir(opt.checkpoint_dir, None)

    # log
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(opt.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


    # set gpu_id
    torch.cuda.set_device(opt.gpu_id)

    # set random seed
    if opt.seed is not None:
        set_random_seed(opt.seed, True, False)


    # train_loader, val_loader
    train_loader, val_loader = prepare_dataloaders(opt)
    
    # model
    model = prepare_model(opt).cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay= opt.wd)

    # resume
    model, optimizer = load_pretrainted_weight(model, optimizer, opt)

    # train
    train(model, train_loader, val_loader, optimizer, opt)


def prepare_dataloaders(opt):

    src_stoi_dict = json.load(open(opt.src_stoi_path, 'r',encoding='utf8'))
    trg_stoi_dict = json.load(open(opt.trg_stoi_path, 'r',encoding='utf8'))

    opt.src_vocab_size = len(src_stoi_dict)
    opt.trg_vocab_size = len(trg_stoi_dict)

    opt.src_bos_idx = src_stoi_dict[Constants.BOS_WORD]
    opt.trg_eos_idx = src_stoi_dict[Constants.EOS_WORD]
    opt.src_pad_idx = src_stoi_dict[Constants.PAD_WORD]

    opt.src_eos_idx = trg_stoi_dict[Constants.EOS_WORD]
    opt.trg_bos_idx = trg_stoi_dict[Constants.BOS_WORD]
    opt.trg_pad_idx = trg_stoi_dict[Constants.PAD_WORD]
    
    
    train = mydataset_cnn(opt.image_folder, opt.train_path, is_aug = True,
                    src_max_len = opt.src_max_len,trg_max_len = opt.trg_max_len)
    val = mydataset_cnn(opt.image_folder, opt.val_path, is_aug = False,
                    src_max_len = opt.src_max_len,trg_max_len = opt.trg_max_len)

    train_iterator = DataLoader(train,batch_size=opt.batch_size, shuffle=True,num_workers=8,
                                collate_fn= my_fn,pin_memory=True,prefetch_factor=8)
    val_iterator = DataLoader(val,batch_size=16, shuffle=True,num_workers=8,
                              collate_fn= my_fn,pin_memory=True,prefetch_factor=8)
    return train_iterator, val_iterator


def prepare_model(opt):
    model = CNNTransformer(
        n_src_vocab = opt.src_vocab_size,
        n_trg_vocab = opt.trg_vocab_size,
        trg_pad_idx=opt.trg_pad_idx,
        d_word_vec=opt.d_word_vec,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        en_n_layers=opt.en_n_layers,
        de_n_layers=opt.de_n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        src_n_position= opt.src_max_len,
        trg_n_position = opt.trg_max_len,
        track_bn = opt.track_bn,
        use_local = opt.use_local,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        scale_emb_or_prj=opt.scale_emb_or_prj)
    
    return model


def load_pretrainted_weight(model, optimizer, opt):
    if opt.pretrained is not None:
        ckpt = torch.load(opt.pretrained, map_location='cpu')
        model.load_state_dict(ckpt["model"],strict = True)
        opt.start_epoch = ckpt["epoch"] + 1
        if "optimizer" in ckpt.keys():
            optimizer.load_state_dict(ckpt["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        print("load pretrained weights")
    return model, optimizer


def train(model, training_data, validation_data, optimizer, opt):
    ''' Start training '''

    valid_bleu, epochs  = [], []
    train_losses, train_accus, train_ppls = [], [], []
    valid_losses, valid_accus, valid_ppls = [], [], []
    for epoch_i in range(opt.start_epoch, opt.epoch):

        epochs.append(epoch_i)

        # log
        logging.info('*'*100)
        lr = optimizer.param_groups[0]['lr']
        logging.info('epoch: %d lr: %e', epoch_i, lr)

        # train,  acc
        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        elapse = (time.time()-start)/60
        logging.info('train: loss:{:.4f} | accu:{:.4f} | ppl:{:.4f} | elapse:{:.3f} min'
                     .format(train_loss, train_accu, train_ppl, elapse))

        # val, acc
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        elapse = (time.time()-start)/60
        logging.info('valid: loss:{:.4f} | accu:{:.4f} | ppl:{:.4f} | elapse:{:.3f} min'
                     .format(valid_loss, valid_accu, valid_ppl, elapse))
        
        # val, bleu
        start = time.time()
        bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
        if len(valid_losses) == 0 or valid_loss < min(valid_losses):
            if valid_accu > 0.88:  
                bleu1, bleu2, bleu3, bleu4 = test_epoch(model, validation_data, opt)
        elapse = (time.time()-start)/60
        logging.info('valid: bleu1:{:.4f} | bleu2:{:.4f} | bleu3:{:.4f} | bleu4:{:.4f} '\
                            '| elapse:{:.3f} min'.format(bleu1, bleu2, bleu3, bleu4, elapse))

       # model_name
        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict(),
                      'optimizer':  optimizer.state_dict()}
        model_name = '{out}/model_{epoch}_{accu:3.2f}_{bleu4:2.4f}.ckpt'.format(out=opt.checkpoint_dir,
                                                    epoch=epoch_i,accu=100*valid_accu,bleu4 = bleu4)
        # save model
        if opt.save_mode == 'all':
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_latest = '{out}/latest.ckpt'.format(out=opt.checkpoint_dir)
            torch.save(checkpoint, model_latest)
            if len(valid_losses) == 0 or valid_loss < min(valid_losses):
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')

        # save metrics
        train_loss = train_loss if train_loss < 2. else 2.
        valid_loss = valid_loss if valid_loss < 2. else 2.
        train_losses.append(train_loss)
        train_accus.append(train_accu)
        train_ppls.append(train_ppl)
        valid_losses.append(valid_loss)
        valid_accus.append(valid_accu)
        valid_ppls.append(valid_ppl)
        valid_bleu.append(bleu4)

        # plot
        plot(epochs,train_losses,valid_losses,opt.save,"loss","loss")
        plot(epochs,train_accus,valid_accus,opt.save,"accu","accu")
        plot(epochs,train_ppls,valid_ppls,opt.save,"ppl","ppl")
        plot(epochs,valid_bleu,valid_bleu,opt.save,"bleu4","bleu4")



def train_epoch(model, training_data, optimizer, opt, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        src_img, img_seq_g, img_seq_l, img_idx_g, img_idx_l, \
                    src_seq, trg_seq = batch_joint(batch, opt)
        
        target_seq, gold = patch_trg(trg_seq, opt.trg_pad_idx)

        optimizer.zero_grad()
        # forward
        pred = model(src_img, img_idx_g, img_seq_g, img_idx_l, img_seq_l, src_seq, target_seq)
        # cal loss & acc
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, None, smoothing=smoothing) 
        # backward and update parameters
        loss.backward()
        optimizer.step()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            
            src_img, img_seq_g, img_seq_l, img_idx_g, img_idx_l, \
                            src_seq, trg_seq = batch_joint(batch, opt)
            
            target_seq, gold = patch_trg(trg_seq, opt.trg_pad_idx)

            # forward 
            pred = model(src_img, img_idx_g, img_seq_g, img_idx_l, img_seq_l, src_seq, target_seq)

            # cal loss & acc
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, None, smoothing=False) 

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def test_epoch(model, validation_data, opt):
    model.eval()
    translator = Translator(
        model = model,
        beam_size=5,
        use_local = opt.use_local,
        max_seq_len=opt.trg_max_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).cuda()

    desc = '  - (Validation) '
    idx = 0
    pred_seqs, target_seqs = [], []
    score1, score2, score3, score4 = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            idx += 1
            if idx > 100:
                break

            for i in range(len(batch)):
                sample = batch[i]
                img = sample["imgs"]    # n*3*h*w
                img_seq_g = sample["img_seq_g"]
                img_seq_l = sample["img_seq_l"]

                src = sample["src_seq"]
                trg = sample["trg_seq"]

                src_img = img.cuda(non_blocking = True)
                img_seq_g = img_seq_g.cuda(non_blocking = True)
                img_seq_l = img_seq_l.cuda(non_blocking = True)

                src_seq = src.squeeze(0)
                trg_seq = trg.squeeze(0)

                src_seq = np.array(src_seq).tolist()
                trg_seq = np.array(trg_seq).tolist()
                src = src.cuda(non_blocking = True)

                for idxx,item in enumerate(trg_seq):
                    if item == 3:
                        trg_seq = trg_seq[1:idxx]
                        break
                    
                target_seqs.append(trg_seq)
                pred_seq = translator.translate_sentence(src_img, img_seq_g, img_seq_l, src)
                pred_seqs.append(pred_seq[1:-1])

        for trg_seq, pred_seq in zip(target_seqs, pred_seqs):
            bleu1 = sentence_bleu([trg_seq], pred_seq, weights=(1, 0, 0, 0),
                                    smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            bleu2 = sentence_bleu([trg_seq], pred_seq, weights=(0.5, 0.5, 0, 0),
                                    smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            bleu3 = sentence_bleu([trg_seq], pred_seq, weights=(0.33, 0.33, 0.33, 0),
                                    smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            bleu4 = sentence_bleu([trg_seq], pred_seq, weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            score1.append(bleu1)
            score2.append(bleu2)
            score3.append(bleu3)
            score4.append(bleu4)
        score1 = np.mean(np.array(score1))
        score2 = np.mean(np.array(score2))
        score3 = np.mean(np.array(score3))
        score4 = np.mean(np.array(score4))

        return score1, score2, score3, score4



if __name__ == '__main__':
    main(opt)
