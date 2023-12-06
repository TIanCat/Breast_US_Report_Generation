''' Translate input text with trained model. '''

import json
import torch
import argparse
from utils import *
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
 
import transformer.Constants as Constants
from transformer.Models_cnn import CNNTransformer
from transformer.Translator_cnn import Translator
from dataset import mydataset_cnn
from torch.utils.data import DataLoader


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-image_folder', default="data/image")
    parser.add_argument('-model', default="", help='Path to model weight file')
    parser.add_argument('-test_path', default="data/split/test.json")     # test text
    parser.add_argument('-itos_path', default="data/text_map/findings_itos_min10.json")     # stoi data 
    parser.add_argument('-stoi_path', default="data/text_map/findings_stoi_min10.json")     # stoi data 
    parser.add_argument('-target',type=str, default= "findings")  # pretrained weight
    parser.add_argument('-output', default='results.txt')
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-gpu_id', type=int, default=1, help='gpu device id')


    opt = parser.parse_args()
    opt.track_bn = True

    torch.cuda.set_device(opt.gpu_id)
    set_random_seed(1, True, False)


    itos = json.load(open(opt.itos_path,'r',encoding='utf8'))
    stoi = json.load(open(opt.stoi_path,'r',encoding='utf8'))
    opt.src_pad_idx = stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = stoi[Constants.EOS_WORD]

    translator = Translator(
        model=load_model(opt),
        beam_size=opt.beam_size,
        use_local = opt.use_local,
        max_seq_len=opt.max_token_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).cuda()

    rouge = Rouge(metrics=["rouge-3","rouge-4","rouge-l"])
    test = mydataset_cnn(opt.image_folder, opt.test_path, is_aug = False,
                    src_max_len = opt.src_max_len,trg_max_len = opt.trg_max_len)

    test_loader = DataLoader(test,batch_size=4, shuffle=False,num_workers=4,
                              collate_fn= my_fn,pin_memory=True,prefetch_factor=8)

    count = 0
    results = {}
    predict_str, predict_seq, label_str, label_seq = [], [], [],[]
    metrics = {'bleu1':0, 'bleu2':0, 'bleu3':0, 'bleu4':0, 'rougel':0, 'meteor':0}
    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):

            for i in range(len(example)):
                sample = example[i]
                patient_id = sample["patient_id"]
                img = sample["imgs"]    # n*3*h*w
                seq_g = sample["img_seq_g"]
                seq_l = sample["img_seq_l"]
                if opt.target == "findings":
                    trg = sample["src_seq"]
                elif opt.target == "impression":
                    trg = sample["trg_seq"]
                else:
                    raise
                src_img = img.cuda()
                img_seq_g = seq_g.cuda()
                img_seq_l = seq_l.cuda()
                trg_seq = trg.squeeze(0)
                trg_seq = np.array(trg_seq).tolist()
                # remove bos/eos/pad
                for idx,item in enumerate(trg_seq):
                    if item == 3:
                        trg_seq = trg_seq[1:idx]
                        break
                # infer
                pred_seq = translator.translate_sentence(src_img, img_seq_g, img_seq_l)
                pred_seq = pred_seq[1:-1]  # remove bos/eos

                # seq -> seq_str
                pred_seq_str = [str(seq) for seq in pred_seq]
                gold_seq_str = [str(seq) for seq in trg_seq]

                # seq -> seq_sentence
                pred_seq_sentence = ' '.join(seq_str for seq_str in pred_seq_str)
                gold_seq_sentence = ' '.join(seq_str for seq_str in gold_seq_str)

                # bleu
                bleu1, bleu2, bleu3, bleu4 = compute_bleu_score(gold_seq_str, pred_seq_str)

                # meteor
                meteor = meteor_score([gold_seq_str], pred_seq_str)

                # rouge
                rouge_score = rouge.get_scores(pred_seq_sentence, gold_seq_sentence)

                # seq -> text
                pred_line = 'pred: '+ ''.join(itos[seq_str] for seq_str in pred_seq_str)
                gold_line = 'gold: '+ ''.join(itos[seq_str] for seq_str in gold_seq_str)

                predict_str.append(pred_line)
                predict_seq.append(pred_seq_sentence)
                label_str.append(gold_line)
                label_seq.append(gold_seq_sentence)

                metrics['bleu1'] += bleu1
                metrics['bleu2'] += bleu2
                metrics['bleu3'] += bleu3
                metrics['bleu4'] += bleu4
                metrics['rougel'] += rouge_score[0]["rouge-l"]["r"]
                metrics['meteor'] += meteor

                metric = "bleu4:{:.4},  rougel:{:.4},  meteor:{:.4}"\
                            .format(bleu4, rouge_score[0]["rouge-l"]["r"], meteor)
                f.write(pred_line.strip() + '\n')
                f.write(gold_line.strip() + '\n')
                f.write(metric.strip() + '\n\n')

                temp = {}
                temp["patient_id"] = patient_id
                temp["pred_line"] = pred_line
                temp["gold_line"] = gold_line
                temp["pred_seq"] = pred_seq_sentence
                temp["gold_seq"] = gold_seq_sentence
                temp["bleu1"] = bleu1
                temp["bleu2"] = bleu2
                temp["bleu3"] = bleu3
                temp["bleu4"] = bleu4
                temp["rouge3"] = rouge_score[0]["rouge-3"]
                temp["rouge4"] = rouge_score[0]["rouge-4"]
                temp["rougel"] = rouge_score[0]["rouge-l"]
                temp["meteor"] = meteor
                temp["cider"] = 0.
                results[str(count)] = temp
                count += 1

        metric = "bleu1:{:.4}, bleu2:{:.4}, bleu3:{:.4}, bleu4:{:.4}, rougel:{:.4}, meteor:{:.4}"\
                    .format(metrics['bleu1']/count, metrics['bleu2']/count, metrics['bleu3']/count,
                            metrics['bleu4']/count, metrics['rougel']/count, metrics['meteor']/count)
        f.write(metric.strip() + '\n')

        # cider
        results = compute_cider_score(predict_seq, label_seq, results)

        # avg
        results = compute_avg_score(results, predict_seq, label_seq)


    json.dump(results, open(opt.output.replace('txt', 'json'), 'w', encoding='utf8'), ensure_ascii=False, indent=2)
    print('[Info] Finished.')






def load_model(opt):

    checkpoint = torch.load(opt.model, map_location="cpu")
    model_opt = checkpoint['settings']

    opt.max_token_seq_len = model_opt.max_token_seq_len
    opt.src_max_len = model_opt.src_max_len
    opt.trg_max_len = model_opt.trg_max_len
    opt.use_local = model_opt.use_local

    model = CNNTransformer(
        n_trg_vocab= model_opt.trg_vocab_size,
        trg_pad_idx= model_opt.trg_pad_idx,
        d_word_vec= model_opt.d_word_vec,
        d_model= model_opt.d_model,
        d_inner= model_opt.d_inner_hid,
        de_n_layers= model_opt.de_n_layers,
        n_head= model_opt.n_head,
        d_k= model_opt.d_k,
        d_v= model_opt.d_v,
        dropout= model_opt.dropout,
        trg_n_position = model_opt.max_token_seq_len,
        track_bn = model_opt.track_bn,
        use_local = model_opt.use_local,
        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        scale_emb_or_prj=model_opt.scale_emb_or_prj).cuda()

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 





if __name__ == "__main__":

    main()
