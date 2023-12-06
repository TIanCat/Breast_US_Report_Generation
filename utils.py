import os
import shutil
import random
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge 
from cider import build_cider_scorer
from matplotlib import pyplot as plt


def plot(x_epoch,data1,data2,output_path, y_label, title):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x_epoch,data1, 'r', label='train')
	ax.plot(x_epoch,data2, 'b', label='valid')

	ax.set_title("{}".format(title))
	ax.set_xlabel("epoch")
	ax.set_ylabel(y_label)

	# 生成网格
	plt.grid()  
	plt.legend()
	plt.savefig(os.path.join(output_path, "{}.png".format(title)))
	plt.close()




def check_dir(path):
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)



def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)



def set_random_seed(seed, deterministic = True, benchmark = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def cal_performance(pred, gold, trg_pad_idx, weight = None, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, weight, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)

    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word



def cal_loss(pred, gold, trg_pad_idx, weight, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss



def batch_joint(batch, opt):
    start_g, start_l, end_g, end_l = 0, 0, 0, 0 
    src_img, img_seq_g, img_seq_l = [], [], []
    src_seq, trg_seq, img_idx_g, img_idx_l = [], [], [], []
    for i in range(len(batch)):
        sample = batch[i]
        img = sample["imgs"]
        seq_g = sample["img_seq_g"]
        seq_l = sample["img_seq_l"]
        src = sample["src_seq"]
        trg = sample["trg_seq"]

        src_img.append(img)
        img_seq_g.append(seq_g)
        img_seq_l.append(seq_l)
        src_seq.append(src)
        trg_seq.append(trg)

        end_g = end_g + img.size(0)
        end_l = end_l + img.size(0) * 49
        img_idx_g.append((start_g, end_g))
        img_idx_l.append((start_l, end_l))
        start_g = end_g
        start_l = end_l

    src_img = torch.cat(src_img, dim=0)
    img_seq_g = torch.cat(img_seq_g, dim=0)
    img_seq_l = torch.cat(img_seq_l, dim=0)
    src_seq = torch.cat(src_seq, dim=0)
    trg_seq = torch.cat(trg_seq, dim=0)

    src_img = src_img.cuda(non_blocking = True)
    img_seq_g = img_seq_g.cuda(non_blocking = True)
    img_seq_l = img_seq_l.cuda(non_blocking = True)
    src_seq = src_seq.cuda(non_blocking = True)
    trg_seq = trg_seq.cuda(non_blocking = True)

    return src_img, img_seq_g, img_seq_l, img_idx_g, img_idx_l, src_seq, trg_seq




def pad_seq(seq, max_len):
    if seq.size(1) < max_len:
        pad_len = max_len - seq.size(1)
        pad = torch.ones((1,pad_len),dtype=torch.int64)
        seq = torch.cat([seq, pad],dim=1)
    return seq



def patch_trg(trg, pad_idx):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold



def my_fn(batch):
    len_imgs_g = [sample["img_seq_g"].size(1) for sample in batch]
    len_imgs_l = [sample["img_seq_l"].size(1) for sample in batch]
    len_srcs = [sample["src_seq"].size(1) for sample in batch]
    len_trgs = [sample["trg_seq"].size(1) for sample in batch]
    max_len_img_g = max(len_imgs_g)
    max_len_img_l = max(len_imgs_l)
    max_len_src = max(len_srcs)
    max_len_trg = max(len_trgs)
    for sample in batch:
        sample["img_seq_g"] = pad_seq(sample["img_seq_g"], max_len_img_g)
        sample["img_seq_l"] = pad_seq(sample["img_seq_l"], max_len_img_l)
        sample["src_seq"] = pad_seq(sample["src_seq"], max_len_src)
        sample["trg_seq"] = pad_seq(sample["trg_seq"], max_len_trg)
    return batch



def compute_bleu_score(trg_seq, pred_seq):
    bleu1 = sentence_bleu([trg_seq], pred_seq, weights=(1, 0, 0, 0),
                            smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
    bleu2 = sentence_bleu([trg_seq], pred_seq, weights=(0.5, 0.5, 0, 0),
                            smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
    bleu3 = sentence_bleu([trg_seq], pred_seq, weights=(0.33, 0.33, 0.33, 0),
                            smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
    bleu4 = sentence_bleu([trg_seq], pred_seq, weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
    return bleu1, bleu2, bleu3, bleu4


def compute_cider_score(predict_seq, label_seq, results):
    id_to_captions = {}
    for idx, seq in enumerate(label_seq):
        id_to_captions[idx] = [seq]
    scorer = build_cider_scorer(id_to_captions, 4)

    for idx, seq in enumerate(predict_seq):
        cider = scorer(seq, idx)
        results[str(idx)]['cider'] = cider

    return results


def compute_avg_score(results, predict_str, label_str):
        n = len(results)
        temp = {}
        meteor, cider = 0, 0
        bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
        rouge3, rouge4, rougel = 0, 0, 0

        for k,v in results.items():
            bleu1 += v['bleu1']
            bleu2 += v['bleu2']
            bleu3 += v['bleu3']
            bleu4 += v['bleu4']
            rouge3 += v['rouge3']['r']
            rouge4 += v['rouge4']['r']
            rougel += v['rougel']['r']
            meteor += v['meteor']
            cider += v['cider']

        temp['pred_num'] = len(list(set(predict_str)))
        temp['glod_num'] = len(list(set(label_str)))

        temp['bleu1'] = bleu1 / n
        temp['bleu2'] = bleu2 / n
        temp['bleu3'] = bleu3 / n
        temp['bleu4'] = bleu4 / n
        temp['rouge3'] = rouge3 / n
        temp['rouge4'] = rouge4 / n
        temp['rougel'] = rougel / n
        temp['meteor'] = meteor / n
        temp['cider'] = cider / n
        results["avg_score"] = temp

        return results


