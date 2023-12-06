''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
import torchvision.models as models
import torch.nn.functional as F


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=300, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,



class  Resnet(nn.Module):
    def __init__(self, d_model =512, track_bn = False,  pretrained = True):
        super(Resnet, self).__init__()

        self.track_bn = track_bn
        self.pretrained = pretrained
        self.position_enc = PositionalEncoding(d_model, n_position=50)
        resnet18 = models.resnet18(pretrained = pretrained)


        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        # if not self.track_bn:
            #self.modify_bn()
    

    # def modify_bn(self):
    #     for k, v in self.named_modules():
    #         if isinstance(v, nn.BatchNorm2d):
    #             v.track_running_stats = False
    #             v.momentum = 0.9
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # b*512*h*w

        # avgpool -> global_feature
        x_g = self.avgpool(x)        # b*512*1*1
        x_g = torch.flatten(x_g, 1)  # b*512

        # local_feature -> positional embedding
        x_l = x.permute(0,2,3,1)                      # b*h*w*512
        x_l = x_l.view(x_l.size(0), -1, x_l.size(-1)) # b*hw*512
        x_l = self.position_enc(x_l)                  # b*(hw)*512
        x_l = x_l.reshape(-1, x_l.size(-1))           # bhw*512

        return x_g, x_l



class CNNTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, trg_pad_idx, d_word_vec=512, d_model=512, d_inner=2048, 
            en_n_layers=6, de_n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, 
            src_n_position = 350, trg_n_position = 300, track_bn = True, use_local = False, 
            trg_emb_prj_weight_sharing=True, scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx = self.trg_pad_idx = trg_pad_idx
        self.use_local = use_local
        self.d_model = d_model


        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        
        # padding
        self.visual_pad_token = nn.Parameter(torch.randn(1, d_model))

        # Encoder1
        self.Visual_Encoder = Resnet(d_model= d_model,
                                    track_bn = track_bn, 
                                    pretrained = True)

        # Encoder2
        self.Textual_Encoder = Encoder(
            n_src_vocab=n_src_vocab, d_word_vec=d_word_vec,
            n_layers=en_n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx,
            dropout=dropout, n_position=src_n_position,scale_emb=scale_emb)


        # Decoder
        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=trg_n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=de_n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)


        for n, p in self.named_parameters():
            if p.dim()>1 and 'Visual_Encoder.' not in n:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight


    def visual_padding(self,img_feat, max_len):
        pad_len = max_len - img_feat.size(0)
        pad = self.visual_pad_token.repeat(pad_len,1)
        feat = torch.cat([img_feat, pad],dim=0)
        return  feat.unsqueeze(0)


    def batch_regroup(self, img_feats, img_idx, max_len):
        feats = []
        for be in img_idx:
            img_feat = img_feats[be[0]:be[1], ]   # n * 512
            img_feat = self.visual_padding(img_feat, max_len)
            feats.append(img_feat)
        feats = torch.cat(feats, dim=0)
        return feats


    def forward(self, src_img, img_idx_g, img_seq_g, img_idx_l, img_seq_l, src_seq, trg_seq):

        # visual token mask
        img_mask_g = get_pad_mask(img_seq_g, self.src_pad_idx)
        img_mask_l = get_pad_mask(img_seq_l, self.src_pad_idx)  
        
        # target mask
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # image features 
        img_feat_g, img_feat_l  = self.Visual_Encoder(src_img) 
        
        # text features
        txt_feat, *_ = self.Textual_Encoder(src_seq, src_mask) 

        # batch regroup
        img_feat_g = self.batch_regroup(img_feat_g, img_idx_g, img_seq_g.size(1))
        img_feat_l = self.batch_regroup(img_feat_l, img_idx_l, img_seq_l.size(1))
        

        # feature concat
        if self.use_local:
            cat_feat = torch.cat([img_feat_l, txt_feat], dim=1)
            cat_mask = torch.cat([img_mask_l, src_mask], dim=2)
        else:
            cat_feat = torch.cat([img_feat_g, txt_feat], dim=1)
            cat_mask = torch.cat([img_mask_g, src_mask], dim=2)

        # decode
        dec_output, *_ = self.decoder(trg_seq, trg_mask, cat_feat, cat_mask)


        # final linear layer
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
