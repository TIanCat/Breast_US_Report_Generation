from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import torch
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class mydataset_cnn(Dataset):
    def __init__(self, data_root, json_path, is_aug, 
                        src_max_len = 450, trg_max_len = 150):

        self.data_root = data_root
        self.is_aug = is_aug

        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

        with open(json_path, 'r', encoding="utf8") as f:
            self.json_dict = json.load(f)        
        self.data_pair = read_list(self.json_dict)

        self.train_trans = transforms.Compose([
            PaddingAndResize(size = (288,288)),
            transforms.RandomResizedCrop(224, scale=(0.3, 1),ratio=(9/10., 10/9.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)],p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(
                        kernel_size=3, sigma=(0.1, 2.0))],p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()])
        self.val_trans = transforms.Compose([
            PaddingAndResize(size = (256,256)),
            transforms.CenterCrop(size = (224,224)),
            transforms.ToTensor()])

    def __getitem__(self, item):

        data_item = self.data_pair[item]
        patient_id = data_item["patient_id"]
        img_list = data_item["img_list"]
        src_seq = data_item["findings"]
        trg_seq = data_item["impression"]

        # findings 
        if len(src_seq) > self.src_max_len-1:
            src_seq = src_seq[:self.src_max_len-1]
        src_seq.append(3)

        # impression
        if len(trg_seq) > self.trg_max_len-1:
            trg_seq = trg_seq[:self.trg_max_len-1]
        trg_seq.append(3)

        # images
        imgs, img_seq_g, img_seq_l = [], [], []
        for im in img_list:
            img_path = os.path.join(self.data_root,patient_id,im)
            image = Image.open(img_path)
            if self.is_aug:
                image = self.train_trans(image)
            else:
                image = self.val_trans(image)
            image = image.unsqueeze(0)
            imgs.append(image)
            img_seq_g.append(10)
            img_seq_l.extend([10]*49)


        img_seq_g = np.array(img_seq_g)  # g
        img_seq_l = np.array(img_seq_l)  # l
        src_seq = np.array(src_seq)     
        trg_seq = np.array(trg_seq)
        img_seq_g = torch.from_numpy(img_seq_g).unsqueeze(0) # 1xg
        img_seq_l = torch.from_numpy(img_seq_l).unsqueeze(0) # 1xl
        src_seq = torch.from_numpy(src_seq).unsqueeze(0)  
        trg_seq = torch.from_numpy(trg_seq).unsqueeze(0)
        imgs = torch.cat(imgs,dim=0)  # 1x3xhxw

        return {"imgs":imgs,"img_seq_g":img_seq_g, "img_seq_l":img_seq_l,
                "src_seq":src_seq,"trg_seq":trg_seq, "patient_id":patient_id}

    def __len__(self):
        return len(self.data_pair)


  
def read_list(json_dict):
    data_pair = []
    for k,v in json_dict.items():
        img_list = v["images"]
        findings = v["findings_num"]
        impression = v["impression_num"]
        temp = {}
        temp["patient_id"] = k
        temp["img_list"] = img_list
        temp["findings"] = [int(n) for n in findings.split()]
        temp["impression"] = [int(n) for n in impression.split()]
        data_pair.append(temp)
    return data_pair

  

class PaddingAndResize(object):

    def __init__(self, size = (320,320)):
        self.size = size

    def __call__(self, img):
        width, height = img.size
        if width > height:
            diff = width - height
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            img = transforms.functional.pad(img,padding = (0,pad_top,0,pad_bottom),fill= 114)
        else:
            diff = height - width
            pad_left = diff // 2
            pad_right = diff - pad_left
            img = transforms.functional.pad(img,padding = (pad_left,0,pad_right,0),fill= 114)
        img = transforms.functional.resize(img, self.size)
        return img



if __name__ == '__main__':

    pass