# Development and Evaluation of AI Models for Breast Ultrasound Report Generation

Jian Wang,
Jinfeng Xu,
Xin Yang,
Huaiyu Wu,
Xiliang Zhu,
Rusi Chen,
Ao Chang,
Yanlin Chen,
Jun Cheng,
Yongsong Zhou,
Rui Gao,
Keen Yang,
Guoqiu Li,
Jing Chen,
Hongtian Tian,
Ning Gu,
Fajin Dong,
Dong Ni

**Abstrct:** Ultrasound (US) imaging is an important medical imaging modality and is increasingly used in developing countries. US reports are the record carrier of examination results. Their writing is highly subjective, poorly reproducible, and time-consuming. To alleviate these problems, we developed an artificial intelligence (AI) system to automatically generate US reports from multi-modality and multi-view US images. We constructed a large-scale dataset containing 878,090 images and corresponding reports from 104,364 patients who underwent breast US (BUS) examination. The quantitative evaluation results show that the BI-RADS categories given by AI-generated reports have cancer risk prediction capabilities. The qualitative evaluation results show that the quality of AI-generated reports is close to that of a mid-level radiologist with seven years of experience, and the AI system helps junior radiologists with about three years of experience improve BI-RADS diagnostic accuracy. The developed AI system has the potential to conduct BUS examinations in developing countries and can be expanded to other US applications.
![image](https://github.com/TIanCat/Breast_US_Report_Generation/blob/main/figure/framework_1.png)
The overall design of this study. (a) Based on multi-modality and multi-view BUS images, the developed AI models generate findings and impressions in reports. (b) The reports generated by AI models were compared with a mid-level radiologist’s reports to evaluate the quality of the generated reports. (c) The reports of three junior radiologists with and without AI assistance were compared to evaluate the AI models’ assistance to junior radiologists.


## train
To train the finding model, please use the following command:
```
python train_cnn.py  -image_folder data/image   -train_path  data/split/train.json  -val_path data/split/val.json
```

To train the impression model, please use the following command:
```
python train_cat.py  -image_folder data/image   -train_path  data/split/train.json  -val_path data/split/val.json
```

## inference
To test the finding model, please use the following command:
```
python translate_cnn.py  -image_folder data/image   -test_path  data/split/test.json -model your_finding_model_path.pth
```

To test the impression model, please use the following command:
```
python translate_cat.py  -image_folder data/image   -test_path  data/split/test.json -model your_impression_model_path.pth
```

## demo
Online [Demo](http://www.ai4busrg.com/) of automatically generated BUS reports and expert evaluation.

## preprocessing
A Chinese sentence is divided into multiple words using the following function:
```
def chinese_split(chinese_str):
    seg_list = jieba.cut(chinese_str, cut_all=False)
    seg_str = " ".join(seg_list)
    return seg_str
```
To use the above function, you need to install jieba first：
```
pip install jieba
```
## Acknowledgement
We implemented our transformer model with reference to [jadore801120](https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master) and implemented the cider metric with reference to [brandontrabucco](https://github.com/brandontrabucco/cider). We thank them for their open source codes.
