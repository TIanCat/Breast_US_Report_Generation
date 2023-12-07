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

**Abstrct** Breast ultrasound (BUS) reports are important because subsequent treatment recommendations are based on them. The quality of BUS reports is easily affected by the radiologist's experience level; inexperienced radiologists may provide less accurate diagnoses than those from experienced radiologists. To alleviate this problem, we developed two artificial intelligence (AI) models that jointly and automatically generate BUS reports from BUS images. A total of 878090 multi-modality and multi-view ultrasound images and corresponding reports from 104364 patients who underwent BUS examination were collected; of these, data from 103664 patients were used for model development and quantitative evaluation, and that from 700 patients were used for qualitative expert evaluation. The expert evaluation results showed that the quality of the AI-generated reports was not significantly different from that of a mid-level radiologist with seven years of experience in BUS. The AI models helped three junior radiologists with approximately three years of BUS experience improve their diagnostic accuracy.
![image](https://github.com/TIanCat/Breast_US_Report_Generation/blob/main/figure/framework_1.png)


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

