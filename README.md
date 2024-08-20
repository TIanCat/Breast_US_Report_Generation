# Artificial Intelligence in Breast Ultrasound Diagnosis and Report Generation: A Retrospective, Multicenter Study

**Introduction:** During ultrasound (US) examinations, radiologists usually capture multi-modality US images, such as B-mode, color Doppler, and elastography images, as well as multi-view US images. Then they analyze these images and write US reports consisting of essential findings and impressions. For US reports to be effective, radiologists must ensure accuracy and consistency, particularly between the findings and impressions sections. However, report writing is highly subjective and less reproducible due to inter-observer and intra-observer variability. In addition, writing US reports during examinations is also time-consuming, reducing examination efficiency. A natural idea is to use artificial intelligence (AI) to automatically generate reports with diagnostic capabilities.

AI based on large language models (LLMs) has made progress. Recent studies have demonstrated potential applications of LLMs in medical report processing. For instance, transforming free-text radiology reports into structured formats, automatically generating the impression section from finding section, detecting errors in radiology reports, automatically creating synoptic reports from original reports, and simplifying radiology reports. In addition, diagnosis based on structured feature descriptions, patient history and (or) imaging findings, combination of imaging data, patient history and figure legends have also been explored in LLMs. Although LLMs hold promise for report generation, three challenges hinder their clinical application: 1) LLMs are not good at making diagnostic decisions based on medical images. 2) Interaction with LLMs requires appropriate prompts, otherwise undesired results will be obtained. 3) LLMs are difficult to deploy inside hospitals. On the other hand, specific model based on cross-modal techniques from image to text are gaining more and more attention in the deep learning community, which provides a solution for the automatic generation of medical reports. However, there are very few studies exploring such techniques on US images. In particular, these studies only collect a small number of B-mode US images from the single center to build models, and evaluated model performance only using quantitative metrics in the field of natural language processing (NLP). Therefore, their clinical effectiveness is still unclear.

Therefore, the aim of this study was to develop an AI system capable of generating US reports and evaluate its diagnostic capabilities and value to radiologists. Considering the high incidence and mortality of breast cancer and the important role of US in breast cancer screening, we chose to explore breast US (BUS).


![image](https://github.com/TIanCat/Breast_US_Report_Generation/blob/main/figure/framework.png)
The overall design of this study. (a) The developed system consists of the finding and impression models, which were used to generate findings and impressions of BUS reports from multi-modality and multi-view US images. (b) Similarity measurement between AI-generated reports and radiologist-written reports. (c) Malignancy risk assessment for each BI-RADS level in AI-generated reports. (d) Three senior radiologists evaluated reports generated by the AI system and reports written by one mid-level radiologist to determine the diagnostic level of AI-generated reports. (e) Three senior radiologists evaluated reports of junior radiologists with and without AI assistance to determine the value of the AI system to radiologists.


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
