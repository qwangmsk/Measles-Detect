# Measles Detect

Introduction
------------
Measles is a highly contagious disease, one of the largest vaccine-preventable illnesses and leading causes of death in developing countries, claiming more than 140,000 lives each year. Measles was declared eliminated in the United States in the year 2000 due to decades of successful vaccination but it resurged in 2025 with 1,356 confirmed cases as of August 5, 2025. Due to rapid spread of this disease among people in contact, rapid and automated diagnostic systems are required for early prevention and containment. 

This study aims to develop deep convolutional neural network (CNN) and vision transformer (ViT) models for accurate measles identification, with the long-term goal of creating an open-source, smartphone-based application to assist both physicians and patients in timely diagnosis. Such a tool could also be integrated into telemedicine platforms and deployed in high-traffic settings, such as airports, to help curb travel-related transmission.

This project comprised two stages. First, we conducted a pilot study employing transfer learning to develop deep CNN models for distinguishing measles rashes from other skin conditions. Using a curated skin rash image dataset, our ResNet-50 model achieved a classification accuracy of 95.2%, sensitivity of 81.7%, and specificity of 97.1%. We also evaluated a lightweight CNN model, MobileNet-V2, on the same dataset; this model was trained and tested using Google Colab. The complete image dataset we constructed for assessing these two models is publicly available through IEEE DataPort (DOI: 10.21227/9r41-4x79, https://ieee-dataport.org/documents/image-dataset-various-skin-conditions-and-rashes).

In the second stage of the project, we enhaneced the sample size and diversity by incorporting 755 skin images from a publicly available Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0) on Kaggle (https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data). After combining the two datasets, our final collection comprised 2,070 skin images, including 212 depicting measles rashes. On this dataset, we fine-tuned a pretrained Data-efficient Image Transformer (DeiT) model, achieving a median classification accuracy of 96.38%, precision of 96.24%, recall of 96.38%, and an F1-score of 96.23%, demonstrating high effectiveness in accurate detection to aid outbreak control. 

Citation
----------
Q. Wang (2025) Accurate Measles Rash Detection via Vision Transformer Fine-Tuning. arXiv:2005.09112 (https://arxiv.org/abs/2005.09112).

K. Glock, C. Napier, T. Gary, V. Gupta, J. Gigante, W. Schaffner, Q. Wang (2021) Measles Rash Identification Using Transfer Learning and Deep Convolutional Neural Networks. IEEE International Conference on Big Data, pp. 3905-3910 (https://ieeexplore.ieee.org/document/9671333).
