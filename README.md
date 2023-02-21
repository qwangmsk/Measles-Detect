# Measles Detect

Introduction
------------
Measles is a highly contagious disease, one of the largest vaccine-preventable illnesses and leading causes of death in developing countries, claiming more than 140,000 lives each year. Measles was declared eliminated in the United States in the year 2000 due to decades of successful vaccination but it resurged in 2019 with 1,282 confirmed cases. Due to rapid spread of this disease among people in contact, rapid and automated diagnostic systems are required for early prevention. 

In this work, we employed transfer learning to build deep convolutional neural networks (CNNs) to distinguish measles rash from other skin conditions. Experiments with ResNet-50 model, trained on our diverse and curated skin rash image dataset, produce classification accuracy of 95.2%, sensitivity of 81.7%, and specificity of 97.1%, respectively. The performance of a small CNN model MobileNet-V2 on our image data set is also provided.

MobileNet-V2 was run and tested on Google Colab. Our complete image data set is provided in a publicly available repository, IEEE-DataPort (DOI: 10.21227/9r41-4x79, https://ieee-dataport.org/documents/image-dataset-various-skin-conditions-and-rashes).

Citation
----------
K. Glock, C. Napier, T. Gary, V. Gupta, J. Gigante, W. Schaffner, Q. Wang (2021) Measles Rash Identification Using Transfer Learning and Deep Convolutional Neural Networks. IEEE International Conference on Big Data, pp. 3905-3910 (https://ieeexplore.ieee.org/document/9671333).

K. Glock, C. Napier, A. Louie, T. Gary, J. Gigante, W. Schaffner, Q. Wang* (2020) Measles Rash Identification Using Residual Deep Convolutional Neural Network. arXiv:2005.09112 (https://arxiv.org/abs/2005.09112).
