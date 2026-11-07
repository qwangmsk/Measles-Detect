Introduction
------------

Using a curated skin rash image dataset (DOI: 10.21227/9r41-4x79), we applied transfer learning to develop deep convolutional neural network (CNN) models for differentiating measles rashes from other skin conditions. Our ResNet-50 model achieved a classification accuracy of 95.2%, sensitivity of 81.7%, and specificity of 97.1%. We also evaluated MobileNet-V2, a lightweight CNN architecture, on the same dataset, with model training and evaluation conducted using Google Colab.

We subsequently investigated vision transformer (ViT) architectures to further improve measles identification. Using an expanded skin rash image dataset with increased sample size and diversity, we fine-tuned a pretrained Data-efficient Image Transformer (DeiT) model, which demonstrated enhanced classification performance compared with ResNet-50. Detailed methods and results are available in our preprint.

Citation
----------
1. Q. Wang (2025) Accurate Measles Rash Detection via Vision Transformer Fine-Tuning. arXiv:2005.09112 (https://arxiv.org/abs/2005.09112).

2. K. Glock, C. Napier, T. Gary, V. Gupta, J. Gigante, W. Schaffner, Q. Wang (2021) Measles Rash Identification Using Transfer Learning and Deep Convolutional Neural Networks. IEEE International Conference on Big Data, pp. 3905-3910 (https://ieeexplore.ieee.org/document/9671333).
