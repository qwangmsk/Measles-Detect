# LLMs for Measles Identification Using Skin Rash Images

This repository contains the evaluation framework, prompts, and results for assessing multimodal large language models (LLMs) on measles identification from skin rash photographs. We compare individual LLMs on binary measles classification and top-three differential diagnosis, and investigate simple decision-level fusion strategies that combine complementary model predictions.

## 1. Assessments of LLMs for measles identification

We evaluated two multimodal LLMs, **GPT-6.1-terra** and **Gemini-3.8-flash**, using an image-only assessment protocol. Neither model was fine-tuned on the evaluation images. The analysis used **2,068 skin rash images**, including **211 measles** and **1,857 non-measles** images.

* **Binary identification:** Each model assessed whether an image was consistent with measles or a non-measles condition.
* **Top-three differential diagnosis:** Each model provided three ranked candidate diagnoses. An image was considered positive for measles when *measles* or *rubeola* appeared among the three diagnoses; *rubella* was not counted as measles.

The evaluations emphasize sensitivity, specificity, precision, and F1 score because measles images represent a minority of the benchmark.

## 2. Decision-level fusion of LLM predictions

We investigated complementary predictions from the two models and two assessment tasks using deterministic decision rules. Fusion was performed on saved model outputs and did not require model fine-tuning.

* **Binary OR:** Positive if either model's binary assessment predicts measles. 
* **Selective hybrid:** Positive if Gemini's binary prediction is positive, or if Gemini's top-three result contains measles **and** GPT's binary prediction is positive.
* **Simplified three-way OR:** Positive if **GPT binary OR Gemini binary OR Gemini top-three** identifies measles. 

The simplified three-way rule matched the four-way OR result in this benchmark: adding GPT top-three predictions did not change the fused classifications. These findings illustrate the potential of combining complementary LLM outputs while maintaining an interpretable decision rule.

## 3. Data source

The benchmark builds on a curated public image dataset of skin conditions and rashes:

* **IEEE DataPort:** *Image Dataset of Various Skin Conditions and Rashes*, DOI: [10.21227/9r41-4x79](https://doi.org/10.21227/9r41-4x79).

The evaluation set contains 211 measles and 1,857 non-measles images. A local image directory may be organized as:

```text
rash_dataset/
├── measles/
└── non_measles/
```

## 4. System setup

To reproduce API-based assessments, obtain access to the corresponding model providers and configure valid API credentials. You can follow the procedure below to set up your running environment:

1. Configure an OpenAI API account and set the `OPENAI_API_KEY` environment variable for GPT evaluations.
2. Configure access and credentials for the Gemini API for Gemini evaluations.
3. Install the Python dependencies required by the evaluation scripts in this repository.
4. Prepare the benchmark images and labels, preserving the measles/non-measles split and a stable image identifier for matching predictions across models.
5. Run the model assessments, then apply the fusion rules to the saved per-image predictions.

## 5. Earlier deep learning assessments

Our earlier work used transfer learning to evaluate **ResNet-50** and **MobileNet-V2** for distinguishing measles rashes from other skin conditions. ResNet-50 achieved **95.2% accuracy**, **81.7% sensitivity**, and **97.1% specificity** on the earlier curated dataset. We subsequently fine-tuned a pretrained **Data-efficient Image Transformer (DeiT)** on an expanded and more diverse image collection and observed improved performance over ResNet-50. These studies provide background for the present LLM-focused work; their results are **not directly comparable** with the current LLM benchmark unless the evaluation sets and protocols are matched. These models and source code are available in [cnn-models](./cnn-models).

## 6. Intended use and limitations

This repository is intended for **research and methodological evaluation only**, not clinical diagnosis or patient management. Rash images alone cannot establish measles infection. Clinical assessment, epidemiological context, and appropriate laboratory testing are necessary for suspected measles cases. The benchmark is imbalanced, and the observed fusion performance should be validated on independent datasets before any clinical application.

## 7. Manuscript and citation

A manuscript describing the LLM comparison and decision-level fusion is being prepared for the 11th International Workshop on Health Intelligence (**W3PHIAI-27**). We will add the finalized preprint or publication citation here when available. 

If you use this repository, please cite the associated manuscript once published and acknowledge the original dataset providers.

