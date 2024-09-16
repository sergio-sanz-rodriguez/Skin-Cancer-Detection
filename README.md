![Project Image](images/intro_picture.png)
# Skin Cancer Detection Project

## Authors

Sergio Sanz  
Mila Miletic  

## Overview

This project belongs to a Kaggle competition on [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge/overview), in which the submission deadline was on Friday, September 6 2024.

This project focuses on the development of machine learning (ML) and deep learning (DL) algorithms to identify histologically confirm skin cancer cases with single-lesion crops from body images. In addition to the image database, a comprehensive set of metadata information related to the lesion is provided. For more information about the metadata features, please visit the aforementioned link. By using all this information, the goal of this challenge is to build, test, and evaluate a binary classification system that combines image-based deep learning neural networks with advanced machine learning models to identify whether the sample correspond to a benign or a malignant case. 

## Description (from Kaggle Website)
Skin cancer can be deadly if not caught early, but many populations lack specialized dermatologic care. Over the past several years, dermoscopy-based AI algorithms have been shown to benefit clinicians in diagnosing melanoma, basal cell, and squamous cell carcinoma. However, determining which individuals should see a clinician in the first place has great potential impact. Triaging applications have a significant potential to benefit underserved populations and improve early skin cancer detection, the key factor in long-term patient outcomes.

Dermatoscope images reveal morphologic features not visible to the naked eye, but these images are typically only captured in dermatology clinics. Algorithms that benefit people in primary care or non-clinical settings must be adept to evaluating lower quality images. This competition leverages 3D TBP to present a novel dataset of every single lesion from thousands of patients across three continents with images resembling cell phone photos.

This competition aims to develop AI algorithms that differentiate histologically-confirmed malignant skin lesions from benign lesions on a patient. This work will help to improve early diagnosis and disease prognosis by extending the benefits of automated skin cancer detection to a broader population and settings.

## Evaluation Metric

The metric that has been used in the project is the [partial area under the ROC curve](https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve) (pAUC) above 80% true positive rate (TPR) for binary classification of malignant examples. (See the implementation in the notebook [ISIC pAUC-aboveTPR](https://www.kaggle.com/code/metric/isic-pauc-abovetpr).)

## Challenges

Apart from the magnitude of its complexity, the main challenge of this project is working with an extremely imbalanced dataset. Specifically, the database includes about 400 malignant cases versus 400,000 benign cases, 1,000 more benign cases than malignant lesions.

Another key aspect to highlight is that there is no feature in the metadata that is highly correlated with the target. An in-depth analysis of the metadata, as well as a lot of feature engineering effort, is required to obtain very good partial AUC scores. Therefore, the machine learning models proposed in this project are fed with a large number of features, most of which have been re-generated from the original ones.

## Proposed Machine Learning Architecture (Submited to Kaggle)

As shown in Figure 1, the proposed machine learning model consists of a **Convolutional Neural Network (CNN)** and three boosting classifiers: **XGBoost**, **LightGBM**, and **Gradient Boosting Machine (GBM)**. The CNN architecture is composed of a [ResNet152V2](https://keras.io/api/applications/resnet/#resnet152v2-function) backbone for feature extraction, a global average pooling layer as an input layer to the neural network, a hidden layer with 64 neurons, and an output layer with a sigmoid activation function for binary classification. It is worth mentioning that the backone layers have also been trained (i.e., no transfer learning was used) to maximize prediction accuracy and recall. 

The training process of the CNN has been carefully designed to ensure class balancing and, more importantly, to avoid data leakage to the next stages of the proposed pipeline. The CNN output feeds the thee bossting machines along with the metadata features provided by Kaggle. In addition to the original metadata, new usesul feature were calculated using some of the published notebooks as references (e.g. click [here](https://www.kaggle.com/code/vyacheslavbolotin/isic-2024-only-tabular-data-new-features)).

<div align="center">
  <img src="images/model_pipeline.png" alt="Model Pipeline" width="500"/>
  <p><strong>Figure 1:</strong> Proposed machine learning architecture</p>
</div>

The three boosting classifiers rely on the same pre-processing pipeline (see Figure 2). The numerical features are scaled using the robust scaler method and the categorical features are one-hot encoded. The CNN predictions do not have to be scaled, since they already represent cancer probabilities between 0 (benign) and 1 (malignant). Due to the huge amount of metadata samples, the features belonging to the majority class (benign cases) are randomly downscaled up to 40,000 samples, and those belonging to the minory class are upscalded (with SMOTE) up to 4,000 samples.

<div align="center">
  <img src="images/boosting_classifiers.png" alt="Boosting Classifiers" width="1000"/>
  <p><strong>Figure 2:</strong> Pipeline of the boosting classifiers: XGBoost, LightGBM, Gradient Boosting</p>
</div>

Finally, the outputs of the boosting classifiers are evaluated using the Soft Voting ensemble approach. Although different weitghed averages were tested, the one that produced the best balance between publick pAUC scores, representing around 20% of the total test samples, and private pAUC scores, representing the remaining 70%, was the arithmetic averate, that is, weights [1, 1, 1]. 

The Partial AUC (pAUC) scores achieved in the competition are as follows:

- **Public Score**: 0.1649
- **Private Score**: 0.1516

## Description of the Notebooks

Fixme.

