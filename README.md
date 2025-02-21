# Shoplifting Detection Using Xception

This repository contains a deep learning model based on the Xception architecture to detect shoplifting from surveillance footage.

## Repository Structure

```
├── dataset/                    # Directory containing training and validation data
├── Dataloader.py               # Script to load the dataset
├── loadPreTrain.py             # Script to load the pretrained model
├── Prediction.py               # Script to predict given input frame
├── trainEval.py                # Train and eval model performance

```
## Model Overview

The Xception model is a deep convolutional neural network optimized for image classification tasks. In this project, we fine-tune the Xception model to classify shoplifting behavior from video frames.


## Dataset
The dataset consists of labeled frames extracted from surveillance footage, categorized as **shoplifting** and **non-shoplifting**. Ensure the dataset is correctly structured before training.

