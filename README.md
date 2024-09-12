# RepliForDeeper: Some Minimal Implementation of Classical Deep Learning Models in Various Domains

[\[📖中文版\]](./README_zh.md)

This repository contains some minimal implementations of classical deep learning models in various domains, including computer vision, natural language processing, and so on. The implementations are based on PyTorch and are designed to be easy to understand and modify.

## Introduction

Have you ever found yourself struggling to understand the implementation of a deep learning model in the official codes? 

As for me, I'm just feeling them talking nonsense when I first step into the world of deep learning.
I often find myself looking at the code of a model and wondering how it works, 
but I can't seem to understand it even through debugging, 
and soon after I've realized that those official implementations are just too 'engineered' for most learners.

So, I decided to create this repository and try to replicate some classical deep learning models in minimal implementations, 
also serving as a record of my learning process to help others who may be in the same situation.


## Project Structure

The root directory repository contains:

- **/datas**: Data files for training, validation and testing.
- **/global_utilis**: Global utilities for some data preprocessing, model evaluation, model saving, etc.
- **/Models**: main directory for all models, classified by domain.

For each model, a typical corresponding directory contains the following files:
```
<model name>/
├── checkpoints/
├── outputs/
├── modules/
├── config/
├── train.py
├── model.py (or models/)
└── inference.py
```

- **/checkpoints**: Checkpoints saved from training. Ignored in th repo. Automatically created by the training script.
- **/outputs**: Output files from inference for parts of models. Ignored in th repo. Automatically created by the inference script.
- **/modules**: Some modules used by the model. Doesn't exist if the model is simple enough.
- **/config**: Configuration files for the model, usually contains `config.py` and `config.yaml`. The former is used for reading config in the latter with a class
- **/train.py**: Training script.
- **/model.py**: Model definition. If the model has some variants, it may become subdirectories.
- **/inference.py**: Inference script.

## Currently Supported Models
- Auto Encoder
  - Vanilla Auto Encoder
  - Conditional Auto Encoder
  - Variational Auto Encoder
  - Vector Quantization Auto Encoder
- Diffusion
  - Denoising Diffusion Probabilistic Models(DDPM)
  - Denoising Diffusion Implicit Models(DDIM)
- GAN
  - Vanilla Generative Adversarial Networks(GAN)
  - Conditional GAN(CGAN)
  - Deep Convolutional GAN(DCGAN)
  - Wasserstein GAN(WGAN)
  - Wasserstein GAN with Gradient Penalty(WGAN-GP)
  - WGAN-div
  - Auxiliary Classifier GAN(ACGAN)
  - Boundary Equilibrium GAN(BEGAN)
  - CycleGAN
  - Pix2Pix
  - Energy-Based GAN(EBGAN)
  - InfoGAN
  - Least Squares GAN(LSGAN)
  - Self Attention GAN(SAGAN)
  - Spectral Norm GAN(SNGAN)
  - StyleGAN
- Image Classification
  - AlexNet
  - VGG
  - ResNet
  - GoogLeNet
  - InceptionV3
  - DenseNet
  - ConvNeXt
  - Vision Transformer(ViT)
  - SqueezeNet
- Multi Modal(In Progress)
- Object Detection(In Progress)
- RNN
  - Vanilla RNN
  - LSTM
  - GRU
- Seq2Seq
- Transformer
- Segmentation
  - UNet
  - TransUNet
- Super Resolution
  - SRCNN
  - ESPCN
  - VDSR
  - DRCN
  - DRRN
  - EDSR
  - SRGAN

## TODO
- [ ] Add README for every model, including some necessary information and usage
- [ ] Extract the Loss Functions of every model into single file `loss.py` for better readability
- [ ] (Maybe) Integrated Trainer and Inference class for every domain to provide uniform template (based on the similarity of the tasks in one domain)

## License
This repo is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.