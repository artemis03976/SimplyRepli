# RepliForDeeper:一些领域中经典深度学习模型的简易实现

此代码仓库包含了一些领域中经典深度学习模型的简易实现，包括但不限于计算机视觉、自然语言处理等。这些实现均使用PyTorch框架，并且希望做到易于理解与修改。

## 前言

你是否有时会觉得阅读一个深度学习的官方实现代码时，很难理解其中的思路？

至少对于当时那个刚踏入深度学习大门的我来说，就跟看天书一样。
我常常坐在电脑前，一杯茶一包烟一段代码看一天()，然后发现，即便用debugging来观察代码走向，我好像也什么都没明白。

后来我才明白，很多官方实现总是为了所谓的“工程化”而大大牺牲了代码的可读性，使得这些模型代码对大部分初学者来说变得难以理解。

因此，本人创建了这个代码仓库，尝试以最简单直观的代码来复现深度学习中许多经典的模型，既是记录自己的学习过程，也希望能帮助那些有同样困惑的初学者。

## 项目结构

项目根目录包含以下内容：

- **/datas**: 用于训练，验证，以及测试的数据集保存路径
- **/global_utilis**: 一些全局可用的功能性代码，如部分数据集处理，模型评估，模型存储等
- **/Models**: 主要的模型实现代码，以领域为单位进行分类

对于每个模型，其目录下包含以下内容：
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
- **/checkpoints**: 通过训练得到的模型权重检查点，在仓库中不可见，训练时会自动创建
- **/outputs**: 对于部分模型保存推理时的输出路径，在仓库中不可见，推理时会自动创建
- **/modules**: 放置一些模型的子组件代码，如果模型结构足够简单则不会存在
- **/config**: 模型的配置文件, 通常包含`config.py`和`config.yaml`. 前者以一个类保存从后者中读取的配置信息
- **/train.py**: 训练脚本
- **/model.py**: 主要的模型定义代码。如果模型有一些变体，则会以一个目录来保存
- **/inference.py**: 推理脚本
- 
## 现有模型
- 自编码器
  - Vanilla Auto Encoder
  - Conditional Auto Encoder
  - Variational Auto Encoder
  - Vector Quantization Auto Encoder
- 扩散模型
  - Denoising Diffusion Probabilistic Models(DDPM)
  - Denoising Diffusion Implicit Models(DDIM)
- 生成对抗网络
  - 原始Generative Adversarial Networks(GAN)
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
- 图像分类
  - AlexNet
  - VGG
  - ResNet
  - GoogLeNet
  - InceptionV3
  - DenseNet
  - ConvNeXt
  - Vision Transformer(ViT)
  - SqueezeNet
- 多模态(建设中)
- 目标检测(建设中)
- 循环神经网络
  - 原始RNN
  - LSTM
  - GRU
- Seq2Seq
- Transformer
- 图像分割
  - UNet
  - TransUNet
- 图像超分
  - SRCNN
  - ESPCN
  - VDSR
  - DRCN
  - DRRN
  - EDSR
  - SRGAN

## 待办事项
- [ ] 为每个模型都添加一个README文件, 包括一些必要的信息与用法
- [ ] 将每个模型的损失函数都单独提取出来形成`loss.py`，提供更好的代码可读性
- [ ] (可能)对每个领域任务设计集成的Trainer和Inference来提供统一的模板（基于领域内任务的相似性）

## 许可协议
本项目使用MIT许可 - 详情请见[LICENSE](LICENSE)。