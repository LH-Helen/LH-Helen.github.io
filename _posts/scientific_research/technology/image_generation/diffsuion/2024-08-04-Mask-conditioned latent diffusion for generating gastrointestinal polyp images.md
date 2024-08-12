---
title:  "「论文阅读」 Mask-conditioned latent diffusion for generating gastrointestinal polyp images+医学图像生成+潜在扩散+mask引导"
mathjax: true
key: diffusion_essay
toc: true
toc_sticky: true
category: [scientific_research, image_generation, diffusion]
tags: [eassy, diffusion, generation]


---

<span id='head'></span>

### Mask-conditioned latent diffusion for generating gastrointestinal polyp images

图像生成+潜在扩散+mask生成+mask引导

<center class="half"><img src="./../../../../../assets/img/scientific_research/扩散模型.assets/image-20240804113448242.png" alt="image-20240804113448242" style="zoom:80%;" /></center>

- [论文](https://arxiv.org/abs/2304.05233)
- [代码](https://github.com/simulamet-host/conditional-polyp-diffusion)

#### 相关信息

- 时间：11th Apr., 2023

- **期刊：**

- **关键词：** diffusion model, polyp generative model, polyp segmentation, gen

  erating synthetic data

#### 文章主要思想

<center class="half"><img src="./../../../../../assets/img/scientific_research/扩散模型.assets/image-20240804113513275.png" alt="image-20240804113513275" style="zoom:80%;" /></center>

训练：

- mask生成扩散
- 图像生成
  - mask引导+潜在扩散

预测

- 生成mask引导随机潜在噪声生成图像

训练细节

- A server with Nvidia A100 80𝐺𝐵 graphic processing units (GPUs)

  AMD EPYC 7763 64-cores processor with 2𝑇𝐵 RAM