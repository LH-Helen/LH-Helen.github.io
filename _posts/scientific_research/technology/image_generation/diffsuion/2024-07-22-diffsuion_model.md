---
title:  "「Diffusion Model」 扩散模型大合集"
mathjax: true
key: diffusion model
toc: true
toc_sticky: true
category: [scientific_research, image_generation, diffusion]
---
>
<span id='head'></span>

[TOC]

# 一些个想法

## 小灯泡（idea）

- 既然扩散模型和VAE蛮像的，所以好用的VAE改进框架，扩散模型也得有

- 扩散模型作为一种解耦方法应用于域自适应分割中
  - 利用扩散模型学习潜在空间信息或者内容特征，在利用内容特征做分割，结合跨模态分割方法

    > 穿越回来！貌似没用呢？或许是我理解不够深刻吧


# 扩散模型调研

## 关键词

- diffusion model
- 生成扩散模型
- image to image translation network
- 多模态translation / multi-modal
- 域迁移

## 阅读论文列表

扩散模型：

- 扩散模型：[Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)

- 扩散模型综述

  - [A Survey on Generative Diffusion Model](https://arxiv.org/abs/2209.02646) （[中文](https://mp.weixin.qq.com/s/zKVHoK0ou8ky0IJBnnsStg)）
  - [Diffusion Models in Vision: A Survey](https://arxiv.org/pdf/2209.04747.pdf)

- 扩散模型图像生成（网站文章）：

  - [Diffusion Models for Image-to-Image and Segmentation](https://medium.com/@myschang/diffusion-models-for-image-to-image-and-segmentation-d30468114b27)

- 扩散生成模型：

  - 模型简化）[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) ([中文](https://zhuanlan.zhihu.com/p/603740431))
  - 颜色）[Palette: Image-to-Image Diffusion Models](https://dl.acm.org/doi/abs/10.1145/3528233.3530757)

- 扩散+gan：[Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262)

  > 参考：[Diffusion-GAN: Training GANs with Diffusion 解读](https://blog.csdn.net/weixin_43135178/article/details/127933698?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169072026916800222824988%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169072026916800222824988&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduend~default-1-127933698-null-null.268^v1^koosearch&utm_term=Diffusion-GAN%3A%20Training%20GANs%20with%20Diffusion&spm=1018.2226.3001.4450)

多模态：

- 综述：Multimodal Machine Learning:A Survey and Taxonomy

不同阶段之间的风格迁移：

- [Lifespan age transformation synthesis](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_44)

## 问题

- GAN+扩散模型，能否实现？计算量大

  > 扩散模型用作GAN的生成器，再用判别器判别
  >
  > 扩散模型包括前向过程（加噪）和反向过程（去噪），反向过程就相当于生成器

- 风格迁移到底要迁移成什么程度？粗配准？风格变换后要还原回去吗？

- 配对的问题，从那开始到哪里结束，可能不一样，结构比例有差距，配不上？

## 随笔记

- 扩散模型对于结构改变应该强于GAN，GAN应该适用于图像颜色改变。不同模态的大脑风格迁移应该是在颜色和结构（形态）均有不同。

  > 循环一致损失可以保留图像大致样子
  >
  > 参考：[cyclegan循环一致损失为什么可以保留图像大致样子（轮廓）？](https://www.zhihu.com/question/327326397/answer/701859310)

- 扩散模型，图片引导生成

  ![image-20230723214621412](./扩散模型.assets/image-20230723214621412.png)

## 基础理解

### 正态分布

<img src="./扩散模型.assets/image-20230720112251511.png" alt="image-20230720112251511" style="zoom:50%;" />

- 标准正态分布随机变量。

  > 高斯噪声：一组标准正态随机数，有一小部分超出标准差的范围

### 扩散模型

- 前向过程

  <img src="./扩散模型.assets/image-20230720120900244-1702285268416-71.png" alt="image-20230720120900244" style="zoom:50%;" />

- 反向过程

  > 目的是将$X_T$恢复为$X_0$

### 优缺点分析

![img](./扩散模型.assets/generative-overview.png)

> - GAN：额外的判别器
> - VAE：对准后验分布
> - EBM基于能量的模型：处理分区函数
> - 归一化流：施加网络约束

# 扩散模型理论篇——发疯整理版

闲谈叨叨叨：

> 哇塞，怎么要学的东西那么多！想学习扩散模型，结果看资料开头就讲到VAE，不懂...，好！学VAE！结果KL散度到底是什么！怎么这么多概率公式啊！（抓狂）。没办法，还是了解下吧...
>
> 这篇文章是整个扩散模型学习过程的笔记记录（还有发疯式的心路历程o((>ω< ))o）。尽量整理的知识点全面一点，和大家一起学习！强烈建议先复习复习**概率论**！

目录附上：





## 1 补充一些知识点

> 这一部分如果都了解过，那就忽略掉。不想忽略的，可以在看正文部分涉及到时来翻一翻。（把这一部分放在前面，是因为怕你们坚持不到最后(*/ω＼*)）

概率的东西，大家还记否？如果想推公式的话，建议复习一下。我先列几个可能会用到的公式，按需查看。

### 1.1 一些公式

#### 1.1.1 条件概率的一般形式

$$
\begin{aligned}
&P(A,B,C) = P(C|B,A)P(B,A)=P(C|B,A)P(B|A)P(A)\\
&P(B,C|A)=P(B|A)P(C|A,B)
\end{aligned}
$$

嗯~，能看懂哈＜（＾－＾）＞

#### 1.1.2 基于马尔科夫假设的条件概率

如果满足马尔科夫链关系$A\rightarrow{B}\rightarrow{C}$，那么有
$$
\begin{aligned}
&P(A,B,C) = P(C|B,A)P(B,A) = P(C|B)P(B|A)P(A) \\
&P(B,C|A)=P(B|A)P(C|B)
\end{aligned}
$$
就是A距离C太过遥远，C已经不记得A了，所以A对C的影响没那么大了的意思。

#### 1.1.3 函数的期望

已知概率密度函数$p(x)$，那$x$的期望就是
$$
\begin{aligned}
\mathbb{E}[x] &= \int xp(x)dx \\
&\approx \sum^n_{i=1}x_ip(x_i)(x_i-x_{i-1}),\quad(x_0<x_1<\cdots<x_n) \\
&\approx \frac{1}{n}\sum^n_{i=1}x_i,\quad x_i\sim p(x)
\end{aligned}
$$
第一行就是那个函数期望公式。第二行代表的是数值积分，就是求函数与x轴的面积的那个意义。第三行是采样计算，采样出若干个点算平均值，这个没有用到概率，但是采样出来的数据已经有概率意义了，就是概率大的多出现、概率小的出现次数也少。

那想必下面这个公式，大家应该还有印象吧。
$$
\mathbb{E}_{x \sim p(x)}[f(x)]= \int f(x)p(x)dx \approx \frac{1}{n} \sum^n_{i=1}f(x_i), \quad x_i \sim p(x)
$$


### 1.2 KL散度

> 参考：[初学机器学习：直观解读KL散度的数学概念 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37452654)
>
> 文章里的例子很贴切，公式解释挺好理解的，通俗易懂。这里我就只列一下定义公式，不做过多解释了。

#### 1.2.1 KL散度

KL散度（Kullback-Leibler Divergence）：用来衡量两个分布（比如两条直线之间的匹配程度的方法）

定义公式：
$$
\begin{aligned}
D_{KL}(p||q) &= \sum^{N}_{i=1} \bigg[ p(x_{i}) log \frac{p(x_{i})}{q(x_{i})} \bigg] \\
&= \int p(x) ln \frac{p(x)}{q(x)} dx \\
&= \mathbb{E}_{x \sim p(x)}\bigg[ log\frac{p(x)}{q(x)} \bigg]
\end{aligned} \\
$$
其中，$q(x)$是近似分布，$p(x)$是想要用$q(x)$拟合的真实分布。所以如果两个分布完全匹配的话，应该是$D_{KL}(p||q)=0$. 所以KL散度越小越好，也就是近似分布与真实分布匹配的就越好。后面两行后面扩散模型公式推导会用到。

这里注意一下，$D_{KL}(p||q) \neq D_{KL}(q||p)$，毕竟公式里的$p$和$q$的位置也不一样嘛。$D_{KL}(p||q)$我们叫它前向KL散度，$D_{KL}(q||p)$是反向KL散度。（这个有需要的时候再扩展吧。）

#### 1.2.2 高斯分布的KL散度

假设分布$p$和$q$满足高斯分布，那它们的KL散度的公式就可以写成下面的形式。（推导过程就不写了，有yi点麻烦）
$$
KL(p,q)=log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}
$$
（还记得啥是正态分布...吧...，我相信你记得！）
$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}exp\bigg(-\frac{(x-\mu)^2}{2\sigma^2}\bigg)
, \quad
\sigma = \sqrt{\frac{\sum^{n}_{i=1}(x_i-\mu)^2}{n}}
$$


### 1.3 VAE

> VAE它来了！想必涉猎过生成模型的都听过它的大名吧。之前搞GAN的时候听过但没打算学，结果扩散模型火了，又提到VAE，那就了解了解吧，总得知道谁为啥好，好在哪里，谁有啥不好，不好在哪里。（怎么越学越多啊o(≧口≦)o

> 参考：[变分自编码器（一）：原来是这么一回事 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/5253)
>
> 整个讲解浅显易懂，尤其是VAE的发展过程，涉及到的公式的含义，讲的挺清晰的，建议大家细细品读。这篇文章：[变分自编码器（二）：从贝叶斯观点出发 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/5343)，我觉得也得好好看看，里面的概念和公式推导对之后扩散模型公式理解挺有帮助的。

#### 1.3.1 单层VAE

接下来就用我自己的话大致解释一下VAE是怎么回事。

VAE毕竟也是个生成模型，那生成器是要有的。那用谁来生成？生成之后跟谁比较然后优化呢？VAE的思路是，用我手里的数据$X$（真实样本）去拟合一个分布$p(X)$，然后根据$p(X)$采样出来数据$Z$，通过生成器$g(Z)$生成$\hat{X}$，用$\hat{X}$去与$X$对比，也就是希望把开始的$X$给还原回来。大概就是这么个思路。苏神画了VAE的结构图，挺好理解的，对吧！

![img](https://kexue.fm/usr/uploads/2018/03/4168876662.png)

上面这个图当中，有几个点需要注意：

- VAE为每个样本构造了专属于该样本的正态分布，然后采样来重构。

  - 这个涉及到后验分布。不是假设$p(Z)$是正态分布的，而是假设$p(Z|X)$（后验分布）是正态分布，可以理解吗？就是我在知道X的情况Z的概率属于正态分布，这个情况就是后验分布。因为如果直接假设$p(Z)$是正态分布，采样得到若干$Z$，然后生成$\hat{X}$，我i们不能判断$\hat{X}$与原数据$X$的分布是否一样，毕竟原数据的分布我们不知道。

  - 至于为什么要专属的呢，是因为我们希望的是由$X$样本采样出来$Z$，能够还原为$X$，就是一对一的关系，避免我随便一个$Z$生成的$\hat{X}$不知道对应着哪个$X$.

  - 还有为什么是正态分布呢？这就又要提到KL散度了，所以为什么用到KL散度了呢？... （啊啊啊啊~怎么这么多问题！一会儿再讲！冷静！慢慢来！先往下看！我会提醒你的:)

  - 你先假设，假设是正态分布。

- 均值方差计算模块产生各个样本的均值和方差。这里均值就代表着采样得到的结果，方差就是噪声。

  - 这里的采样结果$Z$是有均值和方差共同影响生成的。训练希望$\hat{X}$接近$X$，那也就意味着，希望噪声没那么强，想让噪声越小越好。但是没有噪声，那得到也只是个确定的结果，也不就没有生成的意义了嘛。

  - VAE为了防止噪声为0，保证生成的能力。规定$p(Z|X)$接近标准正态分布$\mathcal{N}(0,I)$。这里就比较巧妙，先放推导公式：
    $$
    \begin{aligned}
    p(Z) &= \sum_{X}p(Z|X)p(X) \\
    &= \sum_{X}\mathcal{N}(0,I)p(X) \\
    &= \mathcal{N}(0,I)\sum_{X}p(X) \\
    &= \mathcal{N}(0,I)
    \end{aligned}
    $$
    还记得之前说过的先验分布和后验分布吧。你想想看，在规定$p(Z|X)$接近标准正态分布$\mathcal{N}(0,I)$后，先验分布$p(Z)$也满足了正态分布，那是不是就意味着，我们就只是从一个正态分布中采样出来若干个$Z$，这些$Z$也满足正态分布，确确实实地保留着随机性。同时我们也可以判断由$Z$生成的$\hat{X}$与原数据$X$的分布是否一样，因为都是接近正态分布的嘛。

    > 妙啊！兜来兜去又转回来了！（要是没有理解我的解释的话，可以去看一下原文，会讲的更详细一点）

  - 那怎么样让$p(Z|X)$接近标准正态分布$\mathcal{N}(0,I)$呢？KL散度来啦！用$KL\bigg(\mathcal{N}(\mu,\sigma^2)||\mathcal{N}(0,I)\bigg)$来作为额外的loss。

    至于之前问为什么选择正态分布，也是与KL散度有关系，因为公式中存在除法，会涉及到分母为零的情况。正态分布所有点的概率密度都是非负的，所以就避免了这个问题。

    > 这一部分的公式推导很多，但是我感觉目前不用了解太详细。毕竟VAE在这里只是补充了解的作用。如果想深入学习的话，看原文！

下面这张图我觉得挺好的，贴上来看看。

![img](https://kexue.fm/usr/uploads/2018/03/2674220095.png)

其实VAE里也有所谓对抗的过程，之前提到希望$\hat{X}$接近$X$就让噪声越小越好，但是KL Loss又希望有高斯噪声去接近正态分布。这就是一个对抗的过程，但是是共同进步的。与GAN不同，GAN是在训练生成器G的时候，将判别器D冻住。

VAE大概就是这样啦！(o゜▽゜)o☆

#### 1.3.2 重参数技巧

然后还有一个小点，重参数技巧（reparameterization trick）。这个对扩散模型公式推导有大用！

我们是从$p(Z|X_k)$中采样出$Z_k$，没问题吧。我们知道这是正态分布的，但是其实均值和方差都是模型算出来，之后还要通过backward优化这些参数，采样这个动作是不可导的，但是结果是可导的。

所以我们从$\mathcal{N}(\mu,\sigma^2)$中采样一个$Z$，就相当于从标准正态分布$\mathcal{N}(0,I)$中采样一个$\epsilon$，然后让$Z=\mu + \epsilon \times \sigma$. 大概就是这个意思。这段话有大用！

### 1.4 负对数似然

这个在后面求loss函数的时候会涉及到。

 #### 1.4.1 似然函数

似然（likelihood）针对的是影响概率的位置参数，根据观察结果，来估计概率模型的参数。

数学表示就是：假设$X$是离散随机变量，其概率质量函数$p$依赖于参数$\theta$，则
$$
L(\theta | x) = p_{\theta}(x) = P_{\theta}(X = x)
$$
其中，$L(\theta|x)$就是参数$\theta$的似然函数，$x$为随机变量$X$的某一值。

#### 1.4.2 最大似然估计

最大似然估计（Maximum Likelihood Estimation, MLE）就是：

我们有一个数据分布$P_{data}(x)$，但是不知道分布的数学表达。所以就定义一个分布模型$P_G(x;\theta)$，分布由参数$\theta$决定。设$L$是分布模型$P_G(x;\theta)$的样本的似然函数，我们要求出一个$\theta$，使得含有参数$\theta$的似然函数$L$最大，也就是让我们定义的分布模型越接近数据分布。

似然函数表达式如下：
$$
L = \prod^m_{i=1}P_G(x^i;\theta)
$$

#### 1.4.3 对数似然

似然函数是很多项的乘积，取个对数是不是就相加了，计算相对更简单，求导也容易。就这么个作用。
$$
l(\theta)=\sum^m_{i=1}log \big( P_G(x^i;\theta) \big)
$$

#### 1.4.4 负对数似然

要做loss函数的话，就是要越小越好。似然函数我们是要越大越好，所以取负，就是越小越好啦。
$$
l(\theta)= - \sum^m_{i=1}log \big( P_G(x^i;\theta) \big)
$$

## 2 扩散模型

> 好！可算是到正文部分了，还在嘛(\*≧︶≦))(￣▽￣* )ゞ

附上一篇经典论文：[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

<img src="./扩散模型.assets/image-20231128165916608.png" alt="image-20231128165916608" style="zoom:80%;" />

> 一些个参考：
>
> - 视频：[54、Probabilistic Diffusion Model概率扩散模型理论与完整PyTorch代码详细解读\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV1b541197HX/?spm_id_from=333.337.search-card.all.click&vd_source=886b79dff7fd3ea9c414872fed24df59)
> - 文字：[扩散模型之DDPM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/563661713)
>
> 建议视频文字同时看，视频没跟上或者讲的不详细的，看文字版，手边放个演草纸推导一下。然后你会发现虽然不知道到底有没有用，但是推出来的感觉针布戳！我这里就不过多的涉及公式推导了，~~因为打公式打起来真的要命啊~~。建议自己去看给的参考链接，很详细。感觉这些公式对于理解代码来说也挺重要的。
>
> 我如果我没有理解错的话，所有的公式推导和化简的最终形态，就是最后的损失函数，这样就很简单合理地翻译成代码来训练模型啦！感觉公式推导化简完全就是理论抽象成简单的公式的过程，这就是数学的魅力吗ヽ(✿°▽ﾟ)ノ。

扩散模型的过程其实和VAE有点像。整体思路就是先给数据加噪声，反反复复加噪声，让其接近正态分布，然后再还原回来。就这么个事！好！结束（bushi

理论上，模型主要分成两个部分，**前向过程（forward process）**和**反向过程(reverse process)**。前向过程（扩散过程）就是给数据$x_0$加噪，让其充分接近正态分布图像$x_T$（对应着下面这张图的从右向左过程）；反向过程（逆扩散过程）就是去噪的过程，逐渐生成回$x_0$（从左向右的过程）。

<img src="./扩散模型.assets/image-20231128210341142.png" alt="image-20231128210341142" style="zoom:80%;" />

有几个点可以留意一下：

- 扩散过程中的当前图是由上一时间点的图像加噪来的（所有都是满足正态分布的）。但是由于遵循**马尔可夫假设原则**，实际计算上可以直接由$x_0$计算出来。
- 反向过程，就是我们需要建立模型的环节，就是我们需要的生成模型了。那这个模型其实是**拟合一个正态分布**。当然优化的话，就是让模型去拟合扩散过程的正态分布啦，就要用到KL散度。
- 优化目标。这个推导相当之复杂（开摆！）。总之通过一系列假设、化简、代换，最后由计算两个模型的KL散度，转化为计算噪声的拟合程度。也就是说由预测正态分布的均值、方差这些转化成了**预测噪声**。（这是咋想出来的？😔，只能说，🐂🍺！预测噪声这个点知道了的话，代码就能理解了。

### 2.1 扩散过程

扩散过程就是给数据一直加高斯噪声，直到这个数据变成了随机噪声。

具体的数学实现是怎么计算的呢！嘿！嘿！嘿！准备好了嘛（奸笑。

那先进行一些相关说明。我们假设原始数据$x_0$是满足分布$q(x_0)$的，也就是$x \sim q(x_0)$。因为是一遍遍加噪声嘛，所以我们设总共要$T$步的扩散过程，每一步都要对上一步的数据$x_{t-1}$增加噪声，这个噪声的标准差$\sigma$是由一个固定值$\beta_t$来确定的，均值$\mu$是由固定值$\beta_t$和t时刻的数据$x_t$确定的 ，所以加噪的操作就是
$$
q(x_t|x_{t-1})=\mathcal{N}\big(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_tI\big)
$$
整个过程是满足马尔可夫链的，所以就可以得到下面这个公式（提醒你一下，用到了[1.1.2](#1.1.2 基于马尔科夫假设的条件概率)的公式）
$$
q(x_{1:T}|x_0) = \prod^T_{t=1}q(x_t|x_{t-1})
$$
然后，精彩的来了！先说结论，**$x_t$其实可以直接用$x_0$算出来**！

还记得重参数技巧嘛，我们可以将上面这个加噪的过程翻译一下:
$$
\begin{aligned}
x_t &= \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1} \\
&= \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}
\end{aligned}
$$
这里为了方便之后的表达式书写，所以设$a_t=1-\beta_t$，到这里没问题吧？

然后递推公式，t-1时刻也可以用t-2时刻的来表示。具体的推导过程不多说，直接上结论，反正最后化简成这玩应儿了
$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
&= \cdots \\
&= \sqrt{\overline{\alpha_t}}x_0+\sqrt{1-\overline{\alpha_t}}\epsilon
\end{aligned}
$$
所以就可以用分布$q(x_t|x_0)$来表示啦，然后翻译一下：
$$
q(x_t|x_0) = \mathcal{N}\big(x_t;\sqrt{\overline{\alpha_t}}x_0, (1-\overline{\alpha_t})I\big)
$$
是不是很神奇！是！这个先放在这里，后面有用。

正向的部分结束！

### 2.2 反向过程

#### 2.2.1 训练网络建模

逆扩散过程就是从高斯噪声中恢复原始数据，也就是生成数据的过程。我们用神经网络来估计这些分布。反向过程也是一个马尔科夫链过程哦。那我们可以定义一个分布模型：
$$
p_\theta(x_{0:T})=p(x_T) \prod^T_{t=1} p_\theta (x_{t-1}|x_t)
\\
p_\theta(x_{t-1}|x_t) = \mathcal{N} \big(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\big), \quad p(x_T) = \mathcal{N}(x_T; 0,I)
$$
这里的$p(x_T) = \mathcal{N}(x_T;0,I)$，$p_\theta(x_{t-1}|x_t)$是参数化的高斯分布，其均值和方差就是$\mu_\theta(x_t, t)$和$\Sigma_\theta(x_t, t)$，也就是我们要训练的网络啦。

好！记住它！下一个环节！

#### 2.2.2 对比用的标准分布

还记得正向过程中我们留下的式子吧，就是分布$q(x_t|x_0)$。你想想看，我们建好的模型$p_\theta(x_{t-1}|x_t)$的意义是什么？是不是由$x_t$推到$x_{t-1}$的概率，那我们需要的标准是不是也得是由$x_t$推到$x_{t-1}$的概率，所以应该是$q(x_{t-1}|x_t)$啊。

可是现在只有$q(x_t|x_0)$，怎么办？算啊！怎么算？马尔科夫链！直接算不了$q(x_{t-1}|x_t)$，我们可以求$q(x_{t-1}|x_t,x_0)$嘛！
$$
\begin{aligned}
q(x_{t-1}|x_t, x_0) &= q(x_t|x_{t-1}, x_0) \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}
\\
&\propto exp\bigg( -\frac{1}{2} \big( \frac{(x_t-\sqrt{a_t}x_{t-1})^2}{\beta_t}+\frac{(x_{t-1}-\sqrt{\overline{a}_{t-1}}x_0)^2}{1-\overline{\alpha}_{t-1}}-\frac{(x_t-\sqrt{\overline{\alpha}_t}x_0)^2}{1-\overline{\alpha}_t} \big) \bigg)
\\
&= exp \bigg( -\frac{1}{2} \big( (\frac{\alpha_t}{\beta_t}+\frac{1}{1-\overline{\alpha}_{t-1}}) x^2_{t-1} - (\frac{2\sqrt{\alpha_t}}{\beta_t}x_t+\frac{2\sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t-1}}x_0)x_{t-1}+C(x_t, x_0) \big) \bigg)
\end{aligned}
$$
这里C与$x_{t-1}$无关，所以忽略，看成常量就行。所以exp大括号里面就可以看成是一个关于$x_{t-1}$的一元二次的式子了。高中数学的记忆还在不在？配方会不会？就是$ax^2+bx=a(x+\frac{b}{2a})^2+C$，把上面配个方，就成了个正态分布的形式！整理化简一下，这个正态分布$q(x_{t-1}|x_t,x_0)$的均值和方差就是：
$$
\widetilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t} x_t +\frac{\sqrt{\overline{\alpha}_{t-1}}\beta_t}{1-\overline{\alpha}_t} x_0
\\
\widetilde{\beta}_t = \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t} \beta_t
$$
还可以再化简一步，还记得$x_t$可以用$x_0$表示的那个环节吗？是不是$x_0$也可以用$x_t$表示一下，就是移个项再除一下系数那种。就是这样：
$$
x_t = \sqrt{\overline{\alpha_t}}x_0+\sqrt{1-\overline{\alpha_t}}\epsilon
\\
\downarrow
\\
x_0 = \frac{1}{\sqrt{\overline{\alpha}_t}}(x_t-\sqrt{1-\overline{\alpha}_t}\epsilon_t)
$$
那均值里面有$x_0$，所以就可以进一步化简：(把方差也放过来，方便之后观看)
$$
\widetilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \bigg( x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_t \bigg)
\\
\widetilde{\beta}_t = \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t} \beta_t
$$
$q(x_{t-1}|x_t, x_0)$也可以写成正态分布的表达式：
$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\widetilde{\mu}
_t(x_t, x_0),\widetilde{\beta}_tI)
$$
好！这下我们有了训练模型$p_\theta(x_{t-1}|x_t)$和模型的标准参考$q(x_{t-1}|x_t, x_0)$（可忽略$x_0$，因为马尔科夫链），接下来就是优化，也就是要知道loss函数是什么。

### 2.3 优化!

> 从这里开始，逐渐离谱，不想写了（阴暗爬行）我尽力整理归纳，你们尽力理解（扶额）（叹气）这些都是咋想到的呢？
>
> 前面所有的表达式，都是为了优化这步做准备。没记住没关系，要用到什么，我会再相应的地方提醒你或者重复一遍，别嫌我墨迹就行~(￣▽￣)~*

#### 2.3.1 Loss总项

首先，要用到[负对数似然函数](#1.4 负对数似然)（emmm...，这里怎么解释的都有，反正最后推导过程都一样，问题不大）。

我们在负对数似然函数的基础上，加上KL散度，这样有了一个上界，上界小了，负对数似然函数也能小。
$$
\begin{aligned}
-\log{p_\theta(x_0)} &\leq -\log{p_\theta(x_0)}+D_{KL}(q(x_{1:T}|x_0)||p_\theta(x_{1:T}|x_0))
\\
&= \cdots
\\
&=\mathbb{E}_q \big[ \log{\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}} \big]
\\
&=L_T + \sum^T_{t=2} L_{t-1} - L_0
\end{aligned}
$$
然后我们让
$$
L = L_{VLB}= \mathbb{E}_{q(x_{1:T}|x_0)}\big[ \log{\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}} \big]
$$
然后就是展开、化简。
$$
\begin{aligned}
L &= \mathbb{E}_{q(x_{1:T}|x_0)}\big[ \log{\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}} \big]
\\
&= \cdots
\\
&= \underbrace{D_{KL} \big( q(x_T|x_0) || p_\theta(x_T) \big)}_{L_T} + \sum^{T}_{t=2} \underbrace{\mathbb{E}_{q(x_t|x_0)} \big[ D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t)) \big]}_{L_{t-1}} \underbrace{-\mathbb{E}_{q(x_1|x_0)} \log{p_\theta(x_0|x_1)}}_{L_0}
\end{aligned}
$$
过程你别管，就是化简（偷懒）。反正就是最后剩下着三个部分。
$$
\begin{aligned}
& L_T = D_{KL} \big( q(x_T|x_0) || p_\theta(x_T) \big)
\\
& L_{t-1} = \mathbb{E}_{q(x_t|x_0)} \big[ D_{KL}(q(x_t-1|x_t,x_0)||p_\theta(x_{t-1}|x_t)) \big]
\\
& L_0 = -\mathbb{E}_{q(x_1|x_0)} \log{p_\theta(x_0|x_1)}
\end{aligned}
$$
接下来，这三项，就分别求呗。

#### 2.3.2 $L_0$分项

> 这部分没太看懂＞︿＜，等我去研究研究扩散模型的的代码，再回来补充。
>
> 论文中最后对loss进行了化简，没有算上这个部分，所以可以暂时忽略这一趴。

$L_0$这部分相当于从连续空间到离散空间的解码loss，优化的是对数似然，$L_0$可以用估计的$\mathcal{N}(x_0;\mu_\theta(x_1, 1),\sigma^2_1I)$来构建一个离散化的解码器来计算：
$$
\begin{aligned}
p_{\theta}(x_0|x_1) &= \prod^D_{i=1} \int_{\delta_{-}(x_0^i)}^{\delta_{+}(x_0^i)} \mathcal{N}(x_0; \mu_\theta^i(x_1,1),\sigma_1^2)dx
\\
\delta_{+}(x) &=
\begin{cases}
\infty & if \quad x = 1 \\
x+\frac{1}{255} & if \quad x < 1 \\
\end{cases}
\\
\delta_{-}(x) &=
\begin{cases}
-\infty & if \quad x = -1 \\
x - \frac{1}{255} & if \quad x > -1 \\
\end{cases}
\end{aligned}
$$

#### 2.3.3 $L_T$分项

回忆！$q(x_T|x_0)$和$p_\theta(x_T)$是啥？
$$
q(x_t|x_0) = \mathcal{N}\big(x_t;\sqrt{\overline{\alpha_t}}x_0, (1-\overline{\alpha_t})I\big)
\\
p(x_T) = \mathcal{N}(x_T;0,I)
$$
这里DDPM论文里是将方差固定了，所以$L_T$这项就是个常数了。

#### 2.3.4 $L_{t-1}$小项



这里，有意思的来了，就是那个把预测均值转化成预测噪声的骚操作！这个经过实验证明，确实预测噪声要比预测均值效果要好。嗯，神奇！

还是回忆！$q(x_{t-1}|x_t,x_0)$和$p_\theta(x_{t-1}|x_t)$是啥?
$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\widetilde{\mu}
_t(x_t, x_0),\widetilde{\beta}_tI)
\\
p_\theta(x_{t-1}|x_t) = \mathcal{N} \big(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\big)
$$
求它俩的KL散度。这里论文里是将$p_\theta(x_{t-1}|x_t)$的方差$\Sigma_\theta(x_t, t)$设置成一个与$\beta$相关的常数了，还记得[两个正态分布的KL散度](#1.2.2 高斯分布的KL散度)怎么计算吧？
$$
KL(p,q)=log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}
$$
代进去，化简：
$$
D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t)) = \frac{1}{2\sigma^2_t}||\widetilde{\mu}
_t(x_t, x_0) - \mu_{\theta}(x_t, t)||^2 + C
$$
所以
$$
L_{t-1}=\mathbb{E}_{q(x_t|x_0)} \big[ \frac{1}{2\sigma^2_t}||\widetilde{\mu}_t(x_t, x_0) - \mu_{\theta}(x_t, t)||^2 \big] + C
$$
还记得$\widetilde{\mu}_t(x_t, x_0)$是可以进一步化简的吧，就是$x_0$也可以用$x_t$表示的那个：
$$
x_0 = \frac{1}{\sqrt{\overline{\alpha}_t}}(x_t-\sqrt{1-\overline{\alpha}_t}\epsilon_t)
\\
\widetilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \bigg( x_t - \frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_t \bigg)
$$
写标准一点的话，就是
$$
\widetilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \bigg( x_t(x_0,\epsilon) - \frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_t \bigg)
$$
先放一边！还有一项$\mu_{\theta}(x_t, t)$，这个是我们建的模$p_\theta(x_{t-1}|x_t)$的均值。这里作者的骚操作就来了，$\widetilde{\mu}_t(x_t, x_0)$最后是可以用$x_t$和随机变量$\epsilon$表示，对于我们的模型来说，$x_t$是什么？是不是我前向的时候加噪得到的乱糟的图像，所以是已知的，但是噪声模型它不知道啊，所以我们是不是可以预测一个噪声去拟合设立的标准分布！也就相当于用均值去拟合，对不对！

所以：
$$
\begin{aligned}
\mu_{\theta}(x_t, t) &= \widetilde{\mu}_t(x_t, x_0(x_t,\epsilon_{\theta}))
\\
&= \widetilde{\mu}_t \bigg( x_t, \frac{1}{\sqrt{\overline{\alpha}_t}}(x_t-\sqrt{1-\overline{\alpha}_t}\epsilon_{\theta}(x_t)) \bigg)
\\
&= \frac{1}{\sqrt{\alpha}_t} \bigg( x_t-\frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_{\theta}(x_t,t) \bigg)
\end{aligned}
$$
这俩代进去：
$$
\begin{aligned}
L_{t-1} - C &= \mathbb{E}_{x_0,\epsilon} \bigg[ \frac{1}{2\sigma^2_t}|| \frac{1}{\sqrt{\alpha_t}} \bigg( x_t(x_0,\epsilon) - \frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_t \bigg) - \frac{1}{\sqrt{\alpha}_t} \bigg( x_t-\frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_{\theta}(x_t,t) \bigg) ||^2 \bigg]
\\
&= \mathbb{E}_{x_0,\epsilon} \bigg[ \frac{\beta^2_t}{2\sigma^2_t \alpha_t(1-\overline{\alpha}_t)} || \epsilon - \epsilon_{\theta}(\sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t} \epsilon, t) ||^2 \bigg]
\end{aligned}
$$

#### 2.3.5 整理总结

论文里，作者又进一步简化，直接系数不要了，化简成这样：
$$
L_{simple}(\theta)= \mathbb{E}_{x_0,\epsilon} \bigg[ || \epsilon - \epsilon_{\theta}(\sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t} \epsilon, t) ||^2 \bigg]
$$




> 扩散模型理论部分就先写到这里吧！我去看看代码ヽ(゜▽゜　)－C<(/;◇;)/~
>
> 感谢观看！

# 扩散模型改进篇——学麻了版

闲谈叨叨叨

> 上一篇介绍的扩散模型理论是最最最基础的，也就是DDPM这篇论文。后来许多论文都对其进行了改进优化，并且相当有用，有几篇也成为了近些时间发表的扩散模型论文的baseline。总而言之，使用优化版好处多多！

> 这里介绍三篇，循序渐进，一环套一环。

# 扩散模型代码篇——被迫更新版

闲谈叨叨叨：

> 原本想着先做完手头的事情然后再来看代码的，结果被迫一周内学完。来吧！（壮士赴死）
>
> 在这里感谢实验室师兄的讲解以及整理的学习资料，节省了很多时间去背调。

没有看扩散模型理论部分快去看！要不然接下来的代码学习也是一知半解的。公式推导还是很重要的，代码大部分都是根据公式写的（除了网络模块是个U-Net）。理论篇更新了一点内容，以便于更好地理论结合代码（就不应该偷懒不写的T T）。如果不想从头看一遍的话，可以看代码的时候翻到相应的部分，以供理解。

一共两种代码

- 简易通俗demo版（DDPM）

  > 最最最原始的DDPM，可以清楚了解扩散模型的框架结构和运行机制。

- 优化详细baseline版（beat GANs）

  > 对DDPM进行了各种优化改进的综合版，也是大多数论文代码修改的baseline，所以之后想要改代码的话，可以在这个版本的代码进行改动。

# 论文阅读

## 综述论文

### diffusion models in medical

![image-20240109204105751](./扩散模型.assets/image-20240109204105751.png)

![image-20240109204305551](./扩散模型.assets/image-20240109204305551.png)

![image-20240109204143272](./扩散模型.assets/image-20240109204143272.png)

## 基础论文

### Diffusion model DDPM模型

[Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)

- [论文](https://arxiv.org/pdf/2006.11239.pdf)
- [代码]( https://github.com/hojonathanho/diffusion.)

#### 相关信息

- **期刊：**

  > Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.

- **关键词：**

#### 文章主要思想

**原理**：参数化的马尔可夫链（数学推导很巧妙）

![image-20230723215718935](./扩散模型.assets/image-20230723215718935.png)

- 前向过程（扩散过程）<img src="./扩散模型.assets/image-20230720120900244.png" alt="image-20230720120900244" style="zoom:50%;" />

- 反向去噪（生成数据的过程）

  > 目的是将$X_T$恢复为$X_0$

**训练过程**：

![image-20230723215808337](./扩散模型.assets/image-20230723215808337.png)

整体架构是一个基于residual block和attention block的U-Net模型

- 前向加噪

  传入样本数据，随机生成高斯噪声和时间t，获得样本数据的加噪图像

  > 对应的时间t对应不同噪声，代表距离初始状态加噪了几次。所以预测时也需要输入相应的时间t。

- 反向去噪

  预测噪声，还原样本图像，计算预测噪声与前向过程的加噪的损失

![img](https://picx.zhimg.com/v2-cddc136cfb0f8d279ee4ece948e8b86b_r.jpg?source=1940ef5c)

预测则是输入一个随机噪声图，经过去噪过程（上采样）后输出生成结果。

#### 自我想法

- 缺点：大量的采样步骤、较长的采样时间。生成时间长
- 整体就是不断迭代来预测噪声图去噪过程，从而生成结果。如果于GAN结合，是否可以先进行预处理的环节，或者将GAN的生成器换为扩散模型

#### 知识点补充

参考：

- [DDPM模型——pytorch实现](https://blog.csdn.net/Peach_____/article/details/128663957?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-128663957-blog-127990163.235%5Ev38%5Epc_relevant_yljh&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-128663957-blog-127990163.235%5Ev38%5Epc_relevant_yljh&utm_relevant_index=9)

- 扩散模型实际预测的是x0的分布，而不是x0这张图片，保证了随机性

  > 参考：[扩散模型（diffusion model）为什么不训练一个模型，使用x(t)预测x(t-1)?](https://www.zhihu.com/question/574586781/answer/2815559289)



### IDDPM

- [论文]()
- [代码]()

#### 相关信息

- **期刊：**
- **关键词：**

#### 摘要、结论、引言

##### 背景

##### 目的

##### 方法

- 通过简单修改，DDPM可以保持高样本质量的同时，实现竞争对数似然

- 反向扩散过程的学习方差，允许以更少数量级的正向采样进行采样，样本差异忽略不计，有利于后续的部署。

  DDPM是固定方差训练的，IDDPM是使用

##### 结论

#### 文章主要思想

#### 自我想法

##### 想法

##### 图表

##### 英语句式

#### 知识点补充

- [翻译](https://blog.csdn.net/zzfive/article/details/127169343?ops_request_misc=%7B%22request%5Fid%22%3A%22170305493716800185893909%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170305493716800185893909&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduend~default-3-127169343-null-null.nonecase&utm_term=Improved Denoising Diffusion Probabilistic Model&spm=1018.2226.3001.4450)
- 竞争对数似然

### DDIM

### Stable Diffusion

> paper: [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
>
> code：https://github.com/CompVis/latent-diffusion

#### 文章主要思想

![image-20240306185759610](./扩散模型.assets/image-20240306185759610.png)

流程：

- 正向扩散：
  - 图像输入encoder，提取图像特征
  - 扩散加噪过程，得到加噪特征
- 逆向过程
  - 加噪特征输入unet去噪模型
  - 同时伴有条件，如文本、图像......：CLIP text encoder
    - 通过cross attention方式送入扩散模型中作为条件
  - 直至还原为目标特征
  - 预测特征经过decoder还原图像

<img src="./扩散模型.assets/image-20240306193641390.png" alt="image-20240306193641390" style="zoom:50%;" />

generation model 部分：

<img src="./扩散模型.assets/image-20240306193713824.png" alt="image-20240306193713824" style="zoom:50%;" />

<img src="./扩散模型.assets/image-20240306193814108.png" alt="image-20240306193814108" style="zoom:50%;" />

训练细节：

- 所以是用输入的特征进行训练，然后采样回原特征。
- encoder和decoder预训练，用自编码器。
- 中间扩散模型对特殊任务进行微调。

优点

- 图像的潜在空间比像素级空间要小，所以参数量会小
- 这个条件，是为了使用多模态

#### 思考

- decoder可换成其他下游任务
  - 是特征层面上的迁移，encoder和decoder有操作空间。
  - 源域训练分割encoder-decoder，源域训练扩散模型
  - 可将目标域图像提取特征，特征迁移至源域特征。迁移特征经过源域分割decoder，完成域适应分割。

#### 知识点补充

> 参考：
>
> - [【论文简介】Stable Diffusion的基础论文:2112.High-Resolution Image Synthesis with Latent Diffusion Models](https://blog.csdn.net/imwaters/article/details/127269368?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170925912116800227437238%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170925912116800227437238&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduend~default-2-127269368-null-null.nonecase&utm_term=stable%20diffusion%20%E8%AE%BA%E6%96%87&spm=1018.2226.3001.4450)

- 预训练

  > 参考：[深入理解：什么是预训练？预训练有什么作用？预训练和训练的本质区别？？？](https://blog.csdn.net/weixin_45325693/article/details/132084298)

  - 无监督预训练：自编码器、变分自编码器、......
  - 有监督预训练：具体的任务

- Autoencoder 自编码器

  > 参考：
  >
  > - [自编码器(Autoencoder)基本原理与模型实现](https://blog.csdn.net/weixin_60737527/article/details/126787892)
  > - [自编码器（Autoencoders）：从原理到实践](https://blog.csdn.net/DeepViewInsight/article/details/132959108?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-132959108-blog-126787892.235^v43^pc_blog_bottom_relevance_base2&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

  - 无监督学习算法，数据降维或者特征抽取
  - 自己训练生成自己！

- 自编码器+latent diffusion：维度处理、

- 预训练和微调

  > 参考：[预训练（pre-training/trained）和微调（fine-tuning）的区别](https://juejin.cn/post/7280861248421396491

- 冻结

  > 参考：[Pytorch在训练时冻结某些层使其不参与反向传播](https://www.cnblogs.com/douzujun/p/16912324.html)

- VQ-VAE

  > 参考：[详解VQVAE：Neural Discrete Representation Learning](https://blog.csdn.net/justBeHerHero/article/details/128203770)

- 少样本学习

  > 参考：[什么是Few-shot Learning](https://www.jianshu.com/p/12bac27a6a1b)

  - meta learning

    - 给个support set训练，去判别不知道的query类别

      > 还和训练集不太一样。support set数据少，training set数据量大。

    - 如果support set的每个类别只有一个数据，就叫one-shot learning

    meta learning是从其他学习算法的输出中学习，其他学习算法被预训练过，然后mata learning算法将其他学习算法的输出作为输入，进行回归和分类预测。

    > 主要是起到一个区分的作用，至于任务目标不重要。
    >
    > 跟监督学习也不太一样：监督学习虽然测试样本没有见过，但是类别什么的还是在训练中看见过的。query样本没有见过，类别也不知道，所以就需要提供一个support set，通过对比query和support set间的相似度，来预测query属于哪个类别。

### Diffusion-GAN

[Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262)

- [论文](https://arxiv.org/pdf/2206.02262.pdf)
- [代码](https://github.com/Zhendong-Wang/Diffusion-GAN)

#### 文章主要思想

**diffusion与GAN的结合**

diffusion-gan由三个部分组成：自适应扩散过程、扩散时间步长相关鉴别器和生成器。

- 观测数据和生成数据都通过相同的自适应扩散过程进行扩散。
- 在每个扩散时间步长，有不同的噪声数据比，时间步长相关的鉴别器学习区分扩散的真实数据和扩散的生成数据。
- 生成器通过正向扩散链的反向传播从鉴别器的反馈中学习，该正向扩散链的长度可自适应调整以平衡噪声和数据水平。

<img src="./扩散模型.assets/image-20230730223103328.png" alt="image-20230730223103328" style="zoom:80%;" />

由第一行对real image进行扩散加噪，同时又由第三行对noise去噪恢复fake image。鉴别器对每一个时间步长均进行鉴别

#### 自我想法

- 将cyclegan的思想加入，自监督

  > 可能会速度慢？

### Diffusion Autoencoders

[扩散模型论文阅读 | Diffusion Autoencoders: Toward a Meaningful and Decodable Representation](https://zhuanlan.zhihu.com/p/591063051)

[【AIGC第十三篇】Diffusion Autoencoders：基于latent space控制的图像生成技术 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/625386246)

> **【文章题目】**Diffusion Autoencoders: Toward a Meaningful and Decodable Representation
>
> **【文章出处】**CVPR2022
>
> **【原文链接】**[https://arxiv.org/pdf/2111.15640.pdfarxiv.org/pdf/2111.15640.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2111.15640.pdf)
>
> 【代码】https://github.com/phizaz/diffae

![image-20240305203419834](./扩散模型.assets/image-20240305203419834.png)

### 去噪编码器

何凯明的关于扩散模型：

> 论文：[Deconstructing Denoising Diffusion Models for Self-Supervised Learning](https://arxiv.org/pdf/2401.14404.pdf)
>
> 代码：https://github.com/FutureXiang/ddae/tree/main/model
>
> 讲解：[何恺明最新力作：一文解构扩散模型，l-DAE架构或将颠覆AI认知？](https://zhuanlan.zhihu.com/p/680129777)

## 网络简化

### [ICLR2023] 扩散生成模型新方法：极度简化，一步生成

[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)

- [论文](https://arxiv.org/abs/2209.03003)
- [代码](https://github.com/gnobitab/RectifiedFlow)

#### 相关信息

- **期刊：**
- **关键词：**

#### 摘要、结论、引言

##### 背景

##### 目的

##### 方法

##### 结论

#### 文章主要思想

#### 自我想法

##### 想法

##### 图表

##### 英语句式

#### 知识点补充

## 图像迁移

## 超分辨率

### Image Super-Resolution via Iterative Refinement

[Image Super-Resolution via Iterative Refinement | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9887996)

- [代码](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/master)

超分辨率的应用，扩散模型的条件生成

改进点：将低分辨率进行三次双线性插值后与噪声图进行拼接后作为估计噪声。采样的时候也是

训练阶段：

![img](https://img-blog.csdnimg.cn/74a5ed06a080497eb9e753914c16abf8.png)

推理阶段：

![img](https://img-blog.csdnimg.cn/d37ed08f901c48b780f6ba26684584c8.png)

![image-20240109155255141](./扩散模型.assets/image-20240109155255141.png)

![image-20240109160751061](./扩散模型.assets/image-20240109160751061.png)

没事了，钟师兄就是基于这个改的。

### Relay Diffusion

[Relay Diffusion：Unifying diffusion process across resolutions for image Synthesis](https://arxiv.org/abs/2309.03350)

![image-20240630162714930](./扩散模型.assets/image-20240630162714930.png)

- [论文](https://arxiv.org/abs/2309.03350)
- [代码](https://github.com/THUDM/RelayDiffusion)

#### 相关信息

- **期刊：**ICLR 2024

#### 摘要、结论、引言

##### 背景

- 流行的解决方案

  - latent (stable) diffusion：在隐空间内训练，再映射回像素空间
    - 但不可避免地会受到底层伪影(low-level artifacts)的影响；

  - 训练一系列不同分辨率的超分扩散模型构成级联
    - 现有的级联方法是有效的，但它需要每个阶段的从噪音开始完整采样，效率较低，且效果严重依赖于条件增强等训练技巧。

- 低频：纹理信息（语义信息）；高频：边缘信息

- noise schedule：

  - 理想的噪声调度应该是分辨率相关的。

    > 直接使用32\*32或者64\*64分辨率设计的通用调度，来训练高分辨率模型，性能肯能不是最优。

##### 结论

- 对于更高分辨率的图像进行加噪，其加噪后的结果在低频处有更高的信噪比（Signal-to-Noise Ratio，SNR）

#### 文章主要思想

![image-20240722091153366](./扩散模型.assets/image-20240722091153366.png)

RDM通过离散余弦变换频谱分析发现

- 相同噪声强度在更高的分辨率下对应于频率空间的信噪比(SNR)在低频部分更高

  > 这意味着自然图像的低频信息没有被很好的破坏掉。

- 为此，RDM提出了一种像素点间具有相关性的块状噪音--block noise

  - 它在高分辨率下对应的SNR在低频部分和高斯噪音在低分辨率下的SNR相当。

![image-20240630162655813](./扩散模型.assets/image-20240630162655813.png)

- block noise
  - 低分辨率的高斯噪声通过一个s*s的卷积和将分辨率映射为原来的s倍，从而适配高分辨率图像的训练。
  - 如果我们要训练256×256分辨率下的diffusion models，那么s=4，即将64×64的低分辨率噪声映射成256×256的。前文图中的实验也表明，通过block noise加噪之后的图像直观上的污染程度与低分辨率图像是相近的，同时，其低频段上的问题也得到了很好解决。

- 两段式：生成+超分

  以64➡256为例，RDM的整体流程为：

  - 先通过标准扩散过程生成低分辨率图片
  - 再将其上采样为每个4×4网格具有相同像素值的模糊高分辨率图片
  - 之后对每个4×4的网格独立进行模糊扩散过程(blurring diffusion)

  > 这样使得前向过程的终态和上采样的模糊图片对齐
  >
  > 因此RDM的第二阶段可以直接以模糊图片为起始点，而不是现有级联方法中的纯高斯噪音。

#### 自我想法

- 但是还是得在高分辨率上训练，只不过不用加条件，这样能省一部分显存。

#### 知识点补充

> 参考：
>
> - [从Relay Diffusion到Cogview 3：浅谈Noise Scheduling与扩散模型](https://zhuanlan.zhihu.com/p/686899891)
> - [清华团队扩散模型新研究：统一不同分辨率的扩散过程](https://zhuanlan.zhihu.com/p/655134629)

- 模糊扩散

  > 参考：[BLURRING DIFFUSION MODELS (Paper reading)](https://blog.csdn.net/qq_43800752/article/details/129699787)

------

### CogView3

[CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion](https://arxiv.org/abs/2403.05121)

- [论文](https://arxiv.org/pdf/2403.05121)
- [代码]()

#### 相关信息

- **期刊：**
- **关键词：**
  - Text-to-Image Generation
  - Diffusion Models


#### 文章主要思想

##### ![QQ_1721568579840](./扩散模型.assets/QQ_1721568579840.png)

#### 知识点补充

- 信噪比：

  > 参考:[图像的 SNR 和 PSNR 的计算](https://blog.csdn.net/qq_41498261/article/details/110141202)

  首先计算图像所有象素的局部方差，将局部方差的最大值认为是信号方差，最小值是噪声方差，求出它们的比值，再转成dB数，最后用经验公式修正。

  - 噪声方差越小，信噪比越大，图像越接近，越好。

### Diff-Plugin

[Diff-Plugin: Revitalizing Details for Diffusion-based Low-level Tasks](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Diff-Plugin_Revitalizing_Details_for_Diffusion-based_Low-level_Tasks_CVPR_2024_paper.html)

- [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Diff-Plugin_Revitalizing_Details_for_Diffusion-based_Low-level_Tasks_CVPR_2024_paper.pdf)
- [代码](https://github.com/yuhaoliu7456/Diff-Plugin/)

#### 相关信息

- **期刊：** CVPR 2024

#### 摘要、结论、引言

##### 背景

- 单纯的扩散模型，在细节保留上，不得行

##### 方法

- Diff-Plugin：单独的预训练扩散模型

  > 高保真

  1. Task-Plugin模块：提供特定于任务的先验，指导保存图像内容的扩散过程
     1. TPB预先提取任务指导，将扩散模型定向到指定的视觉任务，并最大限度地减少其对复杂文本描述的依赖
     2. SCB利用TPB的任务特定视觉指导来辅助空间细节的捕获和补充，增强生成内容的保真度。
  2. Plugin-Selector：根据文本指令自动选择不同的任务插件

##### 贡献：

1. 我们提出了Diff-Plugin，这是第一个使预训练的扩散模型能够在保持原始生成能力的同时执行各种低级任务的框架。
2. 我们提出了一个Task-Plugin，这是一个轻量级的双分支模块，旨在将特定于任务的先验注入扩散过程，以提高结果的保真度。
3. 我们建议使用Plugin-Selector根据用户提供的文本选择合适的Task-Plugin。这扩展到一个新的应用程序，允许用户通过文本指令编辑图像，用于低级视觉任务。
4. 我们在八个任务上进行了广泛的实验，demon对比了Diff-Plugin与现有的基于扩散和回归的方法的竞争性能

#### 文章主要思想

![QQ_1721569021807](./扩散模型.assets/QQ_1721569021807.png)

根据用户提示，识别合适的Task-Plugin P，提取特定于任务的先验，然后将其注入预训练的扩散模型中，生成用户期望的结果。

- 给定一个图像，用户通过文本提示指定任务，单个或多个都可以，Plugin-Selector为它识别合适的task - plugin。
- 然后，Task-Plugin处理图像以提取特定于任务的先验，指导预训练的扩散模型产生用户期望的结果。
- 对于超出单个插件范围的更复杂的任务，Diff-Plugin使用预定义的映射表将它们分解为子任务。每个子任务都由指定的Task-Plugin处理，展示了框架处理多样化和复杂用户需求的能力。

![QQ_1721569750439](./扩散模型.assets/QQ_1721569750439.png)

- Task-Plugin模块由两个分支组成:任务提示分支(TPB)和空间补语分支(SCB)。
  - TPB对于为预训练的扩散模型提供特定任务的指导至关重要，类似于在文本条件图像合成中使用文本提示
  - 引入了SCB来有效地提取和增强空间细节的保存。

## 图像生成

### ControlNet

[Adding Conditional Control to Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)

- [论文](https://arxiv.org/abs/2302.05543)
- [代码](https://github.com/lllyasviel/ControlNet)

#### 相关信息

- **期刊：**ICCV 2023

#### 摘要、结论、引言

##### 背景

- 对数据有限的大型预训练模型进行直接微调或继续训练肯能会导致过拟合和灾难性遗忘。
  - 这种遗忘，可以通过限制可训练参数的数量或等级来缓解。

##### 方法

- 加入空间条件控制
- 锁定扩散模型，重用深度和robust编码层（预训练过）
  - 锁定参数，保持大模型的质量和功能。
  - 并制作编码层的可训练副本。
- 神经结构与“零卷积”（零初始化卷积层）相连接
  - 该卷积层从零开始逐步增长参数，并确保没有有害噪声影响微调。
  - 可训练副本和原始锁定模型用零卷积层链接。权值初始化为零，一遍在训练过程中逐渐增长。
-
  - 单张3090训练，训5天
  - SD v2

##### 结论

- 贡献：
  1. controlnet：通过有效的微调将空间局部化的输入条件添加到预训练的文本到图像扩散模型中
  2. 预训练的controlnet，控制稳定扩散，canny边缘，hough线，用户涂鸦，人类关键点，分割图，形状法线，深度图，卡通线条图
  3. 实验

#### 文章主要思想

![QQ_1721096467838](./扩散模型.assets/QQ_1721096467838.png)

![QQ_1721097128868](./扩散模型.assets/QQ_1721097128868.png)

> x：特征映射；c：条件向量

- 锁定原始block，创建可训练副本
- 条件向量经过零卷积（权重和偏置初始化为零的1*1卷积）与特征映射结合后输入可训练副本。

![QQ_1721051012432](./扩散模型.assets/QQ_1721051012432.png)

潜在特征是64*64

条件：

- Canny Edge [11],
- Depth Map [69],
- Normal Map [87],
- M-LSD lines [24],
- HED soft edge [91],
- ADE20K segmentation [96],
-  Openpose [12],
- user sketches.

#### 自我想法

- 在外围复制一份预训练的网络块，加入条件特征训练，在回到原流程上，以此类推。

  - 所以这也算是微调模型，只不过是复制一份微调，然后与原版结果还做了个结合。

  整体下来微调训练的是一半的unet，显存消耗也会更小一点。

#### 知识点补充

> 参考：[Stable Diffusion — ControlNet 超详细讲解](https://blog.csdn.net/jarodyv/article/details/132739842)

- 零卷积

### BrushNet

[BrushNet A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion](https://arxiv.org/abs/2403.06976)

- [论文](https://arxiv.org/pdf/2403.06976)
- [代码](https://github.com/TencentARC/BrushNet)

#### 相关信息

- **期刊：**ECCV 2024
- **关键词：**
  - Image Inpainting
  - Diffusion Models
  - Image Generation


#### 摘要、结论、引言

##### 背景

- 语义不一致和图像质量降低的问题
- 绘图需要具有强约束信息的像素到像素约束，而不是依赖文本完成内容的稀疏结构控制

##### 目的

##### 方法

- 新范式：将掩码图像特征和噪声潜在划分为单独的分支
- 提出模型：BrushNet
  - 即插即用双分支模型
  - 将像素级掩码图像特征嵌入任何预训练的DM中，从而保证连贯和增强的图像绘制结果

##### 结论

- 这种划分极大地减少了模型的学习负荷，促进了以分层方式将基本mask图像信息细致地结合在一起

- 设计：

  1. 使用VAE编码器，而不是随机初始化的卷积层来处理掩码图像
  2. 采用分层方法，将完整的unet特征逐层逐渐纳入与预训练的unet中，
  3. 去除了unet当中的文本交叉注意力
  4. 模糊混合策略

- 新数据集：BrushData

  新基准：BrushBench

#### 文章主要思想

![QQ_1721359107330](./扩散模型.assets/QQ_1721359107330.png)

![QQ_1721359126220](./扩散模型.assets/QQ_1721359126220.png)

流程：

1. mask下采样
2. mask image 经过VAE编码器，对其潜在空间的分布
3. 将下采样的mask、latent mask image和latent noise 一起concat一下，作为brushNet的输入
4. BrushNet中提取的特征经过零卷积块后，逐层添加到预训练的unet中
   - 去除交叉注意力层：确保在这个额外的分支中只考虑纯图像信息。
5. 生成的图像和mask image，跟模糊的mask混合
   - 首先模糊蒙版，然后使用模糊的蒙版执行复制和粘贴

展望：

- 依赖基础模型
- mask出现异常形状或者不规则形状的情况，文本与mask不对应的情况，都会导致生成效果不好。

#### 自我想法

##### 想法

- 条件图像特征辅助图像生成，文本决定性强度减弱。

  - --> mask特征辅助生成，边缘条件强度减弱。

    > 不行，测试的时候，测试数据没有mask。

  - 结构解耦特征可用。

#### 知识点补充

------

### VPD

[Unleashing Text-to-Image Diffusion Models for Visual Perception](http://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Unleashing_Text-to-Image_Diffusion_Models_for_Visual_Perception_ICCV_2023_paper.html)

- [论文](http://arxiv.org/abs/2303.02153)
- [代码](https://github.com/wl-zhao/VPD)

#### 相关信息

- **期刊：**ICCV 2023

#### 摘要、结论、引言

##### 背景

- 难点
  - 扩散模型的pipeline与视觉感知任务之间的不兼容性
  - UNet-like扩散模型与流行视觉骨架之间的架构差异

##### 目的

- 研究如何利用从文本到图像中学习到的知识进行视觉感知

##### 方法

- 不使用分步扩散pipeline，而是简单地使用自动编码器作为骨干模型，直接使用没有噪声的自然图像，并通过设计prompt执行一个额外的去噪步骤来提取语义信息。
- 为了与预训练阶段保持一致，并促进视觉内容和文本提示之间的交互，使用适当的文本输入来提示去噪扩散模型，并使用适配器来优化文本特征。
- 受先前扩散模型中提示词与视觉模式关系研究的启发，提出利用视觉和文本特征之间的交叉注意力映射来提供显式的指导。

##### 结论

- 视觉语言预训练的扩散模型可以更快地适应下游的视觉感知任务

#### 文章主要思想

![image-20240722111209696](./扩散模型.assets/image-20240722111209696.png)

![image-20240306104610972](./扩散模型.assets/image-20240306104610972.png)

流程：

- 文本

  - 文本输入使用" a photo of a [CLS] "模板简单定义

  - 文本encoder得到text feature

  - text adapter
    - 将文本编码器转移到下游任务时，通常会出现域gap
    - 两层MLP实现的文本适配器来细化CLIP文本编码器得到的文本特征

- 图像

  - 使用VQGAN（简写为E）的编码器将图像编码到潜在空间(z0=E(x))

- 将潜在特征图和条件输入输入到预训练的eθ网络，在给定输入图像x和条件输入C的情况下，提取分层特征图F

- 将平均后的交叉注意力图与原始的层次特征图进行拼接，并将结果反馈给预测头

- 经过由分层特征图F生成结果的预测头。

  - 它的组成为语义FPN（特征金字塔），由多个卷积层和上采样层组成

缺点：

- 计算成本高

#### 知识点补充

> 参考：[清华团队新作 | 从Text-to-Image扩散模型中提取表征，服务下游任务](https://blog.csdn.net/qq_41234663/article/details/129524446)

- 域gap

  > 参考：[domain gap(域间隙)是什么？==＞在一个数据集上训练好的模型无法应用在另一个数据集上](https://blog.csdn.net/weixin_43135178/article/details/117767751)

### 潜在扩散医学图像生成

DIFFUSION PROBABILISTIC MODELS BEAT GAN ON MEDICAL 2D IMAGES

- [论文](https://arxiv.org/pdf/2212.07501.pdf)
- [代码](https://github.com/mueller-franzes/medfusion?tab=readme-ov-file)

![image-20240311093544491](./扩散模型.assets/image-20240311093544491.png)

## 风格迁移

### InST

[Inversion-based Style Transfer with Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Inversion-Based_Style_Transfer_With_Diffusion_Models_CVPR_2023_paper.html)

- [论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Inversion-Based_Style_Transfer_With_Diffusion_Models_CVPR_2023_paper.pdf)
- [代码](https://github.com/zyxElsa/InST)

#### 相关信息

- **期刊：**CVPR 2023

#### 摘要、结论、引言

##### 背景

- 不提供复杂的文本描述的情况下指导合成

##### 方法

- 以实例为导向的艺术图像生成框架：

  - 基于反转的风格迁移方法

    学习单个绘画图像的高级文本描述，指导文本到图像的生成模型生成特定艺术外观的图像。

    > 高效准确地学习图像的关键信息，从而捕捉和迁移绘画的艺术风格。

- 基于注意力的文本反演方法，将一幅画反演成相应的文本embedding。

##### 结论

- GPU：
  - 保留了sdm原有的超参数选择。
  - 在NVIDIA GeForce RTX3090中，每个图像的训练过程大约需要20分钟，批量大小为1。基本学习率设置为0.001。
  - 合成过程所需的时间与SDMs相同，具体取决于步骤。

#### 文章主要思想

![QQ_1721394352107](./扩散模型.assets/QQ_1721394352107.png)

![QQ_1721395417321](./扩散模型.assets/QQ_1721395417321.png)

获得特定绘画的预训练文本到图像模型的中间表示。假设真实语言中不存在这样一个文本描述，无法表达艺术形象，所以就创造了一个“新词”

- 图像经过CLIP，学习到相应的文本向量，将文本向量编码为文本条件向量。其余与潜在扩散相同。

关于生成的文本向量如何优化问题：

- 基于多层交叉注意的学习方法
  - 图像输入到CLIP图像编码器，提供图像embedding
  - 通过对这些image embedding进行多层关注，可以快速获取图像的关键信息。

生成部分：

- 将基于预训练文本到图像扩散模型的图像表示分为两部分：
  1. 整体表示：涉及文本条件
  2. 细节表示：受随机噪声控制
- 提出了随机反演来保持内容图像的语义
  1. 在内容图像中加入随机噪声
  2. 使用扩散模型中的去噪unet预测图像中的噪声。
  3. 生成过程中使用预测噪声作为初始输入噪声以保留内容。

缺点：

- 颜色存在显著差异时，可能无法在语义上实现一对一对应的颜色传递。

  > 染色问题！

#### 自我想法

- “边缘”反演
- 图像除了边缘信息可以利用，作为结构性约束，还有其他的信息可以利用吗？
  - 重点在结构上的约束
- 文本提示也可以换成这张图像所在位置
  - 利用了3d空间信息

- 颜色迁移问题

##### 想法

##### 图表

##### 英语句式

#### 知识点补充

> 参考：[【深度学习】InST，Inversion-Based Style Transfer with Diffusion Models，论文，风格迁移，实战](https://blog.csdn.net/x1131230123/article/details/131975560)

- 反演

### Cycle-diffusion

[A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_A_Latent_Space_of_Stochastic_Diffusion_Models_for_Zero-Shot_Image_ICCV_2023_paper.html)

- [论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_A_Latent_Space_of_Stochastic_Diffusion_Models_for_Zero-Shot_Image_ICCV_2023_paper.pdf)
- [代码](https://github.com/humansensinglab/cycle-diffusion)

#### 相关信息

- **期刊：**ICCV 2023
- **关键词：**

#### 文章主要思想

![QQ_1721563437129](./扩散模型.assets/QQ_1721563437129.png)

> 1. 使用两个独立预训练的扩散模型（例如，猫狗）进行非配对图像到图像的翻译。
>
> 可以以一种无监督的方式将猫图像的纹理特征转移到狗模型上。
>
> 2. 基于预训练文本到图像扩散模型的zero-shot图像编辑。使用文本提示来编辑图像
> 3. 使用CLIP等现成的图像理解模型对预训练的扩散模型进行即插即用指导。

贡献：

- 使用随机扩散模型进行zero-shot和unpair图像编辑的方法

  > 比确定性模型提高了性能

- 在确定性和随机扩散模型的潜在空间中展示了统一的、即插即用的指导。

本文证明了：

1. 随机扩散模型也可以有一个潜在空间
2. 真实图像可以编码到这个潜在空间中
3. 潜在空间可以像确定性扩散模型的潜在空间一样被使用

提出方法：

- 提出了DPM-Encoder算法
  - 将真实图像编码到随机dpm的潜在空间中。
  - 像真实图像添加噪声的过程是预先定义的。所以可以从前向过程中抽取连续的噪声图像，并根据定义计算去噪所用的噪声。

#### 知识点补充

- 随机扩散模型、确定型扩散模型

## 分割

### Diff-UNet

Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation

![image-20240308150751131](./扩散模型.assets/image-20240308150751131.png)

### LSegDiff

LSegDiff: A Latent Diffusion Model for Medical Image Segmentation

> 去！没代码！啊啊啊啊啊啊啊啊啊！

- 在特征层面上的扩散模型
- 前后的encoder和decoder是预训练的

![image-20240308150422088](./扩散模型.assets/image-20240308150422088.png)

#### 想法

- encoder和decoder为分割模型，扩散模型对深层特征去噪

- 不对啊，没想明白，train的时候图像通过encoder进行特征提取，然后通过扩散模型，再然后通过decoder。那推理呢？我随便生成一个噪声，然后扩散模型采样获得特征，那跟encoder提取特征还有什么关系呢？

  > mouyiyo，wakalanai

- 啊！桥豆麻袋！

- 你想想反正都是纯噪声生成潜在特征，那为什么不能将目标域的特征作为条件，然后生成源域的特征，在进行分割呢

  > 这是啥领域，域适应？半监督？自适应？
  >
  > 效果呢？
  >
  > 哎，又回到这个问题，没效果，咋整

### MedSegDiff

[MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://proceedings.mlr.press/v227/wu24a.html)

> MIDL 2023

- [论文](https://proceedings.mlr.press/v227/wu24a/wu24a.pdf)
- [代码](https://github.com/KidsWithTokens/MedSegDiff/blob/master/README.md)

![image-20240311094348331](./扩散模型.assets/image-20240311094348331.png)

### Medsegdiff-v2

[Medsegdiff-v2: Diffusion based medical image segmentation with transformer](https://arxiv.org/abs/2301.11798)

> AAAI 2024
>

- [论文](https://arxiv.org/pdf/2301.11798v2.pdf)
- [代码](https://github.com/KidsWithTokens/MedSegDiff)
