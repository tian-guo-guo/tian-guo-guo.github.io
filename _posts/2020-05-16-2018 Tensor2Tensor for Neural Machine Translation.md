---
layout:     post           # 使用的布局（不需要改）
title:      2018 Tensor2Tensor for Neural Machine Translation             # 标题 
subtitle:   Tensor2Tensor for Neural Machine Translation Modeling   #副标题
date:       2020-05-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 技术


---

# 2018 Tensor2Tensor for Neural Machine Translation

![image-20200516145158589](https://tva1.sinaimg.cn/large/007S8ZIlly1geuamnlldmj30ry07mmyl.jpg)

# 摘要

​		Tensor2Tensor是一个深度学习模型库，非常适合神经机器翻译，包括最先进的Transformer模型的参考实现。

# 1 神经机器翻译背景

​		使用深度神经网络的机器翻译在序列对序列模型（Sutskever et al.，2014；Bahdanau et al.，2014；Cho et al.，2014）中取得了巨大成功，该模型使用递归神经网络和LSTM细胞（Hochreiter and Schmidhuber，1997）。 基本的序列到序列架构由一个RNN编码器组成，该编码器每次读取源语句一个令牌，并将其转换为固定大小的状态向量。 这之后是一个RNN解码器，它从状态向量生成目标句子，每次一个标记。

​		虽然纯序列到序列递归神经网络已经可以获得很好的翻译结果（Sutskever et al.，2014；Cho et al.，2014），但它的缺点是整个输入句子需要被编码到一个单一大小的向量中。 这明显地表现在较长句子的翻译质量下降，在Bahdanau等人的研究中部分地得到了克服。 （2014）通过使用注意力的神经模型。

​		从Kalchbrenner和Blunsom（2013）开始，到Meng等人的研究，卷积结构在词级神经机器翻译中得到了很好的效果。 （2015年）。 这些早期的模型在卷积的基础上使用标准的RNN来生成输出，这会产生瓶颈并损害性能。

​		Kaiser和Bengio（2016）和Kalchbrenner等人首次实现了无此瓶颈的全卷积神经机器翻译。 （2016年）。 扩展神经GPU模型（Kaiser and Bengio，2016）使用了循环的选通卷积层堆栈，而ByteNet模型（Kalchbrenner et al.，2016）则取消了递归，在解码器中使用了左填充卷积。 在WaveNet（van den Oord等人，2016）中引入的这一思想显著提高了模型的效率。 同样的技术最近在许多神经翻译模型中得到了改进，包括Gehring等人。 （2017）和Kaiser等人。 （2017年）。

# 2 Self-Attention

​		可以使用堆叠的自我注意层来代替卷积层。 这在Transformer模型中引入（Vaswani等人，2017年），显著提高了机器翻译和语言建模的最新水平，同时也提高了训练速度。 研究继续在更多领域应用该模型，探索自我注意机制的空间。 显然，在通用序列建模中，自我注意是一个强有力的工具。

​		虽然RNN表示隐藏状态下的序列历史，但变压器没有这种固定大小的瓶颈。 取而代之的是，通过点积注意机制，每个时间步都可以完全直接访问历史。 这样既可以使模型学习更远的时间关系，又可以加快训练速度，因为不需要等待隐藏状态跨时间传播。 这是以内存使用为代价的，因为注意力机制与t2成比例，其中t是序列的长度。 未来的工作可能会降低这个比例因子。

​		变压器模型如图1所示。 它对编码器和解码器都使用了堆叠的自关注和逐点的，完全连接的层，分别如图1的左半部分和右半部分所示。

​		编码器:编码器由一堆相同的层组成。 每一层有两个子层。 第一个是一个多头自我注意机制，第二个是一个简单的，位置全连接的前馈网络。

​		解码器:解码器也是由一堆相同的层组成的。 除了每个编码器层中的两个子层之外，解码器插入第三个子层，其在编码器堆栈的输出上执行多头关注。
​		关于多头注意力和整体架构的更多细节可以在Vaswani等人中找到。 （2017年）。

![image-20200516145434303](/Users/suntian/Library/Application Support/typora-user-images/image-20200516145434303.png)

## 2.1计算性能

​		如表1所示，自注意层用恒定数量的顺序执行操作连接所有位置，而循环层需要O(n)个顺序操作。 就计算复杂度而言，当序列长度n小于表示维数d时，自注意层比递归层更快，机器翻译中最先进的模型所使用的句子表示最常出现这种情况，如word-piece（Wu et al.，2016）和byte-pair（Sennrich et al.，2015）表示。

​		核宽度k<n的单个卷积层并不连接所有对的输入和输出位置。 这样做在连续核的情况下需要O（n/k）卷积层的堆栈，或者在膨胀卷积的情况下需要O（log k(n)）（Kalchbrenner等人，2016），从而增加网络中任意两个位置之间的最长路径的长度。 卷积层通常比循环层贵一倍。 然而，可分离卷积（Chollet，2016）将复杂度大大降低到O(k·n·d+n·d2)。 然而，即使在k=n的情况下，可分离卷积的复杂度也等于一个自注意层和一个点前馈层的组合，这是我们在模型中采用的方法。

​		自我关注也能产生更多可解释的模型。 在Tensor2Tensor中，我们可以从我们的模型中可视化每个层和头部的注意力分布。 仔细观察它们，我们会发现这些模型学习执行不同的任务，许多模型表现出与句子的句法和语义结构相关的行为。

## 2.2机器翻译

​		我们对模型进行了WMT翻译任务的训练。

​		在WMT 2014英德语翻译任务中，big transformer模型（表2中的transformer（big））比先前报告的最佳模型（包括ensembles）的性能高出2.0以上BLEU，从而建立了28.4的新的最先进BLEU得分。 在8个P100 GPU上进行了3.5天的培训。 甚至我们的基本模型也超过了所有以前发布的模型和集合，只是任何竞争模型训练成本的一小部分。

​		在WMT 2014英法翻译任务中，我们的大模型获得了41.8分的BLEU成绩，超过了之前发布的所有单一模型，而训练成本不到之前最先进模型的1/4。

​		对于基本模型，我们使用了通过对最后5个检查点进行平均得到的单个模型，这些检查点以10分钟的间隔编写。 对于大模型，我们平均了最后20个关卡。 我们使用波束搜索，波束大小为4，长度惩罚=0.6（Wu et al.，2016）。 这些超参数是在开发装置上进行实验后选择的。 我们将推理时的最大输出长度设置为输入长度+50，但在可能的情况下提前终止（Wu等人，2016）。

# 3 Tensor2Tensor

​		Tensor2Tensor(T2T)是一个深度学习模型和数据集库，旨在使深度学习研究更快，更易访问。 T2T始终使用TensorFlow（Abadi et al.，2016），并且非常注重性能和可用性。 通过使用TensorFlow和各种T2T规范抽象，研究人员可以在本地和云端的CPU，GPU（单个或多个）和TPU上训练模型，通常无需或只需最少的设备规范代码或配置。

​		开发开始集中于神经机器翻译，因此Tensor2Tensor包含了许多最成功的NMT模型和标准数据集。 此后，它添加了对其他任务类型以及跨多种媒体（文本，图像，视频，音频）的支持。 模型和数据集的数量都有了显著的增长。

​		跨模型和问题的使用是标准化的，这使得在多个问题上尝试一个新模型或在单个问题上尝试多个模型变得很容易。 参见示例用法（附录B），了解命令标准化和数据集，模型，培训，评估和解码程序统一的一些可用性好处。

​		开发是在GitHub(http://GitHub.com/tensor flow/tensor2tensor)上公开进行的，谷歌内部和外部有许多贡献者。

# 4 系统概述

​		Tensor2Tensor中有指定训练运行的关键组件:

​		1.DataSets:Problem类封装了关于特定数据集的所有内容。 问题可以从头开始生成数据集，通常是从公共源下载数据，构建词汇表，并将编码示例写入磁盘。 问题还会产生用于训练和评估的输入管道，以及每个特征的任何必要的附加信息（例如，特征的类型，词汇量，以及能够将样本与人类和机器可读表示进行转换的编码器）。

​		2.设备配置:设备的类型，数量和位置。 TensorFlow和Tensor2Tensor目前支持单设备和多设备配置中的CPU，GPU和TPU。 Tensor2Tensor还支持同步和异步数据并行训练。

​		3.超参数:控制模型实例化和训练过程的参数（例如，隐藏层数或优化器的学习率）。 这些都在代码中指定并命名，以便于共享和复制。

​		4.模型:模型将前面的组件连接在一起，以实例化从输入到目标的参数化转换，计算损失和评估度量，并构建优化过程。

​		5.Estimator and Experiment:这些类是TensorFlow的一部分，用于实例化运行时，运行训练循环，执行基本的支持服务，如模型检查点，日志记录以及训练和评估之间的交替。

​		这些抽象使用户能够只将注意力集中在他们感兴趣的实验组件上。 希望在新问题上尝试模型的用户通常只需定义一个新问题。 希望创建或修改模型的用户只需创建一个模型或编辑超参数即可。 其他组件保持不受影响，不受干扰，可供使用，所有这些都减少了精神负荷，并允许用户更快地在规模上迭代他们的想法。

​		附录A包含代码概要，附录B包含示例用法。

# 5 研究组件库

​		Tensor2Tensor为研究思路提供了一个工具，可以快速试用和分享。 被证明非常有用的组件可以被提交到更广泛使用的库中，比如TensorFlow，它包含许多标准层，优化器和其他更高级的组件。

​		Tensor2Tensor支持库使用和脚本使用，以便用户可以在自己的模型或系统中重用特定组件。 例如，多名研究人员正在继续研究基于注意力的转换模型的扩展和变体，而注意力构建模块的可用性使这一工作得以实现。

一些例子:

​		•图像变压器（Parmar等人，2018年）将变压器模型扩展到图像。 它在很大程度上依赖于Tensor2Tensor中的许多注意力构建块，并添加了许多自己的注意力构建块。

​		•tf.contrib.layers.rev模块，实现Gomez等人提出的可逆层内存有效模块。 （2017年），首次在Tensor2Tensor实施和行使。

​		•Adafactor优化器（待发布）显著降低了二阶矩估计的内存需求，由Tensor2Tensor开发，并在各种模型和问题上试用。

​		•tf.contrib.data.bucket by sequence length支持在新的tf.data.DataSet输入管道API中高效处理GPU上的序列输入。 它是在tensor2tensor中首次实现和运用的。

# 6 再现性和可持续发展

​		由于模型训练的费用和随机性，在保持模型质量的同时继续开发机器学习代码库是一项困难的任务。 冻结一个代码库以保持一定的配置，或者转移到一个仅追加的进程，会带来巨大的可用性和开发成本。

​		我们试图通过三个机制来减轻正在进行的开发对历史再现性的影响:

​		1.代码中命名和版本化的超参数集

​		2.端到端回归测试，针对重要的模型-问题对定期运行，并验证实现了某些质量度量。

​		3.在多个级别（Python，numpy和TensorFlow）上设置随机种子以减轻随机性的影响（尽管在多线程，分布式，落点系统中这实际上是不可能完全实现的）。

​		如果需要，因为代码在GitHub上处于版本控制之下(http://GitHub.com/tensor flow/tensor2tensor)，我们总是可以恢复产生某些实验结果的确切代码。

# A Tensor2Tensor Code Outline

-   Create HParams
-   Create RunConfig specifying devices
    -   Create and include the Parallelism object in the RunConfig which enables data-parallel duplication of the model on multiple devices (for example, for multi-GPU synchronous training).
-   Create Experiment, including training and evaluation hooks which control support services like logging and checkpointing
-   Create Estimator encapsulating the model function
    -   T2TModel.estimator model fn ∗ model(features)
        -   model.bottom: This uses feature type information from the Problem to transform the input features into a form consumable by the model body (for example, embedding integer token ids into a dense ﬂoat space).
        -   model.body: The core of the model.
        -   model.top: Transforming the output of the model body into the target space using information from the Problem
        -   model.loss 
    -   ∗ When training: model.optimize 
    -   ∗ When evaluating: create evaluation metrics
-   Create input functions
    -   Problem.input fn: produce an input pipeline for a given mode.tf.data.Dataset API.
        -   Problem.dataset which creates a stream of individual examples 
        -   ∗ Pad and batch the examples into a form ready for efﬁcient processing
-   Run the Experiment
    -   estimator.train
        -   train op = model fn(input fn(mode=TRAIN)) 
        -   ∗ Run the train op for the number of training steps speciﬁed
    -   estimator.evaluate
        -   ∗ metrics = model fn(input fn(mode=EVAL)) 
        -   ∗ Accumulate the metrics across the number of evaluation steps speciﬁed

# B Example Usage

​		Tensor2Tensor usage is standardized across problems and models. Below you’ll ﬁnd a set of commands that generates a dataset, trains and evaluates a model, and produces decodes from that trained model. Experiments can typically be reproduced with the (problem, model, hyperparameter set) triple.

​		The following train the attention-based Transformer model on WMT data translating from English to German:

```python
pip install tensor2tensor
PROBLEM=translate_ende_wmt32k MODEL=transformer HPARAMS=transformer_base
# Generate data t2t-datagen \
--problem=$PROBLEM \
--data_dir=$DATA_DIR \
--tmp_dir=$TMP_DIR
# Train and evaluate t2t-trainer \
--problems=$PROBLEM \
--model=$MODEL \
--hparams_set=$HPARAMS \
--data_dir=$DATA_DIR \
--output_dir=$OUTPUT_DIR \
--train_steps=250000
# Translate lines from a file t2t-decoder \
--data_dir=$DATA_DIR \
--problems=$PROBLEM \
--model=$MODEL \
--hparams_set=$HPARAMS \
--output_dir=$OUTPUT_DIR \
--decode_from_file=$DECODE_FILE \
--decode_to_file=translation.en
```

