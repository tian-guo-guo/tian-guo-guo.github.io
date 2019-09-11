---
layout:     post                    # 使用的布局（不需要改）
title:      Learned in Translation :Contextualized Word Vectors中英文翻译             # 标题 
subtitle:   迁移学习在MT的应用 #副标题
date:       2019-09-11              # 时间
author:     甜果果                      # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签 
    - 学习日记
    - paper
    - nlp
    - 机器翻译
    - 迁移学习
---

### Learned in Translation: Contextualized Word Vectors

### Abstract
Computer vision has benefited from initializing multiple deep layers with weights pretrained on large supervised training sets like ImageNet. Natural language processing (NLP) typically sees initialization of only the lowest layer of deep models with pretrained word vectors. In this paper, we use a deep LSTM encoder from an attentional sequence-to-sequence model trained for machine translation (MT) to contextualize word vectors. We show that adding these context vectors (CoVe) improves performance over using only unsupervised word and character vectors on a wide variety of common NLP tasks: sentiment analysis (SST, IMDb), question classification (TREC), entailment (SNLI), and question answering (SQuAD). For fine-grained sentiment analysis and entailment, CoVe improves performance of our baseline models to the state of the art.

计算机视觉已经受益于初始化多个深层，权重预先训练在大型有监督的训练集上，如ImageNet。 自然语言处理(NLP)通常只看到使用预先训练的字向量初始化最下层的深度模型。 在本文中，我们使用一个深度LSTM编码器从一个注意序列到序列模型训练机器翻译(MT)来上下文化单词向量。 我们发现，在情感分析(SST、IMDB)、问题分类(TREC)、蕴涵(SNLI)和问答（Squad）等多种常见的自然语言处理任务中，添加这些上下文向量（CoVE）比仅使用无监督的单词和字符向量提高了性能。 对于细粒度的情感分析和蕴涵，COVE将我们的基线模型的性能提高到了最先进的水平。 

### 1 Introduction
Significant gains have been made through transfer and multi-task learning between synergistic tasks. In many cases, these synergies can be exploited by architectures that rely on similar components. In computer vision, convolutional neural networks (CNNs) pretrained on ImageNet [Krizhevsky et al., 2012, Deng et al., 2009] have become the de facto initialization for more complex and deeper models. This initialization improves accuracy on other related tasks such as visual question answering [Xiong et al., 2016] or image captioning [Lu et al., 2016, Socher et al., 2014]. 

协同任务之间的迁移和多任务学习取得了显著的成效。 在许多情况下，依赖于类似组件的体系结构可以利用这些协同作用。 在计算机视觉中，在ImageNet上预训练的卷积神经网络[Krizhevsky等人，2012，Deng等人，2009]已经成为更复杂和更深层次模型的事实上的初始化。 这种初始化提高了其他相关任务的准确性，例如视觉问题回答[Xiong等人，2016]或图像字幕[Lu等人，2016，Socher等人，2014]。

In NLP, distributed representations pretrained with models like Word2Vec [Mikolov et al., 2013] and GloVe [Pennington et al., 2014] have become common initializations for the word vectors of deep learning models. Transferring information from large amounts of unlabeled training data in the form of word vectors has shown to improve performance over random word vector initialization on a variety of downstream tasks, e.g. part-of-speech tagging [Collobert et al., 2011], named entity recognition [Pennington et al., 2014], and question answering [Xiong et al., 2017]; however, words rarely appear in isolation. The ability to share a common representation of words in the context of sentences that include them could further improve transfer learning in NLP.

在NLP中，使用诸如Word2Vec[Mikolov等人，2013]和Glove[Pennington等人，2014]等模型预先训练的分布式表示已经成为深度学习模型的单词向量的常见初始化。 以词向量的形式从大量未标记的训练数据传输信息已经表明，在各种下游任务上，例如词性标记[Collobert等人，2011]、命名实体识别[Pennington等人，2014]、以及问答[Xiong等人，2017]上，相对于随机词向量初始化，提高了性能； 然而，词语很少单独出现。 在包含单词的句子中共享单词的共同表示的能力可以进一步提高自然语言处理中的迁移学习。

Inspired by the successful transfer of CNNs trained on ImageNet to other tasks in computer vision, we focus on training an encoder for a large NLP task and transferring that encoder to other tasks in NLP. Machine translation (MT) requires a model to encode words in context so as to decode them into another language, and attentional sequence-to-sequence models for MT often contain an LSTM-based encoder, which is a common component in other NLP models. We hypothesize that MT data in general holds potential comparable to that of ImageNet as a cornerstone for reusable models. This makes an MT-LSTM pairing in NLP a natural candidate for mirroring the ImageNet-CNN pairing of computer vision.

在ImageNet上训练的CNN成功地转移到计算机视觉中的其他任务的启发下，我们专注于训练用于大型NLP任务的编码器，并将该编码器转移到NLP中的其他任务。 机器翻译（Machine Translation，MT）需要一个模型对上下文中的单词进行编码，以便将它们译成另一种语言，而MT的注意序列到序列模型通常包含一个基于LSTM的编码器，这是其他NLP模型中常见的组件。 我们假设MT数据通常具有与ImageNet相当的潜力，作为可重用模型的基石。 这使得NLP中的MT-LSTM配对成为计算机视觉的ImageNet-CNN配对的自然候选。 

As depicted in Figure 1, we begin by training LSTM encoders on several machine translation datasets, and we show that these encoders can be used to improve performance of models trained for other tasks in NLP. In order to test the transferability of these encoders, we develop a common architecture for a variety of classification tasks, and we modify the Dynamic Coattention Network for question answering [Xiong et al., 2017]. We append the outputs of the MT-LSTMs, which we call context vectors (CoVe), to the word vectors typically used as inputs to these models. This approach improved the performance of models for downstream tasks over that of baseline models using pretrained word vectors alone. For the Stanford Sentiment Treebank (SST) and the Stanford Natural Language Inference Corpus (SNLI), CoVe pushes performance of our baseline model to the state of the art.

如图1所示，我们首先在几个机器翻译数据集上训练LSTM编码器，并且我们展示了这些编码器可以用来提高NLP中为其他任务训练的模型的性能。 为了测试这些编码器的可转移性，我们为各种分类任务开发了一个通用架构，并修改了用于问答的动态协同注意网络[Xiong等人，2017]。 我们将MT-LSTM的输出（我们称之为上下文向量（CoVE））附加到通常用作这些模型输入的字向量。 与仅使用预先训练的字向量的基线模型相比，该方法提高了下游任务的模型的性能。 对于斯坦福情感树库(SST)和斯坦福自然语言推理语料库(SNLI)，COVE将我们的基线模型的性能提升到了最先进的水平。

![190911-table1.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-table1.png)

实验表明，用于训练MT-LSTM的训练数据量与下游任务的性能正相关。 这是依赖MT的另一个优点，因为MT的数据比大多数其他受监督的NLP任务的数据更丰富，这表明更高质量的MT-LSTM携带更多有用的信息。 这加强了机器翻译是进一步研究具有更强自然语言理解能力的模型的一个很好的候选任务的观点。 

### 2 Related Work
**Transfer Learning.**
Transfer learning, or domain adaptation, has been applied in a variety of areas where researchers identified synergistic relationships between independently collected datasets. Saenko et al. [2010] adapt object recognition models developed for one visual domain to new imaging conditions by learning a transformation that minimizes domain-induced changes in the feature distribution. Zhu et al. [2011] use matrix factorization to incorporate textual information into tagged images to enhance image classification. In natural language processing (NLP), Collobert et al. [2011] leverage representations learned from unsupervised learning to improve performance on supervised tasks like named entity recognition, part-of-speech tagging, and chunking. Recent work in NLP has continued in this direction by using pretrained word representations to improve models for entailment [Bowman et al., 2014], sentiment analysis [Socher et al., 2013], summarization [Nallapati et al., 2016], and question answering [Seo et al., 2017, Xiong et al., 2017]. Ramachandran et al. [2016] propose initializing sequence-to-sequence models with pretrained language models and fine-tuning for a specific task. Kiros et al. [2015] propose an unsupervised method for training an encoder that outputs sentence vectors that are predictive of surrounding sentences. We also propose a method of transferring higher-level representations than word vectors, but we use a supervised method to train our sentence encoder and show that it improves models for text classification and question answering without fine-tuning.

迁移学习，或称领域适应，已经被应用于研究人员发现独立收集的数据集之间协同关系的各个领域。 Saenko等人。 [2010]通过学习最小化域引起的特征分布变化的变换，使为一个视觉域开发的对象识别模型适应新的成像条件。 朱等。 [2011]使用矩阵分解将文本信息合并到标记图像中，以增强图像分类。 在自然语言处理(NLP)中，Collobert等人。 [2011]利用从无监督学习中学习到的表示来提高在有监督任务上的性能，如命名实体识别、词性标记和分块。 NLP最近在这方面继续开展工作，使用预先训练的词表示改进蕴含模型[Bowman等人，2014]、情感分析[Socher等人，2013]、总结[Nallapati等人，2016]和问答[Seo等人，2017，熊等人，2017]。 Ramachandran等人。 [2016]建议使用预先训练的语言模型初始化序列到序列模型，并针对特定任务进行微调。 Kiros等人。 [2015]提出了一种用于训练编码器的无监督方法，该编码器输出预测周围句子的句子向量。 我们还提出了一种比词向量更高级的表示方法，但是我们使用了一种有监督的方法来训练我们的句子编码器，并且表明它改进了文本分类和问题回答的模型，而不需要微调.

**Neural Machine Translation.**
Our source domain of transfer learning is machine translation, a task that has seen marked improvements in recent years with the advance of neural machine translation (NMT) models. Sutskever et al. [2014] investigate sequence-to-sequence models that consist of a neural network encoder and decoder for machine translation. Bahdanau et al. [2015] propose the augmenting sequence to sequence models with an attention mechanism that gives the decoder access to the encoder representations of the input sequence at each step of sequence generation. Luong et al. [2015] further study the effectiveness of various attention mechanisms with respect to machine translation. Attention mechanisms have also been successfully applied to NLP tasks like entailment [Conneau et al., 2017], summarization [Nallapati et al., 2016], question answering [Seo et al., 2017, Xiong et al., 2017, Min et al., 2017], and semantic parsing [Dong and Lapata, 2016]. We show that attentional encoders trained for NMT transfer well to other NLP tasks.

迁移学习的源领域是机器翻译，近年来随着神经机器翻译(NMT)模型的发展，机器翻译得到了显著的改进。 Sutskever等人 [2014]研究由用于机器翻译的神经网络编码器和解码器组成的序列到序列模型。 Bahdanau等人 [2015]提出了序列模型的增强序列，注意机制使解码器能够在序列生成的每个步骤访问输入序列的编码器表示。 Luong等人。 [2015]进一步研究机器翻译方面各种注意机制的有效性。 注意机制也已成功地应用于NLP任务，如蕴涵[Conneau等人，2017]、总结[Nallapati等人，2016]、问答[Seo等人，2017、Xiong等人，2017、Min等人，2017]和语义分析[Dong和Lapata，2016]。 实验结果表明，受过NMT训练的注意编码器能够很好地转移到其他NLP任务中。

**Transfer Learning and Machine Translation.**
Machine translation is a suitable source domain for transfer learning because the task, by nature, requires the model to faithfully reproduce a sentence in the target language without losing information in the source language sentence. Moreover, there is an abundance of machine translation data that can be used for transfer learning. Hill et al. [2016] study the effect of transferring from a variety of source domains to the semantic similarity tasks in Agirre et al. [2014]. Hill et al. [2017] further demonstrate that fixed-length representations obtained from NMT encoders outperform those obtained from monolingual (e.g. language modeling) encoders on semantic similarity tasks. Unlike previous work, we do not transfer from fixed length representations produced by NMT encoders. Instead, we transfer representations for each token in the input sequence. Our approach makes the transfer of the trained encoder more directly compatible with subsequent LSTMs, attention mechanisms, and, in general, layers that expect input sequences. This additionally facilitates the transfer of sequential dependencies between encoder states.

机器翻译是一个适合于迁移学习的源域，因为任务本质上要求模型忠实地再现目标语言中的句子，而不丢失源语言句子中的信息。 此外，还有大量的机器翻译数据可用于迁移学习。 希尔等人。 [2016]研究了Agirre等人提出的语义相似任务从多种源域迁移到语义相似任务的效果。 [2014]。 希尔等人。 [2017]进一步证明从NMT编码器获得的固定长度表示在语义相似性任务上优于从单语言（例如，语言建模）编码器获得的固定长度表示。 与先前的工作不同，我们不从由NMT编码器产生的固定长度表示转移。 相反，我们传输输入序列中每个令牌的表示形式。 我们的方法使得经过训练的编码器的传输更直接地与后续的LSTM、注意机制以及通常期望输入序列的层兼容。 这还有助于在编码器状态之间传输顺序依赖性。 

**Transfer Learning in Computer Vision.**
Since the success of CNNs on the ImageNet challenge, a number of approaches to computer vision tasks have relied on pretrained CNNs as off-the-shelf feature extractors. Girshick et al. [2014] show that using a pretrained CNN to extract features from region proposals improves object detection and semantic segmentation models. Qi et al. [2016] propose a CNN-based object tracking framework, which uses hierarchical features from a pretrained CNN (VGG-19 by Simonyan and Zisserman [2014]). For image captioning, Lu et al. [2016] train a visual sentinel with a pretrained CNN and fine-tune the model with a smaller learning rate. For VQA, Fukui et al. [2016] propose to combine text representations with visual representations extracted by a pretrained residual network [He et al., 2016]. Although model transfer has seen widespread success in computer vision, transfer learning beyond pretrained word vectors is far less pervasive in NLP.

自从CNN在ImageNet挑战中取得成功以来，许多计算机视觉任务的方法都依赖于预先训练的CNN作为现成的特征提取器。 Girshick等人。 [2014]表明，使用预先训练的CNN从区域提案中提取特征改进了对象检测和语义分割模型。 齐等人。 [2016]提出一个基于CNN的目标跟踪框架，该框架使用预先训练的CNN的分层特征(VGG-19，Simonyan和Zisserman[2014])。 对于图像字幕，Lu等人。 [2016]用预先训练的CNN训练视觉哨兵，并以较小的学习速率微调模型。 对于VQA，Fukui等人。 [2016]提议将文本表示与通过预先训练的残差网络提取的视觉表示相结合[He等人，2016年]。 尽管模型迁移在计算机视觉领域取得了广泛的成功，但在自然语言处理领域，超出预先训练好的词向量的迁移学习并不普遍。 

**3 Machine Translation Model**
We begin by training an attentional sequence-to-sequence model for English-to-German translation based on Klein et al. [2017] with the goal of transferring the encoder to other tasks. 

我们首先在Klein等人的基础上训练了一个注意序列到序列的英德翻译模型。 [2017]目标是将编码器转移到其他任务。

For training, we are given a sequence of words in the source language wx = [wx1,...,wxn] and a sequence of words in the target language wz = [wz1,...,wzm]. Let GloVe(wx) be a sequence of GloVe vectors corresponding to the words in wx, and let z be a sequence of randomly initialized word vectors corresponding to the words in wz.

为了训练，我们得到源语言Wx=[Wx1，...，Wxn]中的单词序列和目标语言Wz=[Wz1，...，Wzm]中的单词序列。 设glove(wx)是对应于wx中的单词的glove向量序列，z是对应于wz中的单词的随机初始化的单词向量序列。

We feed GloVe(wx) to a standard, two-layer, bidirectional, long short-term memory network 1 [Graves and Schmidhuber, 2005] that we refer to as an MT-LSTM to indicate that it is this same two-layer BiLSTM that we later transfer as a pretrained encoder. The MT-LSTM is used to compute a sequence of hidden states

我们将手套(WX)馈送到标准的、两层的、双向的、长期的短期记忆网络1[Graves和Schmidhuber，2005]，我们将其称为MT-LSTM，以指示我们稍后作为预训练编码器传输的正是相同的两层BILSTM。 MT-LSTM用于计算隐藏状态序列 
![190911-equation1.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation1.png)
For machine translation, the MT-LSTM supplies the context for an attentional decoder that produces a distribution over output words p(ˆwzt |H,wz1,...,wzt−1) at each time-step. 

对于机器翻译，MT-LSTM提供用于注意解码器的上下文，该注意解码器在每个时间步产生输出字P(WZT H，WZ1，...，WZT−1)上的分布。

At time-step t, the decoder first uses a two-layer, unidirectional LSTM to produce a hidden state hdec t based on the previous target embedding zt−1 and a context-adjusted hidden state ˜ht−1:

在时间步骤t，解码器首先使用两层单向LSTM来基于先前的目标嵌入ZT-1和上下文调整隐藏状态～HT-1:
![190911-equation2.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation2.png)
The decoder then computes a vector of attention weights α representing the relevance of each encoding time-step to the current decoder state.

解码器然后计算表示每个编码时间步长与当前解码器状态的相关性的关注权重α的向量。 
![190911-equation3.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation3.png)

where H refers to the elements of h stacked along the time dimension. 

其中H表示沿时间维堆叠的H的元素。

The decoder then uses these weights as coefficients in an attentional sum that is concatenated with the decoder state and passed through a tanh layer to form the context-adjusted hidden state ˜h: 

然后，解码器使用这些权重作为关注和中的系数，该关注和与解码器状态级联并且经过Tanh层以形成上下文调整的隐藏状态～H
![190911-equation4.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation4.png)

The distribution over output words is generated by a final transformation of the context-adjusted. hidden state: p( ˆwz
t |X, wz 1, . . . , wz t−1) = softmax Wout˜ht + bout)  
输出字上的分布由上下文调整的最终转换生成。 隐藏状态:P（WZ 
T x，Wz 1，。 。。。，WZ t−1)=SOFTMAX WOUT~HT+BOUT？？)

### 4 Context Vectors (CoVe)
We transfer what is learned by the MT-LSTM to downstream tasks by treating the outputs of the MT-LSTM as context vectors. If w is a sequence of words and GloVe(w) the corresponding sequence of word vectors produced by the GloVe model, then  
通过将MT-LSTM的输出作为上下文向量，我们将MT-LSTM所学到的知识转移到下游任务。 如果w是单词序列，并且glove(w)是由glove模型产生的单词向量的相应序列，然后
![190911-equation5.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation5.png)

is the sequence of context vectors produced by the MT-LSTM. For classification and question answering, for an input sequence w, we concatenate each vector in GloVe(w) with its corresponding vector in CoVe(w)  
是MT-LSTM生成的上下文向量序列。 为了分类和回答问题，对于输入序列W，我们将手套(W)中的每个向量与其在凹槽(W)中的相应向量连接起来。   
![190911-equation6.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation6.png)
as depicted in Figure 1b.  
如图1b所示。
![190911-figure1.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-figure1.png)

### 5 Classification with CoVe
We now describe a general biattentive classification network (BCN) we use to test how well CoVe transfer to other tasks. This model, shown in Figure 2, is designed to handle both single-sentence and two-sentence classification tasks. In the case of single-sentence tasks, the input sequence is duplicated to form two sequences, so we will assume two input sequences for the rest of this section.
我们现在描述了一个通用的偏注意分类网络(BCN)，我们使用它来测试Cove到其他任务的迁移情况。 如图2所示，该模型设计用于处理单句和双句分类任务。 在单句子任务的情况下，输入序列被重复以形成两个序列，因此我们将在本节的其余部分假设两个输入序列。
![190911-figure2.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-figure2.png)

Input sequences wx and wy are converted to se- quences of vectors, ˜wx and ˜wy, as described in Eq. 6 before being fed to the task-specific portion of the model (Figure 1b).   
输入序列wx和wy被转换成向量序列，～wx和～wy，如等式中所述。 6在被馈送到模型的任务特定部分之前（图1B)。

A function f applies a feedforward network with ReLU activation [Nair and Hinton, 2010] to each element of ˜wx and ˜wy, and a bidirectional LSTM processes the resulting sequences to obtain task spe- cific representations,  
函数F将具有RELU激活的前馈网络[Nair和辛顿，2010]应用于～Wx和～Wy的每个元素，并且双向LSTM处理所得序列以获得任务特定表示，
![190911-equation7_8.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation7_8.png)

These sequences are each stacked along the time axis to get matrices X and Y.  
这些序列各自沿着时间轴堆叠以得到矩阵X和Y。

In order to compute representations that are interde- pendent, we use a biattention mechanism [Seo et al., 2017, Xiong et al., 2017]. The biattention first com- putes an affinity matrix A = XY?. It then extracts attention weights with column-wise normalization:  
为了计算悬而未决的表示，我们使用双注意机制[Seo等人，2017，Xiong等人，2017]。 Biattention首先计算一个亲和矩阵A=XY？。 然后，使用按列归一化提取注意权重:
![190911-equation9.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation9.png)

which amounts to a novel form of self-attention when x = y. Next, it uses context summaries to condition each sequence on the other.  
这相当于当x=y时的一种新的自我注意形式。 接下来，它使用上下文摘要来调节每个序列。 
![190911-equation10.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation10.png)

We integrate the conditioning information into our representations for each sequence with two separate one-layer, bidirectional LSTMs that operate on the concatenation of the original representations (to ensure no information is lost in conditioning), their differences from the context summaries (to explicitly capture the difference from the original signals), and the element-wise products between originals and context summaries (to amplify or dampen the original signals).  
我们使用两个独立的单层双向LSTM将条件信息集成到每个序列的表示中，这两个LSTM分别处理原始表示的级联（以确保条件中没有信息丢失）、它们与上下文摘要的差异（以显式捕获原始信号的差异）以及原始表示与上下文摘要之间的元素积（以放大或抑制原始信号）。 
![190911-equation11_12.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation11_12.png)

The outputs of the bidirectional LSTMs are aggregated by pooling along the time dimension. Max and mean pooling have been used in other models to extract features, but we have found that adding both min pooling and self-attentive pooling can aid in some tasks. Each captures a different perspective on the conditioned sequences.  
双向LSTM的输出通过沿时间维的池进行聚合。 Max和Mean Pooling在其他模型中被用来提取特征，但是我们发现添加Min Pooling和Self-Attentive Pooling都可以帮助完成一些任务。 每种方法都捕获条件序列的不同透视图。

The self-attentive pooling computes weights for each time step of the sequence  
自注意池为序列的每个时间步骤计算权重
![190911-equation13.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation13.png)

and uses these weights to get weighted summations of each sequence:  
并使用这些权重得到每个序列的加权和:
![190911-equation14.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation14.png)

The pooled representations are combined to get one joined representation for all inputs.  
合并的表示被组合以获得所有输入的一个联接表示。
![190911-equation15_16.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-equation15_16.png)

We feed this joined representation through a three-layer, batch-normalized [Ioffe and Szegedy, 2015] maxout network [Goodfellow et al., 2013] to produce a probability distribution over possible classes.  
我们通过三层、批归一化的[IOFFE和Szegedy，2015]Maxout网络[Goodfellow等人，2013]来馈送这个连接的表示，以在可能的类上产生概率分布。

### 6 Question Answering with CoVe
For question answering, we obtain sequences x and y just as we do in Eq. 7 and Eq. 8 for classification, except that the function f is replaced with a function g that uses a tanh activation instead of a ReLU activation. In this case, one of the sequences is the document and the other the question in the question-document pair. These sequences are then fed through the coattention and dynamic decoder implemented as in the original Dynamic Coattention Network (DCN) [Xiong et al., 2016].
对于问题的回答，我们得到序列X和Y，就像我们在方程中所做的那样。 7和等式。 8用于分类，除了函数f被使用tanh激活而不是relu激活的函数g代替。 在这种情况下，一个序列是文档，另一个序列是问题-文档对中的问题。 然后，这些序列通过与原始动态共同注意网络(DCN)中实现的共同注意和动态解码器馈送[Xiong等人，2016]。 

### 7 Datasets
**Machine Translation.** We use three different English-German machine translation datasets to train three separate MT-LSTMs. Each is tokenized using the Moses Toolkit [Koehn et al., 2007].  
我们使用三个不同的英德机器翻译数据集来训练三个独立的MT-LSTM。 每个都使用Moses工具包进行标记[Koehn等人，2007年]。

Our smallest MT dataset comes from the WMT 2016 multi-modal translation shared task [Specia et al., 2016]. The training set consists of 30,000 sentence pairs that briefly describe Flickr captions and is often referred to as Multi30k. Due to the nature of image captions, this dataset contains sentences that are, on average, shorter and simpler than those from larger counterparts.  
我们最小的MT数据集来自WMT2016多模态翻译共享任务[Specia et al.，2016]。 训练集包含30,000个句子对，这些句子对简要描述了Flickr字幕，通常称为multi30k。 由于图像标题的性质，此数据集包含的句子平均比较大对应的句子更短和更简单。 

Our medium-sized MT dataset is the 2016 version of the machine translation task prepared for the International Workshop on Spoken Language Translation [Cettolo et al., 2015]. The training set consists of 209,772 sentence pairs from transcribed TED presentations that cover a wide variety of topics with more conversational language than in the other two machine translation datasets.  
我们的中型MT数据集是为国际口语翻译研讨会[Cettolo等人，2015年]准备的机器翻译任务的2016年版本。 培训集由209,772对抄录的TED演讲句子组成，这些句子比另外两个机器翻译数据集包含更多会话语言的广泛主题。 

Our largest MT dataset comes from the news translation shared task from WMT 2017. The training set consists of roughly 7 million sentence pairs that comes from web crawl data, a news and commentary corpus, European Parliament proceedings, and European Union press releases.
我们最大的MT数据集来自WMT2017的新闻翻译共享任务。 这个训练集由大约700万对句子组成，这些句子来自网络爬行数据、新闻和评论语料库、欧洲议会议事录和欧盟新闻稿。 

We refer to the three MT datasets as MT-Small, MT-Medium, and MT-Large, respectively, and we refer to context vectors from encoders trained on each in turn as CoVe-S, CoVe-M, and CoVe-L. 
我们将这三个MT数据集分别称为MT-Small、MT-Medium和MT-Large，并将来自编码器的上下文向量依次称为COVE-S、COVE-M和COVE-L。

**Sentiment Analysis.** We train our model separately on two sentiment analysis datasets: the Stanford Sentiment Treebank (SST) [Socher et al., 2013] and the IMDb dataset [Maas et al., 2011]. Both of these datasets comprise movie reviews and their sentiment. We use the binary version of each dataset as well as the five-class version of SST. For training on SST, we use all sub-trees with length greater than 3. SST-2 contains roughly 56, 400 reviews after removing “neutral” examples. SST-5 contains roughly 94, 200 reviews and does include “neutral” examples. IMDb contains 25, 000 multi-sentence reviews, which we truncate to the first 200 words. 2, 500 reviews are held out for validation.  
我们分别在两个情绪分析数据集上训练我们的模型:斯坦福情绪树库(SST)[Socher等人，2013]和IMDB数据集[Maas等人，2011]。 这两个数据集包括电影评论和他们的情感。 我们使用每个数据集的二进制版本以及SST的五类版本。 对于SST的训练，我们使用长度大于3的所有子树。 在删除“中性”示例之后，SST-2包含大约56,400个评论。 SST-5包含大约94,200篇评论，并包含“中性”的例子。 IMDB包含25,000个多句评论，我们将其截断为前200个单词。 有2,500篇评论等待验证。

**Question Classification.** For question classification, we use the small TREC dataset [Voorhees and Tice, 1999] dataset of open-domain, fact-based questions divided into broad semantic categories. We experiment with both the six-class and fifty-class versions of TREC, which which refer to as TREC-6 and TREC-50, respectively. We hold out 452 examples for validation and leave 5, 000 for training.
Entailment.  
在问题分类方面，我们使用了小型的TREC数据集[Voorhees and Tice，1999]，它是一个开放领域的基于事实的问题集，被划分为广泛的语义类别。 我们对TREC的六级和五十级版本进行了实验，这两个版本分别称为TREC-6和TREC-50。 我们给出了452个验证示例，并留下5000个用于培训。 

**Entailment.** For entailment, we use the Stanford Natural Language Inference Corpus (SNLI) [Bow- man et al., 2015], which has 550,152 training, 10,000 validation, and 10,000 testing examples. Each example consists of a premise, a hypothesis, and a label specifying whether the premise entails, contradicts, or is neutral with respect to the hypothesis.  
对于蕴涵，我们使用斯坦福自然语言推理语料库(SNLI)[Bow-man等人，2015年]，该语料库有550,152个训练、10,000个验证和10,000个测试示例。 每个示例都包含一个前提、一个假设和一个标签，该标签指定该前提对于该假设是包含、矛盾还是中立的。

**Question Answering.** The Stanford Question Answering Dataset (SQuAD) [Rajpurkar et al., 2016] is a large-scale question answering dataset with 87,599 training examples, 10,570 development examples, and a test set that is not released to the public. Examples consist of question-answer pairs associated with a paragraph from the English Wikipedia. SQuAD examples assume that the question is answerable and that the answer is contained verbatim somewhere in the paragraph.
斯坦福问答数据集（Squad）[Rajpurkar等人，2016年]是一个大型问答数据集，包含87599个培训示例、10570个开发示例和一个未向公众发布的测试集。 示例包括与英文维基百科中的段落相关联的问答对。 小组的例子假定问题是可以回答的，答案逐字地包含在段落的某处。

### 8 Experiments

### 8.1 Machine Translation
The MT-LSTM trained on MT-Small obtains an uncased, tokenized BLEU score of 38.5 on the Multi30k test set from 2016. The model trained on MT-Medium obtains an uncased, tokenized BLEU score of 25.54 on the IWSLT test set from 2014. The MT-LSTM trained on MT-Large obtains an uncased, tokenized BLEU score of 28.96 on the WMT 2016 test set. These results represent strong baseline machine translation models for their respective datasets. Note that, while the smallest dataset has the highest BLEU score, it is also a much simpler dataset with a restricted domain.  
在MT-Small上训练的MT-LSTM从2016年起在Multi30K测试集上获得38.5的无约束的、标记化的BLEU分数。 在MT-medium上训练的模型从2014年起在IWSLT测试集上获得了25.54的无约束、标记的BLEU分数。 在MT-LARGE上训练的MT-LSTM在WMT2016测试集上获得28.96的无约束的、标记化的BLEU分数。 这些结果代表了各自数据集的强基线机器翻译模型。 请注意，虽然最小的数据集具有最高的BLEU得分，但它也是一个具有受限域的简单得多的数据集。 

**Training Details.** When training an MT-LSTM, we used fixed 300-dimensional word vectors. We used the CommonCrawl-840B GloVe model for English word vectors, which were completely fixed during training, so that the MT-LSTM had to learn how to use the pretrained vectors for translation. The hidden size of the LSTMs in all MT-LSTMs is 300. Because all MT-LSTMs are bidirectional, they output 600-dimensional vectors. The model was trained with stochastic gradient descent with a learning rate that began at 1 and decayed by half each epoch after the validation perplexity increased for the first time. Dropout with ratio 0.2 was applied to the inputs and outputs of all layers of the encoder and decoder.  
在训练MT-LSTM时，我们使用固定的300维词向量。 我们使用Commoncrawl-840B手套模型对英语单词向量进行训练，这些向量在训练期间完全固定，因此MT-LSTM必须学习如何使用预先训练的向量进行翻译。 所有MT-LSTM中LSTM的隐藏大小为300。 因为所有MT-LSTM都是双向的，所以它们输出600维向量。 该模型以随机梯度下降的方式进行训练，学习速率从1开始，在验证困惑度首次增加后，每一个历元衰减一半。 在编码器和解码器的所有层的输入和输出上施加比为0.2的压差。 

**8.2 Classification and Question Answering**
For classification and question answering, we explore how varying the input representations affects final performance. Table 2 contains validation performances for experiments comparing the use of GloVe, character n-grams, CoVe, and combinations of the three.  
对于分类和问答，我们探讨了输入表示形式的变化如何影响最终性能。 表2包含比较手套、字符n克、COVE和三者组合使用的实验验证性能。
![190911-table2.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-table2.png)

**Training Details.** Unsupervised vectors and MT-LSTMs remain fixed in this set of experiments. LSTMs have hidden size 300. Models were trained using Adam with α = 0.001. Dropout was applied before all feedforward layers with dropout ratio 0.1, 0.2, or 0.3. Maxout networks pool over 4 channels, reduce dimensionality by 2, 4, or 8, reduce again by 2, and project to the output dimension.  
在这组实验中，无监督向量和MT-LSTM保持不变。 LSTM具有隐藏大小300。 模型训练采用ADAM，α=0.001。 在所有前馈层之前施加压差，压差比为0.1、0.2或0.3。 MAXOUT网络通过4个通道共用，将维数降低2、4或8，再次降低2，然后投射到输出维度。

**The Benefits of CoVe.** Figure 3a shows that models that use CoVe alongside GloVe achieve higher validation performance than models that use only GloVe. Figure 3b shows that using CoVe in Eq. 6 brings larger improvements than using character n-gram embeddings [Hashimoto et al., 2016]. It also shows that altering Eq. 6 by additionally appending character n-gram embeddings can boost performance even further for some tasks. This suggests that the information provided by CoVe is complementary to both the word-level information provided by GloVe as well as the character-level information provided by character n-gram embeddings.  
图3A显示，与只使用手套的模型相比，将COVE与手套一起使用的模型实现了更高的验证性能。 图3B显示了在EQ中使用COVE。 6比使用字符N-克嵌入带来更大的改进[Hashimoto等人，2016年]。 结果还表明，情商的改变。 6通过附加字符，n-克嵌入可以进一步提高某些任务的性能。 这表明，COVE提供的信息与GLOVE提供的字级信息以及字符n字嵌入提供的字符级信息是互补的。
![190911-figure3.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-figure3.png)

**The Effects of MT Training Data.** 
We experimented with different training datasets for the MT-LSTMs to see how varying the MT train- ing data affects the benefits of using CoVe in downstream tasks. Figure 4 shows an important trend we can ex- tract from Table 2. There appears to be a positive correlation between the larger MT datasets, which contain more complex, varied language, and the improvement that using CoVe brings to downstream tasks. This is evidence for our hypothesis that MT data has potential as a large resource for transfer learning in NLP.  
我们对MT-LSTM的不同训练数据集进行了实验，以了解MT-Training数据的变化如何影响在下游任务中使用COVE的好处。 图4显示了一个重要的趋势，我们可以从表2中推断出来。 包含更复杂、更多样的语言的更大的MT数据集与使用COVE给下游任务带来的改进之间似乎存在正相关。 这证明了我们的假设，即MT数据作为NLP中迁移学习的一个巨大资源具有潜力。
![190911-figure4.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-figure4.png)

**Test Performance.** Table 4 shows the final test accuracies of our best classification models, each of which achieved the highest validation accuracy on its task using GloVe, CoVe, and character n-gram embeddings. Final test performances on SST-5 and SNLI reached a new state of the art.  
表4显示了我们最好的分类模型的最终测试精度，每个模型都使用GLOVE、COVE和字符N-图嵌入在其任务上获得了最高的验证精度。 在SST-5和SNLI上的最终测试性能达到了新的技术水平。 
![190911-table4.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-table4.png)

Table 3 shows how the validation exact match and F1 scores of our best SQuAD model compare to the scores of the most recent top models in the literature. We did not submit the SQuAD model for testing, but the addition of CoVe was enough to push the validation performance of the original DCN, which already used
character n-gram embeddings, above the validation performance of the published version of the R-NET.Test performances are tracked by the SQuAD leaderboard 2.  
表3显示了我们最佳阵容模型的验证精确匹配和F1得分与文献中最新顶级模型的得分的比较。 我们没有提交Squad模型进行测试，但是添加COVE就足以推动已经使用的原始DCN的验证性能 
字符N-gram嵌入，高于已发布版本的R-NET的验证性能。测试性能由Squad Leaderboard 2跟踪。 
![190911-table3.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-table3.png)

**Comparison to Skip-Thought Vectors.** Kiros et al. [2015] show how to encode a sentence into a single skip-thought vector that transfers well to a variety of tasks. Both skip-thought and CoVe pretrain encoders to capture information at a higher level than words. However, skip-thought encoders are trained with an unsupervised method that relies on the final output of the encoder. MT-LSTMs are trained with a supervised method that instead relies on intermediate outputs as- sociated with each input word. Additionally, the 4800 dimensional skip-thought vectors make training more unstable than using the 600 dimensional CoVe. Table 5 shows that these differences make CoVe more suitable for transfer learning in our classification experiments.  
Kiros等人。 [2015]展示了如何将句子编码为一个跳过思考的向量，该向量可以很好地传递到各种任务。 Skip-Though和Cove Pretrain编码器都可以在比单词更高的级别捕获信息。 然而，跳过思想的编码器是用依赖于编码器的最终输出的无监督方法来训练的。 MT-LSTM使用监督方法进行训练，该方法依赖于与每个输入字相关联的中间输出。 此外，4800维跳跃思维向量使得训练比使用600维凹槽更不稳定。 表5显示了这些差异使得COVE更适合于我们的分类实验中的迁移学习。 
![190911-table5.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190911-table5.png)

**9 Conclusion**
We introduce an approach for transferring knowledge from an encoder pretrained on machine translation to a variety of downstream NLP tasks. In all cases, models that used CoVe from our best, pretrained MT-LSTM performed better than baselines that used random word vector initialization, baselines that used pretrained word vectors from a GloVe model, and baselines that used word vectors from a GloVe model together with character n-gram embeddings. We hope this is a step towards the goal of building unified NLP models that rely on increasingly more general reusable weights.   
介绍了一种将机器翻译训练好的编码器的知识转移到多种下游NLP任务的方法。 在所有情况下，使用我们最好的、预先训练的MT-LSTM的COVE的模型比使用随机字向量初始化的基线、使用手套模型的预先训练的字向量的基线、以及使用手套模型的字向量和字符N-图嵌入的基线表现得更好。 我们希望这是朝着建立统一的NLP模型的目标迈出的一步，该模型依赖于越来越广泛的可重用权重。   

The PyTorch code at https://github.com/salesforce/cove includes an example of how to generate CoVe from the MT-LSTM we used in all of our best models. We hope that making our best MT-LSTM available will encourage further research into shared representations for NLP models.  
http://github.com/salesforce/cove上的Pytorch代码包含一个如何从我们在所有最佳模型中使用的MT-LSTM生成cove的示例。 我们希望，使我们最好的MT-LSTM可用将鼓励进一步研究NLP模型的共享表示。 