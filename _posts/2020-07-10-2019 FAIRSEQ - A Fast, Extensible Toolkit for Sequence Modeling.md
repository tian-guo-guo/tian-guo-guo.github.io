---
layout:     post           # 使用的布局（不需要改）
title:      2019 FAIRSEQ - A Fast, Extensible Toolkit for Sequence Modeling             # 标题 
subtitle:   2019 FAIRSEQ - A Fast, Extensible Toolkit for Sequence Modeling   #副标题
date:       2020-07-10             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 技术


---

# 2019 FAIRSEQ - A Fast, Extensible Toolkit for Sequence Modeling

![image-20200710164025979](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200720165606.png)

# Abstract

​		FAIRSEQ是一个开放源代码序列建模工具箱，它使研究人员和开发人员可以训练自定义模型以进行翻译，摘要，语言建模和其他文本生成任务。 该工具包基于PyTorch，并支持跨多个GPU和机器的分布式培训。 我们还支持现代GPU上的快速混合精度训练和推理。 演示视频可以在这里找到：https://www.youtube.com/watch?v=OtgDdWtHvto 

# 1 Introduction

​		神经序列到序列模型已成功完成各种文本生成任务，包括机器翻译，抽象文档摘要和语言建模。因此，研究人员和行业专家都可以从快速且易于扩展的序列建模工具包中受益。

​		有几个具有相似基本功能的工具包，但是它们在重点领域和目标受众方面有所不同。例如，OpenNMT（Klein et al。，2017）是一个社区构建的工具包，用多种语言编写，重点是可扩展性。 MarianNMT（Junczys-Dowmunt等人，2018）专注于性能，后端是用C ++编写的，用于快速自动区分。 OpenSeq2Seq（Kuchaiev等人，2018）提供了用于快速分布式和混合精度训练的参考实现。 Tensor2tensor（Vaswani等人，2018）和Sockeye（Hieber等人，2018）专注于生产就绪。

​		在本文中，我们介绍了FAIRSEQ，这是一种用PyTorch编写的序列建模工具包，具有快速，可扩展的特点，对研究和生产均有用。 FAIRSEQ功能：（i）跨模型和任务的通用接口，可以使用用户提供的插件进行扩展（第2节）； （ii）高效的分布式和混合精度训练，可以在具有当前硬件上数亿个句子的数据集上进行训练（第3节）； （iii）机器翻译，摘要和语言建模的最新实现和预训练模型（第4节）； （iv）利用多种支持的搜索算法优化推论，包括波束搜索，多种波束搜索（Vijayakumar等，2016）和top-k采样。 FAIRSEQ附带BSD许可证，可在GitHub上的https://github.com/pypych/fairseq获得。

# 2 Design

**可扩展性**。 FAIRSEQ可以通过五种类型的用户提供的插件进行扩展，这些插件可以尝试新想法，同时尽可能地重用现有组件。

**模型**定义了神经网络架构并封装了所有可学习的参数。模型扩展了BaseFairseqModel类，该类又扩展了torch.nn.Module。因此，任何FAIRSEQ模型都可以用作其他PyTorch代码中的独立模块。模型还可以预定义具有通用网络配置（例如，嵌入尺寸，层数等）的命名架构。我们还通过逐步预测抽象了模型与生成算法交互的方法，例如波束搜索。这将模型实现与生成算法隔离开。
**标准**在给定模型和一组数据的情况下计算损失，大致为：损失=标准（模型，批次）。由于可以完全访问模型，因此该公式使准则具有很高的表达力。例如，一个标准可以即时生成以支持序列级训练（Edunov等，2018b）或在线回译（Edunov等，2018a; Lample等，2018）。或者，在专家混合模型中，准则可以仅通过产生最低损失的专家来实施EM风格的培训和反向传播（Shen等人，2019）。

**任务**存储字典，为加载和批处理数据提供帮助，并定义训练循环。它们旨在是不可变的，并且主要是各个组件之间的接口。我们提供翻译，语言建模和分类的任务。

**优化器**基于梯度更新模型参数。我们提供了大多数PyTorch优化器的包装器和Adafactor的实现（Shazeer和Stern，2018），这是Adam的内存有效型变体。

**学习率计划程序**会在培训过程中更新学习率。我们提供了几种流行的调度程序，例如Vaswani等人的反平方根调度程序。 （2017）和基于热重启的周期性调度程序（Loshchilov and Hutter，2016）。

**重现性和前向兼容性**。 FAIRSEQ包含旨在提高可重复性和前向兼容性的功能。例如，检查点包含模型，优化器和数据加载器的完整状态，因此，如果中断训练并恢复训练，则结果是可重现的。 FAIRSEQ还提供了前向兼容性，即使用旧版工具包训练的模型将通过自动检查点升级继续在最新版本上运行。

# 3 Implementation

​		FAIRSEQ在PyTorch中实现，并提供有效的批处理，混合精度训练，多GPU以及多机训练。

​		**批处理**。批量输入和输出序列对有多种策略（Morishita等，2017）。 FAIRSEQ通过将相似长度的源序列和目标序列分组来最大程度地减少小批量中的填充。在整个训练过程中，每个小批量的内容都保持不变，但是，每个小批量都会在每个时期随机进行包装。在多于一个GPU或机器上进行训练时，每个工人的小批处理的平均句子长度可能会有所不同，从而导致更具代表性的更新。

​		**多GPU训练**。 FAIRSEQ使用NCCL2库和torch.distributed进行GPU之间的通信。在同步优化设置中训练模型，其中每个GPU都有模型的副本以处理数据的子批处理，然后在GPU之间同步梯度；所有子批次都构成一个小批量。即使子批次包含相似数量的令牌，我们仍会观察到处理时间的巨大差异。在多GPU或多计算机设置中，这会导致大多数GPU处于空闲时间，而速度较慢的工作人员正在完成他们的工作（图1（a））。 FAIRSEQ通过在工作人员之间通过向后传递重叠梯度同步并通过为每个GPU累积多个微型批处理上的梯度来减轻散乱者的影响（Ott等人，2018b）。

​		重叠梯度同步在计算网络部分的梯度时开始同步。特别是，当完成图层的梯度计算时，FAIRSEQ将结果添加到缓冲区。当缓冲区的大小达到预定阈值时，梯度将在后台线程中同步，而反向传播将照常继续（图1（b））。接下来，我们在每个GPU上累积多个子批次的梯度，这减少了工作人员之间的处理时间差异，因为在每个子批次之后都无需等待散乱者（图1（c））。这也增加了有效的批次大小，但是我们发现模型仍然可以有效地训练（Ott等人，2018b）。

​		**混合精度**。最近的GPU支持高效的半精度浮点（FP16）计算。 FAIRSEQ在训练和推理时支持全精度（FP32）和FP16。我们执行所有前向后计算以及FP16中工作人员之间的梯度同步的所有归约操作。但是，参数更新保留在FP32中以保持准确性。 FAIRSEQ实现了动态损耗定标（Micikevicius等，2018），以避免由于FP16提供的有限精度而导致激活和梯度不足的情况。这样可以将正向传递之后的损耗调整为适合FP16范围，而反向传递保持不变。在工作人员之间同步FP16梯度后，我们将其转换为FP32，恢复原始比例并更新权重。

​		**推理**。 FAIRSEQ可通过增量解码为先前模型生成的非递归模型（Gehring等，2017; Vaswani等，2017; Fan等，2018b; Wu等，2019）提供快速推断。被缓存在每个活动波束中并重新使用。由于只为每个令牌计算新的状态，因此可以在不进行高速缓存的情况下加快朴素的实现速度。对于某些模型，这需要特定于组件的缓存实现，例如Transformer体系结构中的多头注意。

​		在推论过程中，我们建立了具有可变数量示例的批处理，直到用户指定数量的令牌为止，这与训练相似。 FAIRSEQ还支持FP16中的推理，与FP32相比，FP16的解码速度提高了54％，而准确性没有损失（表1）。

![image-20200710164631884](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200720165621.png)

# 4 Applications

​		FAIRSEQ已用于许多应用程序中，例如机器翻译（Gehring等人，2017; Edunov等人，2018b，a; Chen等人，2018; Ott等人，2018a; Song等人，2018; Wu等人，2019），语言建模（Dauphin等人，2017; Baevski和Auli，2019），抽象文档摘要（Fan等人，2018a; Liu等人，2018; Narayan等人，2018） ，故事生成（Fan等人，2018b，2019），纠错（Chollampatt和Ng，2018），多语言句子嵌入（Artetxe和Schwenk，2018）和对话（Miller等人，2017; Dinan等人， 2019）。

## 4.1 Machine translation

​		我们提供了可用于机器翻译的几种流行的序列到序列模型的参考实现，包括LSTM（Luong等人，2015），卷积模型（Gehring等人，2017; Wu等人，2019）和变压器（Vaswani等人，2017）。

​		我们评估了两种语言对的“大型”变压器编码器解码器模型：WMT英语到德语（En-De）和WMT英语到法语（En-Fr）。对于En-De，我们复制了Vaswani等人的设置。 （2017）依靠WMT’16进行450万个句子对的训练，我们在newstest13上进行验证，并在newstest14上进行测试。 32K词汇表基于联合的源和目标字节对编码（BPE; Sennrich et al.2016）。对于En-Fr，我们在WMT’14上进行了训练，并借鉴了Gehring等人的设置。 （2017）中包含3600万个训练句子对。我们使用newstest12 + 13进行验证，并使用newstest14进行测试。 40K词汇表基于共同的源BPE和目标BPE。

​		我们使用多漏洞（Hoang et al。，2006）和区分大小写的BLEU使用SacreBLEU 1（Post，2018）测量区分大小写的令牌化BLEU。按照Vaswani等人的方法，所有结果都使用波束搜索，波束宽度为4，长度损失为0.6。 2017年。FAIRSEQ结果总结在表2中。我们报道了BLEU得分高于Vaswani等人。 （2017）通过更大批量的培训和更高的学习率进行培训（Ott等人，2018b）。

## 4.2 Language modelling

​		FAIRSEQ支持使用门控卷积模型（Dauphin等，2017）和Transformer模型（Vaswani等，2017）进行语言建模。 可以使用各种输入和输出表示来训练模型，例如标准令牌嵌入，卷积字符嵌入（Kim等人，2016），自适应softmax（Grave等人，2017）和自适应输入（Baevski和Auli， 2019）。 我们还提供了教程和经过预训练的模型，它们可以复制Dauphin等人的结果。 （2017）以及Baevski和Auli（2019）上的WikiText-103和十亿字数据集。

​		根据Baevski和Auli（2019），我们评估了两个仅使用解码器网络和自适应输入嵌入的Transformer语言模型。 第一个模型有16个块，内部尺寸为4K，嵌入尺寸为1K。 WikiText-103上的结果在表3中。第二个模型有24个块，内部尺寸为8K，嵌入尺寸为1.5K。 表4中的十亿字基准测试结果。

## 4.3 Abstractive document summarization

​		接下来，我们尝试抽象文档摘要，其中我们使用基本的Transformer对输入文档进行编码，然后使用解码器网络生成摘要。我们使用新闻文章的CNN-Dailymail数据集（Hermann等人，2015; Nallapati等人，2016）搭配多句摘要。我们对全文版本进行评估，没有实体匿名化（参见et al。，2017）;我们将文章截断为400个令牌（请参见et al。，2017）。我们采用Fan等人的BPE进行30K运算来形成词汇表。 （2018a）。为了进行评估，我们使用标准的ROUGE度量标准（Lin，2004年）并报告ROUGE -1，ROUGE -2和ROUGE -L。为了生成摘要，我们遵循标准实践来调整最小输出长度，并不允许重复相同的三字母组合（Paulus等人，2017）。表5显示了FAIRSEQ的结果。我们还考虑一种配置，在该配置中，我们将预先训练的语言模型表示形式输入到编码器网络，并且此语言模型在newscrawl和CNN-Dailymail上进行了训练，总共有1.9M个句子。

# 5 Conclusion

​		我们介绍了FAIRSEQ，这是一种用于序列建模的快速，可扩展的工具包，可扩展且适用于许多应用程序。 将来，我们将继续开发该工具包，以实现进一步的研究进展。

# References

Karim Ahmed, Nitish Shirish Keskar, and Richard Socher. 2017. Weighted transformer network for machine translation. arxiv, 1711.02132.

Mikel Artetxe and Holger Schwenk. 2018. Massively multilingual sentence embeddings for zeroshot cross-lingual transfer and beyond. arXiv, abs/1812.10464.

Alexei Baevski and Michael Auli. 2019. Adaptive input representations for neural language modeling. In Proc. of ICLR.

Yun Chen, Victor OK Li, Kyunghyun Cho, and Samuel R Bowman. 2018. A stable and effective learning strategy for trainable greedy decoding. arXiv, abs/1804.07915.

Shamil Chollampatt and Hwee Tou Ng. 2018. A multilayer convolutional encoder-decoder neural network for grammatical error correction. arXiv, abs/1801.08831.

Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. Language modeling with gated convolutional networks. In Proc. of ICML.

Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019. Wizard of Wikipedia: Knowledge-powered conversational agents. In Proc. of ICLR.

Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. 2018a. Understanding back-translation at scale. In Conference of the Association for Computational Linguistics (ACL).

Sergey Edunov, Myle Ott, Michael Auli, David Grangier, et al. 2018b. Classical structured prediction losses for sequence to sequence learning. In Proc. of NAACL.

Angela Fan, David Grangier, and Michael Auli. 2018a. Controllable abstractive summarization. In ACL Workshop on Neural Machine Translation and Generation.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018b.

Hierarchical neural story generation. In Proc. of ACL.

Angela Fan, Mike Lewis, and Yann Dauphin. 2019.

Strategies for structuring story generation. arXiv, abs/1902.01109.

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. 2017. Convolutional Sequence to Sequence Learning. In Proc. of ICML.

Sebastian Gehrmann, Yuntian Deng, and Alexander M Rush. 2018. Bottom-up abstractive summarization. arXiv, abs/1808.10792.

Edouard Grave, Armand Joulin, Moustapha Ciss´e, David Grangier, and Herv´e J´egou. 2017. Efﬁcient softmax approximation for gpus. In Proc. of ICML.

Edouard Grave, Armand Joulin, and Nicolas Usunier.

2016. Improving neural language models with a continuous cache. arXiv, abs/1612.04426.

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In NIPS.

Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton, and Matt Post. 2018. Sockeye: A Toolkit for Neural Machine Translation. arXiv, abs/1712.05690.

Hieu Hoang, Philipp Koehn, Ulrich Germann, Kenneth Heaﬁeld, and Barry Haddow. 2006. multi-bleu.perl. https://github.com/moses-smt/ mosesdecoder/blob/master/scripts/ generic/multi-bleu.perl.

Rafal J´ozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. 2016. Exploring the limits of language modeling. arXiv, abs/1602.02410.

Marcin Junczys-Dowmunt, Roman Grundkiewicz, Tomasz Dwojak, Hieu Hoang, Kenneth Heaﬁeld, Tom Neckermann, Frank Seide, Ulrich Germann, Alham Fikri Aji, Nikolay Bogoychev, Andr´e F. T. Martins, and Alexandra Birch. 2018. Marian: Fast neural machine translation in C++. In Proc. of ACL 2018, System Demonstrations.

Yoon Kim, Yacine Jernite, David Sontag, and Alexander M Rush. 2016. Character-aware neural language models. In Proc. of AAAI.

Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and Alexander M. Rush. 2017. OpenNMT: Open-source toolkit for neural machine translation. In Proc. ACL.

Oleksii Kuchaiev, Boris Ginsburg, Igor Gitman, Vitaly Lavrukhin, Carl Case, and Paulius Micikevicius. 2018. OpenSeq2Seq: Extensible Toolkit for Distributed and Mixed Precision Training of Sequenceto-Sequence Models. In Proc. of Workshop for NLP Open Source Software.

Guillaume Lample, Myle Ott, Alexis Conneau, Ludovic Denoyer, and Marc’Aurelio Ranzato. 2018. Phrase-based & neural unsupervised machine translation. In Proc. of EMNLP.

Chin-Yew Lin. 2004. Rouge: a package for automatic evaluation of summaries. In ACL Workshop on Text Summarization Branches Out.

Yizhu Liu, Zhiyi Luo, and Kenny Zhu. 2018. Controlling length in abstractive summarization using a convolutional neural network. In Proc. of EMNLP.

Ilya Loshchilov and Frank Hutter. 2016. Sgdr:

Stochastic gradient descent with warm restarts. In Proc. of ICLR. Minh-Thang Luong, Hieu Pham, and Christopher D Manning. 2015. Effective approaches to attentionbased neural machine translation. In Proc. of EMNLP.

Stephen Merity, Nitish Shirish Keskar, and Richard Socher. 2018. An analysis of neural language modeling at multiple scales. arXiv, abs/1803.08240.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory F. Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed Precision Training. In Proc. of ICLR.

A. H. Miller, W. Feng, A. Fisch, J. Lu, D. Batra,

A. Bordes, D. Parikh, and J. Weston. 2017. Parlai: A dialog research software platform. arXiv, abs/1705.06476.

Makoto Morishita, Yusuke Oda, Graham Neubig, Koichiro Yoshino, Katsuhito Sudoh, and Satoshi Nakamura. 2017. An empirical study of mini-batch creation strategies for neural machine translation. In Proc. of WMT.

Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, Caglar Gulcehre, and Bing Xiang. 2016. Abstractive text summarization using sequence-to-sequence rnns and beyond. In SIGNLL Conference on Computational Natural Language Learning.

Shashi Narayan, Shay B Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. arXiv, abs/1808.08745.

Myle Ott, Michael Auli, David Grangier, and MarcAurelio Ranzato. 2018a. Analyzing uncertainty in neural machine translation. In Proc. of ICML.

Myle Ott, Sergey Edunov, David Grangier, and Michael Auli. 2018b. Scaling neural machine translation. In Proc. of WMT.

Romain Paulus, Caiming Xiong, and Richard Socher.

2017. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304.

Matt Post. 2018. A call for clarity in reporting bleu scores. arXiv, abs/1804.08771.

Jack W. Rae, Chris Dyer, Peter Dayan, and Timothy P.

Lillicrap. 2018. Fast parametric learning with activation memorization. arXiv, abs/1803.10049.

Abigail See, Peter J. Liu, and Christopher D. Manning.

2017. Get to the point: Summarization with pointergenerator networks. In ACL.

Rico Sennrich, Barry Haddow, and Alexandra Birch.

2016. Neural machine translation of rare words with subword units. In Proc. of ACL.

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani.

2018. Self-attention with relative position representations. In Proc. of NAACL.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv, abs/1701.06538.

Noam Shazeer and Mitchell Stern. 2018. Adafactor:

Adaptive learning rates with sublinear memory cost. arXiv preprint arXiv:1804.04235.

Tianxiao Shen, Myle Ott,

Marc’Aurelio Ranzato. 2019. diverse machine translation: arXiv, abs/1902.07816.

Michael Auli, and Mixture models for Tricks of the trade.

Kaitao Song, Xu Tan, Di He, Jianfeng Lu, Tao Qin, and Tie-Yan Liu. 2018. Double path networks for sequence to sequence learning. arXiv, abs/1806.04856.

A. Vaswani, S. Bengio, E. Brevdo, F. Chollet, A. N. Gomez, S. Gouws, L. Jones, Ł. Kaiser, N. Kalchbrenner, N. Parmar, R. Sepassi, N. Shazeer, and

J. Uszkoreit. 2018. Tensor2Tensor for Neural Machine Translation. arXiv, abs/1803.07416.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In Proc. of NIPS.

Ashwin K Vijayakumar, Michael Cogswell, Ramprasath R Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. 2016. Diverse beam search: Decoding diverse solutions from neural sequence models. arXiv preprint arXiv:1610.02424.

Felix Wu, Angela Fan, Alexei Baevski, Yann N. Dauphin, and Michael Auli. 2019. Pay less attention with lightweight and dynamic convlutions. In Proc. of ICLR.