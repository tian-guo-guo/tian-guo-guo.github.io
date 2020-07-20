---
layout:     post           # 使用的布局（不需要改）
title:      2017 An Empirical Comparison of Simple Domain Adaptation Methods for Neural Machine Translation           # 标题 
subtitle:   mixed fine tuning #副标题
date:       2020-05-26             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 领域适应



---

# 2017 An Empirical Comparison of Simple Domain Adaptation Methods for Neural Machine Translation

![image-20200526203740595](https://tva1.sinaimg.cn/large/007S8ZIlly1gf64tk8s6zj30uq04e0tl.jpg)

# Abstract

​		在本文中，我们提出了一种新的域自适应方法，称为“**混合调谐”，“mixed ﬁne tuning”**，用于神经机器翻译(NMT)。 我们结合了两种现有的方法，即优化和多域NMT **ﬁne tuning and multi domain NMT**.。 **我们首先在一个领域外的并行语料库上训练一个NMT模型，然后在一个混合了领域内和领域外语料库的并行语料库上调整它**。 所有语料库都增加了专业标记，以指示特定领域。 我们将我们提出的方法与优化和多域方法进行了经验比较，并讨论了其优点和缺点。

# 1 Introduction

​		神经机器翻译(NMT)最吸引人的特征之一（Bahdanau et al.，2015；Cho et al.，2014；Sutskever et al.，2014）是有可能训练一个端到端的系统，而不需要处理单词对齐，翻译规则和复杂的解码算法，这是统计机器翻译(SMT)系统的一个特点。 然而，据报道，**只有在平行语料库丰富的情况下，NMT才能比SMT更好地发挥作用。 在低资源域的情况下，vanilla NMT要么比SMT差，要么与SMT相当**（Zoph等人，2016年）。

​		域自适应Domain adaptation已被证明对低资源NMT是有效的。 **传统的域自适应方法是调整，即根据域内数据进一步训练域外模型**（Luong和Manning，2015；Sennrich等人，2016b；Servan等人，2016；Freitag和Al-Onaizan，2016）。 **然而，由于域内数据较小，调优往往会很快过度**。 **另一方面，多域NMT（Kobus等人，2016）涉及为多个域训练单个NMT模型。 该方法通过修改并行语料库来添加标记“<2domain>”，以指示域，而无需对NMT系统架构进行任何修改。 但是，这种方法还没有特别针对域自适应进行研究**。

​		<u>基于这两方面的研究</u>，我们提出了一种名为“混合调优”的新的领域自适应方法，**首先在一个领域外的并行语料库上训练NMT模型，然后在一个混合了领域内和领域外语料库的并行语料库上对其进行调优。 对混合语料库而不是域内语料库进行微调可以解决过度筛选问题。 所有语料库都增加了专业标记，以指示特定领域**。we ﬁrst train an NMT model on an out-of-domain parallel corpus, and then ﬁne tune it on a parallel corpus that is a mix of the in-domain and out-of-domain corpora. Fine tuning on the mixed corpus instead of the indomain corpus can address the overﬁtting problem. All corpora are augmented with artiﬁcial tags to indicate speciﬁc domains. 我们尝试了两种不同的语料库设置:

​		•手工创建资源贫乏语料库:使用NTCIR数据（专利领域；资源丰富）（Goto等人，2013年）改进IWSLT数据（TED Talks；资源贫乏）的翻译质量（Cettolo等人，2015年）。

​		•自动提取资源贫乏语料库:使用ASPEC数据（科学领域；资源丰富）（Nakazawa et al.，2016）提高Wiki数据（资源贫乏）的翻译质量。 后一个域的平行语料库被自动提取（Chu等人，2016a)。

​		**我们观察到，“混合调优”的效果明显优于分别使用调优和基于域标记的方法的方**法。 我们的贡献是双重的:

​		•我们提出了一种结合现有最佳方法的新方法，并已证明它是有效的。

​		•据我们所知，这是对各种域适应方法进行实证比较的首次工作。

# 2 Related Work

​		除了使用标签进行调谐和多域NMT外，域自适应的另一个方向是使用域内单语数据。 不是研究了为NMT解码器训练域内递归神经语言(RNN)语言模型(Güulccoverehre et al.，2015）就是通过反翻译目标域内单语数据生成合成数据（Sennrich et al.，2016b)。

# 3. Methods for Comparison

​		我们比较的所有方法都很简单，不需要对NMT系统进行任何修改。

## 3.1 Fine Tuning

​		微调是领域适应的常规方式，因此作为本研究的基线。 在这种方法中，我们首先在资源丰富的域外语料库上训练NMT系统直到收敛，然后在资源贫乏的域内语料库上调整其参数（图1）。

![image-20200526204205289](https://tva1.sinaimg.cn/large/007S8ZIlly1gf64y5k23rj30dw066t9d.jpg)

## 3.2 Multi Domain

​		多域法最初是由（Sennrich et al.，2016a)提出的，它利用tags来控制NMT翻译的politeness。 该方法的概述如图2中虚线部分所示。 **在这种方法中，我们简单地将多个域的语料库与两个小的修改连接起来:a。 将域标记“<2domain>”添加到各自语料库的源句中。这使NMT解码器为特定域生成句子。 b.对较小的语料库进行过采样，以便训练过程对每个领域给予同等的关注**。In this method, we simply concatenate the corpora of multiple domains with two small modiﬁcations: a. Appending the domain tag “<2domain>” to the source sentences of the respective corpora. This primes the NMT decoder to generate sentences for the speciﬁc domain. b. Oversampling the smaller corpus so that the training procedure pays equal attention to each domain.

​		我们可以根据域内数据进一步调优多域模型，这称为“多域+调优”。

![image-20200527104708524](https://tva1.sinaimg.cn/large/007S8ZIlly1gf6tdgcquxj30lc08y75w.jpg)

​		图2:使用域标签进行域自适应的混合调谐（虚线矩形中的部分表示多域方法）。

## 3.3 Mixed Fine Tuning

​		建议的混合调谐方法是上述方法的组合（如图2所示）。 培训程序如下:
​		1.**在域外数据上训练NMT模型直到收敛**。Train an NMT model on out-of-domain data till convergence.

​		2.**在域内和域外数据的混合上（通过对域内数据进行过采样）恢复从步骤1开始的NMT模型的训练，直到收敛**。Resume training the NMT model from step 1 on a mix of in-domain and out-of-domain data (by oversampling the in-domain data) till convergence.

​		默认情况下，我们利用域标记，但我们也考虑不使用它们的设置（即“w/o标记”）。 我们可以根据域内数据进一步调整步骤2中的模型，这称为“混合调整+优化”。“mixed ﬁne tuning + ﬁne tuning.”

​		请注意，**在“finne tuning”方法中，从域外数据获得的词汇表用于域内数据**； 而对于“多域”和“混合调优”方法，我们在所有训练阶段使用从混合域内和域外数据中获得的词汇表。while for the “multi domain” and “mixed ﬁne tuning” methods, **we use a vocabulary obtained from the mixed in-domain and out-of-domain data for all the training stages.**

# 4 实验设置

​		我们在两种不同的设置下进行了NMT域适配实验，具体如下:

## 4.1高质量的领域内语料库设置

​		汉英翻译是高质量领域内语料库建设的重点。 我们利用资源丰富的领域外专利数据来扩充资源贫乏的领域内口语数据。 专利领域MT是在NTCIR10 workshop 2（Goto et al.，2013）专利MT任务的汉英子任务（NTCIR-CE）上进行的。 **NTCIRCE任务分别使用1000000(一百万)，2000和2000个句子进行训练，开发和测试**。 在2015年国际英语口语教学研讨会（Cettolo et al.，2015）上，我们在TED talk MT任务的汉英子任务（IWSLT-CE）上进行了口语领域MT的研究。 **IWSLT-CE任务包含209,491(20万)个用于训练的句子。 我们使用了DEV2010集进行开发，包含887个句子。 我们在2010年，2011年，2012年和2013年的测试集上评估了所有方法，测试集分别包含1570，1245，1397和1261个句子**。

## 4.2 Low Quality In-domain Corpus Setting

​		汉译日是低质量领域内语料库建设的重点。 **我们利用资源丰富的科学领域外数据来扩充资源贫乏的维基百科（本质上是开放的）领域内数据**。 科学领域MT是在中日论文摘录语料库(ASPEC-CJ)3(Nakazawa et al.，2016）上进行的，这是亚洲翻译研讨会(WAT)4（Nakazawa et al.，2015）的一个子任务。 **ASPEC-CJ任务分别使用672315，2090和2107个句子进行训练，开发和测试**。 Wikipedia领域任务是在从Wikipedia（WIKI-CJ）（Chu et al.，2016a)中自动提取的汉语-日语语料库上进行的，使用ASPEC-CJ语料库作为种子。 **WIKI-CJ任务包含136013，198和198个句子，分别用于训练，开发和测试**。

## 4.3 MT系统

​		对于NMT，我们使用了[KyotoNMT](https://github.com/fabiencro/knmt)系统5（Cromieres等人，2016）。 NMT的培训设置与参加2016年WAT的最佳系统相同。 **源和目标词汇表，源和目标侧嵌入，隐藏状态，注意机制隐藏状态和具有2-Maxout层的deep softmax输出的大小分别设置为32,000，620,1000,1000和500。 我们使用了两层LSTM的源侧和目标侧。 ADAM用作学习算法，inter-layer为20%，L2正则化的权重衰减系数为1e-6。 迷你批处理大小为64个，超过80个代币的句子被丢弃。 当我们观察到开发集的BLEU得分收敛时，我们早期停止了训练过程。 为了进行测试，我们将最佳显影损失，最佳显影BLEU和基本参数这三个参数自组装在一起。 光束大小设置为100**。

​		为了进行性能比较，我们还对基于短语的SMT(PBSMT)进行了实验。 我们使用Moses PBSMT系统（Koehn等人，2007）进行所有的MT实验。 对于各自的任务，我们分别使用[KenLM toolkit 6](https://github.com/kpu/kenlm/)和插值的Kneser-Ney贴现来训练目标侧的5-Gram语言模型。 在我们的所有实验中，我们使用[Giza++Toolkit7](http://code.google.com/p/giza-pp)进行单词对齐； 通过最小错误率训练（Och，2003）进行调优，每次实验都要重新运行。

​		对于两个MT系统，我们对数据进行了如下预处理。 **对于汉语**，我们使用[kyotomorph8](https://bitbucket.org/msmoshen/kyotomorph-beta)进行切分，该切分是在CTB版本5(CTB5)和SCTB（Chu et al.，2016b)上训练的。 **对于英语**，我们使用Moses中的tokenizer.perl脚本对句子进行了标记化和小写。 使用JUMAN 9对日语进行分段（Kurohashi等人，1994）。

​		**对于NMT，我们使用字节对编码(BPE)进一步将单词拆分成子单词（Sennrich等人，2016c)，该方法已被证明对NMT中的稀有单词问题是有效的**。 <u>使用子词的另一个动机是使不同领域共享更多的词汇，这对于资源贫乏的领域尤为重要</u>。 对于汉语到英语的任务，我们分别训练了两个基于汉语和英语词汇的BPE模型。 对于中日任务，由于中日两国可以共享一些汉字词汇，因此我们训练了一个基于中日两国词汇的联合BPE模型。 所有任务的合并操作数都设置为30,000。

# 5 Result

![image-20200526204638739](https://tva1.sinaimg.cn/large/007S8ZIlly1gf652wfkwqj30q80aoq5u.jpg)

![image-20200526204650206](https://tva1.sinaimg.cn/large/007S8ZIlly1gf6533i920j30eq0aujtc.jpg)

​		表1和表2分别显示了汉译英和汉译日任务的翻译结果。 具有SMT和NMT的条目分别是PBSMT和NMT系统； 其他方法是在第3节中描述的不同方法。 在这两个表中，粗体数字表示最佳系统以及与最佳系统无明显差异的所有系统。 显著性测试采用自举重采样法（Koehn，2004年），p<0.05。

​		我们可以看到，在没有域自适应的情况下，SMT系统在资源贫乏的域（即IWSLT-CE和Wiki-CJ）上的性能明显优于NMT系统； 而在资源丰富的领域，即NTCIR-CE和ASPECCJ上，NMT的性能优于SMT。 直接使用在域外数据上训练的SMT/NMT模型来转换域内数据，性能较差。 采用我们提出的“混合调谐”域自适应方法，NMT在域内任务上的性能显著优于SMT。

​		比较不同的域自适应方法，“混合调谐”显示出最佳性能。 我们认为，这是因为“混合调谐”可以解决“微调”的过度问题。我们观察到，“微调”仅在1个训练周期后就会迅速过度，而“混合调谐”只会略微过度直至收敛。 此外，“混合调优”不会恶化域外翻译的质量，而“微调”和“多域”则会。 “混合调谐”的一个缺点是，与“单一调谐”相比，“混合调谐”需要更长的时间，因为直到收敛的时间基本上与用于调谐的数据大小成正比。

​		“多域”的性能要么与（IWSLT-CE）一样好，要么比（WIKI-CJ）“微调”差，但“混合调谐”的性能要么明显优于（IWSLT-CE），要么与（WIKI-CJ）“微调”相当。我们认为，这两项任务的性能差异是由于它们的独特特性造成的。 由于WIKI-CJ数据的质量相对较差，因此将其与域外数据混合使用并不具有与IWSLTCE数据相同的积极效果。

​		域标签对“多域”和“混合调优”都有帮助。本质上，对域内数据的进一步调优对“多域”和“混合调优”都没有帮助。我们认为，这是因为“多域”和“混合调谐”方法已经利用了用于调谐的域内数据。

# 6 结论

​		在本文中，我们提出了一种新的域自适应方法，名为“混合调谐”，用于NMT。 我们将我们提出的方法与优化和多域方法进行了经验比较，结果表明，它是有效的，但对所使用的域内数据的质量敏感。

​		在未来，我们计划将RNN模型融入到我们现有的体系结构中，以利用丰富的域内单语语料库。 我们还计划通过回译大型域内单语语料库来探索合成数据的效果。

# References

[Bahdanau et al.2015] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In In Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015), San Diego, USA, May. International Conference on Learning Representations.

[Cettolo et al.2015] M Cettolo, J Niehues, S St¨uker, L Bentivogli, R Cattoni, and M Federico. 2015. The iwslt 2015 evaluation campaign. In Proceedings of the Twelfth International Workshop on Spoken Language Translation (IWSLT).

[Cho et al.2014] Kyunghyun Cho, Bart van Merri¨enboer, C¸alar G¨ulc¸ehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using rnn encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734, Doha, Qatar, October. Association for Computational Linguistics.

[Chu et al.2016a] Chenhui Chu, Raj Dabre, and Sadao Kurohashi. 2016a. Parallel sentence extraction from comparable corpora with neural network features. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016), Paris, France, may. European Language Resources Association (ELRA).

[Chu et al.2016b] Chenhui Chu, Toshiaki Nakazawa, Daisuke Kawahara, and Sadao Kurohashi. 2016b. SCTB: A Chinese treebank in scientiﬁc domain. In Proceedings of the 12th Workshop on Asian Language Resources (ALR12), pages 59–67, Osaka, Japan, December. The COLING 2016 Organizing Committee.

[Cromieres et al.2016] Fabien Cromieres, Chenhui Chu, Toshiaki Nakazawa, and Sadao Kurohashi. 2016. Kyoto university participation to wat 2016. In Proceedings of the 3rd Workshop on Asian Translation (WAT2016), pages 166–174, Osaka, Japan, December. The COLING 2016 Organizing Committee.

[Freitag and Al-Onaizan2016] Markus Freitag and Yaser Al-Onaizan. 2016. Fast domain adaptation for neural machine translation. arXiv preprint arXiv:1612.06897.

[Goto et al.2013] Isao Goto, Ka-Po Chow, Bin Lu, Eiichiro Sumita, and Benjamin K. Tsou. 2013. Overview of the patent machine translation task at the ntcir-10 workshop. In Proceedings of the 10th NTCIR Conference, pages 260–286, Tokyo, Japan, June. National Institute of Informatics (NII).

[G¨ulc¸ehre et al.2015] C¸aglar G¨ulc¸ehre, Orhan Firat, Kelvin Xu, Kyunghyun Cho, Lo¨ıc Barrault, HueiChi Lin, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2015. On using monolingual corpora in neural machine translation. CoRR, abs/1503.03535.

[Kobus et al.2016] Catherine Kobus, Josep Crego, and Jean Senellart. 2016. Domain control for neural machine translation. arXiv preprint arXiv:1612.06140.

[Koehn et al.2007] Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, Chris Dyer, Ondrej Bojar, Alexandra Constantin, and Evan Herbst. 2007. Moses: Open source toolkit for statistical machine translation. In Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics Companion Volume Proceedings of the Demo and Poster Sessions, pages 177–180, Prague, Czech Republic, June. Association for Computational Linguistics.

[Koehn2004] Philipp Koehn. 2004. Statistical signiﬁcance tests for machine translation evaluation. In Proceedings of EMNLP 2004, pages 388–395, Barcelona, Spain, July. Association for Computational Linguistics.

[Kurohashi et al.1994] Sadao Kurohashi, Toshihisa Nakamura, Yuji Matsumoto, and Makoto Nagao. 1994. Improvements of Japanese morphological analyzer JUMAN. In Proceedings of the International Workshop on Sharable Natural Language, pages 22–28.

[Luong and Manning2015] Minh-Thang Luong and Christopher D Manning. 2015. Stanford neural machine translation systems for spoken language domains. In Proceedings of the 12th International Workshop on Spoken Language Translation, pages 76–79, Da Nang, Vietnam, December.

[Nakazawa et al.2015] Toshiaki Nakazawa, Hideya Mino, Isao Goto, Graham Neubig, Sadao Kurohashi, and Eiichiro Sumita. 2015. Overview of the 2nd Workshop on Asian Translation. In Proceedings of the 2nd Workshop on Asian Translation (WAT2015), pages 1–28, Kyoto, Japan, October.

[Nakazawa et al.2016] Toshiaki Nakazawa, Manabu Yaguchi, Kiyotaka Uchimoto, Masao Utiyama, Eiichiro Sumita, Sadao Kurohashi, and Hitoshi Isahara. 2016. Aspec: Asian scientiﬁc paper excerpt corpus. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016), Paris, France, May. European Language Resources Association (ELRA).

[Och2003] Franz Josef Och. 2003. Minimum error rate training in statistical machine translation. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics, pages 160–167, Sapporo, Japan, July. Association for Computational Linguistics.

[Sennrich et al.2016a] Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016a. Controlling politeness in neural machine translation via side constraints. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 35–40, San Diego, California, June. Association for Computational Linguistics.

[Sennrich et al.2016b] Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016b. Improving neural machine translation models with monolingual data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 86–96, Berlin, Germany, August. Association for Computational Linguistics.

[Sennrich et al.2016c] Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016c. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany, August. Association for Computational Linguistics.

[Servan et al.2016] Christophe Servan, Josep Crego, and Jean Senellart. 2016. Domain specialization: a post-training domain adaptation for neural machine translation. arXiv preprint arXiv:1612.06141.

[Sutskever et al.2014] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to sequence learning with neural networks. In Proceedings of the 27th International Conference on Neural Information Processing Systems, NIPS’14, pages 31043112, Cambridge, MA, USA. MIT Press.

[Zoph et al.2016] Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. 2016. Transfer learning for low-resource neural machine translation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016, pages 1568–1575.