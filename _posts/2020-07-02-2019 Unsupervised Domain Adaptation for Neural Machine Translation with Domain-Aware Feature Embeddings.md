---
layout:     post           # 使用的布局（不需要改）
title:      2019 Unsupervised Domain Adaptation for Neural Machine Translation with Domain-Aware Feature Embeddings           # 标题 
subtitle:   DAFE for MT #副标题
date:       2020-07-02             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 领域适应

---

# 2019 Unsupervised Domain Adaptation for Neural Machine Translation with Domain-Aware Feature Embeddings

![image-20200702105856161](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702105858.png)

# Abstract

​		神经机器翻译模型的最新成功依赖于**高质量的域内数据的可用性**。当特定于域的数据稀少或不存在时，需要进行**域自适应**。<u>先前的非监督域适应策略包括使用域内复制的单语或反向翻译数据训练模型。但是，这些方法使用通用表示形式表示文本，而不考虑域移位，这使得翻译模型无法控制以特定域为条件的输出</u>。在这项工作中，**我们提出了一种通过领域辅助特征嵌入来调整模型的方法，该特征嵌入是通过辅助语言建模任务来学习的。我们的方法允许模型将特定领域的表示形式分配给单词，并在所需领域中输出句子**。我们的经验结果证明了所提出策略的有效性，在多个实验环境中均实现了持续改进。此外，我们表明将我们的方法与反向翻译相结合可以进一步提高模型的性能。

# 1 Introduction

​		虽然神经机器翻译（NMT）系统已被证明在可获得大量域内数据的情况下有效（Gehring等，2017; Vaswani等，2017; Chen等，2018），但它们具有当测试域与训练数据不匹配时，被证明表现不佳（Koehn和Knowles，2017）。在我们感兴趣的所有可能域中收集大量并行数据是昂贵的，而且在许多情况下是不可能的。因此，必须探索有效的方法来训练能够很好地推广到新领域的模型。
​		神经机器翻译的领域自适应已在研究界引起了广泛关注，<u>其中大部分工作都集中在有少量领域内数据可用的有监督的环境下</u>（Luong和Manning，2015年; Freitag和Al-Onaizan，2016年） ; Chu等人，2017年; Vilar，2018年）。<u>一种既定的方法是将域标记用作附加输入，并通过并行数据学习域表示形式</u>（Kobus等人，2017）。在这项工作中，我们**专注于无监督适应，其中没有可用的域内并行数据**。在这种范式中，Currey等人。 （2017年）<u>将域内单语数据从目标端复制到源端</u>，并且Sennrich等。 （2016a）<u>将反向翻译的数据与原始语料库连接在一起</u>。**但是，这些方法学习所有文本的通用表示形式，因为所学习的表示形式在所有领域中都是共享的，并且合成数据和自然数据均得到同等对待。由于来自不同域的数据固有地不同，因此共享嵌入可能不是最佳的。当单词在不同领域中具有不同的含义时，这个问题会更加严重**。

​		在这项工作中，**我们提出了一种DomainAware特征嵌入Domain-Aware Feature Embedding（DAFE）方法，该方法通过将表示分解成不同的部分来执行无监督的域自适应。因为我们没有域内并行数据，所以我们通过辅助任务（即语言建模）学习DAFE。具体来说，我们提出的模型由一个基础网络组成，其参数在设置之间以及域和任务嵌入学习器之间共享。通过将模型分为不同的组件，DAFE可以学习针对特定领域和任务量身定制的表示形式，然后将其用于领域适应。**

​		我们在两种不同的数据设置下，在基于**Transformer**的NMT系统（Vaswani等，2017）中评估了我们的方法。我们的方法表明，与未采用的基准相比，可以持续改进多达5个BLEU点，而对于强大的**反向翻译模型**，则可以持续提高2个BLEU点。**将我们的方法与反向翻译相结合**可以进一步提高模型的性能，这表明所提出的方法和依赖于合成数据的方法是正交的。

# 2 Methods

​		在本节中，我们首先说明DAFE的体系结构，然后描述整体培训策略。

## 2.1 Architecture

![image-20200702110700929](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702110702.png)

​		DAFE将隐藏状态分解为不同的部分，以便网络可以学习特定领域或任务的表示形式，如图1所示。**具体来说，它由三部分组成：a base network with parameters θbase，用于学习跨不同任务和领域的共同特征。 a domain-aware feature embedding learner提供给定输入域τ生成嵌入θ域的领域感知特征嵌入τ学习器和给定输入任务γ输出任务表示θtask的任务感知特征嵌入γ学习器。每层的最终输出是通过基础网络输出和特征嵌入的组合获得的**。

​		基本网络在编码器-解码器框架中实现（Sutskever等，2014； Cho等，2014）。任务嵌入学习者和领域嵌入学习者都可以通过查找操作直接输出特征嵌入。

​		在这项工作中，学习者中的领域感知嵌入分别从域内和域外数据中学习领域表示θin domain和θout domain，而任务感知嵌入精简器mt lm学习任务嵌入θmt task和θlm task。

​		特征嵌入在每个编码层（包括源单词嵌入层）生成，并且具有与基本模型的隐藏状态相同的大小。应该注意的是，特征嵌入学习器在不同的层上生成不同的嵌入。

​		形式上，给定特定域τ和任务γ，（l）第l编码层He的输出为：

![image-20200702112838065](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702112839.png)

​		其中θ域和θtask是单个向量，L AYER e（·）可以是任何层编码函数，

​		例如LSTM（Hochreiter and Schmidhuber，1997）或Transformer（Vaswani et al。，2017）。

​		在本文中，我们采用简单的加法运算来组合不同部分的输出，这些部分已经达到了令人满意的性能，如第3节所示。我们将继续研究更复杂的组合策略以用于将来的工作。

## 2.2 Training Objectives

​		在无人监督的领域适应性设置中，我们假设访问域外并行训练语料（X out，Y out）和目标语言域内单语数据Y in。

​		神经机器翻译。 我们的目标任务是机器翻译。 基础网络和嵌入式学习者都经过共同培训，目标是：

![image-20200702113100463](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702113103.png)

​		语言建模。 我们选择屏蔽语言建模（LM）作为我们的辅助任务。 继兰普尔等。 （2018a，b），我们通过随机删除和略微整理单词来为每个目标句子y创建损坏的版本C（y）。 在训练期间，使用梯度上升来最大化目标：

![image-20200702113207116](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702113208.png)

​		其中，域外数据θ= {θbase，θlm task，θout domain}，{θbase，θlm task，θin domain}表示域内数据。

​		培训策略。 算法1中显示了我们的训练策略。最终目标是在mt中学习用于域内机器翻译的一组参数{θbase，θin domain，θmt task}。 域外并行数据允许我们训练mt {θbase，θout domain，θmt task}，但是单语言数据有助于模型学习θin domain和θout domain。

![image-20200702113441835](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702113443.png)

# 3 Experiments

## 3.1 Setup

​		数据集。我们在两个不同的数据设置中验证我们的模型。首先，我们训练德语-英语OPUS语料库的法律，医学和IT数据集（Tiedemann，2012年），并测试我们的方法从一个领域适应另一个领域的能力。数据集在每个域中包含2K个开发和测试语句，分别包含约715K，1M和337K训练语句。这些数据集相对较小，并且各个域之间相距甚远。在第二个设置中，我们将在通用域WMT-14数据集上训练的模型改编为TED（Duh，2018）和法律，医学OPUS数据集。对于此设置，我们考虑两种语言对，即捷克语和德语对英语。捷克英语和德语英语数据集包含1M和450万个句子，而开发和测试集则包含约2K个句子。
​		模型。我们在Transformer模型的顶部实现DAFE。编码器和解码器均由4层组成，隐藏大小设置为512。字节对编码（Sennrich等，2016b）用于将训练数据处理为最终共享词汇量为50K的子字。

​		基线。我们将我们的方法与两个基准模型进行比较：1）复制的单语数据模型（Currey等人，2017），该模型将目标域内单语数据复制到源端; 2）反向翻译（Sennrich等，2016a），通过目标-源NMT模型生成合成的域内并行数据，从而丰富了训练数据。我们将这两个基准表征为以数据为中心的方法，因为它们依赖于合成数据。相反，我们的方法是以模型为中心的，因为我们主要致力于修改模型体系结构。我们还通过移除嵌入学习器（称为“ DAFE w / o Embed”）来执行消融研究，该模型将仅执行多任务学习。

## 3.2 Main Results

![image-20200702115214151](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702115216.png)

​		域之间的适应。 如表1的前6个结果栏中所示，当在域之间进行适应时，不适应的基线模型（第1行）的性能较差。 复制方法（第2行）和反向翻译（第3行）都显着改善了模型，反向翻译是复制的更好选择。 与反向翻译相比，DAFE（第5行）的性能更高，最多可提高2个BLEU点。 同样，删除嵌入学习器会导致性能下降（第4行），表明存在它们的必要性。

​		从一般领域适应特定领域。 在第二个数据设置（表1的最后6列）中，具有相对大量的generaldomain数据集，反向翻译可实现竞争性能。 在这种情况下，DAFE可以显着改善不适应的基线，但不会超过反向翻译。 我们推测这是因为反向翻译数据的质量相对较好。	

## 3.3 Combining DAFE with Back-Translation

​		我们推测DAFE是对以数据为中心的方法的补充。我们尝试通过将DAFE与反向翻译（最好的以数据为中心的方法）相结合来支持这种直觉。我们尝试了三种不同的策略，将DAFE与反向翻译相结合，如表1所示。

​		仅使用反向翻译的数据来训练DAFE（第6行）就已经实现了多达4个BLEU点的显着改进。我们还可以使用通过DAFE训练的目标到源模型来生成反向翻译的数据，通过该模型我们可以训练正向模型（Back-DAFE，第7行）。通过这样做，反向翻译的数据将具有更高的质量，因此可以改善源到目标模型的性能。最佳的总体策略是使用Back-DAFE生成合成的域内数据，并使用反向翻译后的数据训练DAFE模型（BackDAFE + DAFE，第8行）。根据我们的直觉，在几乎所有的适应设置中，Back-DAFE + DAFE都能带来更高的翻译质量。此设置的优势在于，逆向翻译后的数据允许我们通过翻译任务学习θ域。

## 3.4 Analysis

![image-20200702120305101](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702120425.png)

​		资源短缺的情况。 DAFE相对于反向翻译的一个优势是，我们不需要良好的从目标到源的翻译模型，而在资源匮乏的情况下很难做到这一点。我们随机抽样不同数量的训练数据，并评估DAFE的性能和开发集上的反向翻译。如图2所示，在数据稀少的情况下，DAFE的性能明显优于反向翻译，因为低质量的反向翻译数据实际上可能对下游性能有害。

​		控制输出域。我们模型的另一个优点是能够通过提供所需的域嵌入来控制输出域。如表2所示，馈送不匹配的域嵌入会导致性能下降。表3中的示例进一步表明，以医学嵌入作为输入的模型可以生成特定领域的单词，例如“ EMEA”（欧洲药品评估局）和“肌肉注射”，而IT嵌入则鼓励该模型生成诸如“ bug”和“ developers”之类的单词”。

![image-20200702121009746](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200702121011.png)

# 4 Related Work

​		NMT以前的大多数域适应工作都集中在可用少量域内数据的设置上。继续训练（Luong和Manning，2015； Freitag和Al-Onaizan，2016）方法是首先在域外数据上训练NMT模型，然后在域内数据上对其进行微调。与我们的工作类似，Kobus等。 （2017）提出使用域标签来控制输出域，但是它仍然需要域内并行语料库，我们的体系结构允许更多灵活的修改，而不仅仅是添加其他标签。

​		NMT的无监督领域自适应技术可以分为以数据为中心和以模型为中心的方法（Chu and Wang，2018）。以数据为中心的方法主要集中在使用现有的域内单语言数据选择或生成与域相关的数据。复制方法（Currey等人，2017）和反向翻译（Sennrich等人，2016a）都是以数据为中心的代表性方法。此外，Moore和Lewis（2010）； Axelrod等。 （2011）；杜等。 （2013年）使用LM来对域外数据进行评分，并据此选择类似于域内文本的数据。以模型为中心的方法尚未得到充分研究。 Gulcehre等。 （2015）提出融合LMs和NMT模型，但是他们的方法需要在推论过程中查询两个模型，并且已经证明不如以数据为中心的模型（Chu等人，2018）。还通过检索类似于测试集的训练数据中的句子或n-gram来进行适应性工作（Farajian等，2017; Bapna和Firat，2019）。但是，很难在域自适应设置中找到相似的平行句子。

# 5 Conclusion

​		在这项工作中，我们提出了一种简单而有效的用于神经机器翻译的无监督领域自适应技术，该技术通过使用语言建模学习的领域感知特征嵌入来对模型进行自适应。 实验结果证明了该建议方法在各种环境下的有效性。 此外，分析表明，我们的方法允许我们控制翻译结果的输出域。 未来的工作包括设计更复杂的体系结构和组合策略，以及在其他语言对和数据集上验证我们的模型。

# References

Amittai Axelrod, Xiaodong He, and Jianfeng Gao. 2011. Domain adaptation via pseudo in-domain data selection. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Ankur Bapna and Orhan Firat. 2019. Non-parametric adaptation for neural machine translation. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL).

Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, et al. 2018. The best of both worlds: Combining recent advances in neural machine translation. In Annual Meeting of the Association for Computational Linguistics (ACL).

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using rnn encoder–decoder for statistical machine translation. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Chenhui Chu, Raj Dabre, and Sadao Kurohashi. 2017. An empirical comparison of domain adaptation methods for neural machine translation. In Annual Meeting of the Association for Computational Linguistics (ACL).

Chenhui Chu, Raj Dabre, and Sadao Kurohashi. 2018. A comprehensive empirical comparison of domain adaptation methods for neural machine translation. Journal of Information Processing, 26:529–538.

Chenhui Chu and Rui Wang. 2018. A survey of domain adaptation for neural machine translation. In International Conference on Computational Linguistics (COLING).

Anna Currey, Antonio Valerio Miceli Barone, and Kenneth Heaﬁeld. 2017. Copied monolingual data improves low-resource neural machine translation. In Conference on Machine Translation (WMT).

Kevin Duh. 2018. The multitarget ted talks task.

http://www.cs.jhu.edu/ ˜kevinduh/a/ multitarget-tedtalks/.

Kevin Duh, Graham Neubig, Katsuhito Sudoh, and Hajime Tsukada. 2013. Adaptation data selection using neural language models: Experiments in machine translation. In Annual Meeting of the Association for Computational Linguistics (ACL).

M Amin Farajian, Marco Turchi, Matteo Negri, and Marcello Federico. 2017. Multi-domain neural machine translation through unsupervised adaptation. In Conference on Machine Translation (WMT).

Markus Freitag and Yaser Al-Onaizan. 2016. Fast domain adaptation for neural machine translation. arXiv preprint arXiv:1612.06897.

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. 2017. Convolutional sequence to sequence learning. In International Conference on Machine Learning (ICML).

Caglar Gulcehre, Orhan Firat, Kelvin Xu, Kyunghyun Cho, Loic Barrault, Huei-Chi Lin, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2015. On using monolingual corpora in neural machine translation. arXiv preprint arXiv:1503.03535.

Sepp Hochreiter and J¨urgen Schmidhuber. 1997.

Long short-term memory. Neural computation, 9(8):1735–1780.

Catherine Kobus, Josep Crego, and Jean Senellart. 2017. Domain control for neural machine translation. In International Conference Recent Advances in Natural Language Processing (RANLP).

Philipp Koehn and Rebecca Knowles. 2017. Six challenges for neural machine translation. In Workshop on Neural Machine Translation (WMT).

Guillaume Lample, Alexis Conneau, Ludovic Denoyer, and Marc’Aurelio Ranzato. 2018a. Unsupervised machine translation using monolingual corpora only. In International Conference on Learning Representations (ICLR).

Guillaume Lample, Myle Ott, Alexis Conneau, Ludovic Denoyer, et al. 2018b. Phrase-based & neural unsupervised machine translation. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Minh-Thang Luong and Christopher D Manning. 2015. Stanford neural machine translation systems for spoken language domains. In International Workshop on Spoken Language Translation (IWSLT).

Robert C. Moore and William Lewis. 2010. Intelligent selection of language model training data. In Annual Meeting of the Association for Computational Linguistics (ACL).

Graham Neubig, Zi-Yi Dou, Junjie Hu, Paul Michel, Danish Pruthi, and Xinyi Wang. 2019. compare-mt: A tool for holistic comparison of language generation systems. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) Demo Track.

Rico Sennrich, Barry Haddow, and Alexandra Birch.

2016a. Improving neural machine translation models with monolingual data. In Annual Meeting of the Association for Computational Linguistics (ACL).

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016b. Neural machine translation of rare words with subword units. In Annual Meeting of the Association for Computational Linguistics (ACL).

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems (NeurIPS).

J¨org Tiedemann. 2012. Parallel data, tools and interfaces in opus. In International Conference on Language Resources and Evaluation (LREC).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS).

David Vilar. 2018. Learning hidden unit contribution for adapting neural machine translation models. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL).