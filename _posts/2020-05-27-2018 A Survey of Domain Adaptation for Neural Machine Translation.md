---
layout:     post           # 使用的布局（不需要改）
title:      2018 A Survey of Domain Adaptation for Neural Machine Translation           # 标题 
subtitle:   A Survey of Domain Adaptation for Neural Machine Translation Domains #副标题
date:       2020-05-27             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 领域适应

---

# A Survey of Domain Adaptation for Neural Machine Translation

![image-20200716144454719](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716145257.png)

# Abstract

​		神经机器翻译（Neural MachineTranslation，NMT）是一种基于深度学习的机器翻译方法，在大规模并行语料库的情况下，它能提供最先进的翻译性能。 尽管高质量的领域规范翻译在现实世界中至关重要，但领域规范语料库通常稀缺或根本不存在，因此普通NMT在这种情况下表现不佳。 **域自适应利用域外平行语料库和单语语料库进行域内翻译，对于域规范翻译非常重要**。 本文综述了NMT领域自适应技术的最新进展。

# 1 Introduction

​		神经机器翻译(NMT)（Cho et al.，2014；Sutskever et al.，2014；Bahdanau et al.，2015）允许对翻译系统进行端到端的训练，而不需要处理单词对齐，翻译规则和复杂的解码算法，这些都是统计机器翻译(SMT)系统的特征（Koehn et al.，2007）。 NMT在资源丰富的场景下产生了最先进的翻译性能（Bojar et al.，2017；Nakazawa et al.，2017）。 然而，目前，规模足够的高质量平行语料库仅适用于少数几种语言对，如与英语和几种欧洲语言对配对的语言。 此外，对于每种语言对，领域专用语料库的大小和可用领域的数量都是有限的。 因此，对于大多数语言对和领域，只有很少或没有平行语料库可用。 众所周知，在低资源场景下，vanilla SMT和NMT在领域规范翻译方面表现较差（Duh等人，2013年；Sennrich等人，2013年；Zoph等人，2016年；Koehn和Knowles，2017年）。

​		高质量领域特定机器翻译(MT)系统的需求很高，而通用机器翻译的应用有限。 此外，通用翻译系统通常性能较差，因此开发特定领域的翻译系统非常重要（Koehn and Knowles，2017）。 ==利用域外平行语料库和域内单语语料库改进域内翻译被称为域适应翻译（Wang et al.，2016；Chu et al.，2018）==。 例如，**汉英专利领域平行语料库有100万个句子对（Goto et al.，2013），而口语领域平行语料库只有20万个句子（Cettolo et al.，2015）。 MT通常在资源贫乏或领域不匹配的场景中表现较差，因此重要的是利用口语领域数据和专利领域数据（Chu et al.，2017）。 此外，对于口语领域，有包含数百万句子的单语语料库，这也是可以利用的（Sennrich et al.，2016b)**。

​		面向SMT的领域适配研究很多，主要可以分为两大类:以数据为中心和以模型为中心。 以数据为中心的方法侧重于从基于语言模型(LM)的领域外并行语料库中选择训练数据（Moore and Lewis，2010；Axelrod et al.，2011；Duh et al.，2013；Hoang and Simaan，2014；Durrani et al.，2015；Chen et al.，2016）或生成伪并行数据（Utiyama and Isahara，2003；Wang et al.，2014；Chu，2015；Wang et al.，2016；Marie and Fujita，2017）。 以模型为中心的方法在模型级别（Sennrich et al.，2013；Durrani et al.，2015；Imamura and Sumita，2016）或实例级别（Matsoukas et al.，2009；Foster et al.，2010；Shah et al.，2010；Rousseau et al.，2011；Zhou et al.，2015）内插域内和域外模型。 然而，由于SMT和NMT的不同特点，许多针对SMT开发的方法并不能直接应用于NMT。

​		NMT领域适配是一个比较新的问题，已经引起了研究界的广泛关注。 近两年来，NMT已经成为最流行的MT方法，许多领域自适应技术被提出和评估。 这些研究要么借用以往SMT研究的思想并将这些思想应用于NMT，要么为NMT开发独特的方法。 尽管NMT的领域适应发展迅速，但没有一个单一的汇编总结和分类所有的方法。 由于这样的研究将极大地造福于社区，我们在这篇论文中介绍了所有主要的NMT领域适配技术。 有针对NMT的调查论文（Neubig，2017；Koehn，2017）； 但是，他们侧重于一般的NMT和更多样的题目。 已经从计算机视觉（Csurka，2017）和机器学习（Pan and Yang，2010；Weiss et al.，2016）的角度进行了领域适应调查。 然而，NMT还没有做过这样的调查。 据我们所知，这是NMT领域适配的首次全面调查。

​		本文中，类似于SMT，我们将面向NMT的领域适配分为两大类:以数据为中心和以模型为中心。 以数据为中心的类别侧重于正在使用的数据，而不是用于域适应的专用模型。 所使用的数据可以是域内单语语料库（Zhang and Zong，2016b；Cheng et al.，2016；Currey et al.，2017；Domhan and Hieber，2017），合成语料库（Sennrich et al.，2016b；Zhang and Zong，2016b；Park et al.，2017）或平行语料库（Chu et al.，2017；Sajjad et al.，2017；Britz et al.，2017；Wang et al.，2017a；van der Wees et al.，2017）。 另一方面，以模型为中心的类别关注专门用于域自适应的NMT模型，其可以是训练目标（Luong and Manning，2015；Sennrich et al.，2016b；Servan et al.，2016；Freitag and Al-Onaizan，2016；Wang et al.，2017b；Chen et al.，2017a；Varga，2017；Dakwale and Monz，2017；Chu et al.，2017；Miceli Barone et al.，2017），NMT体系结构（Kobus et al.，2016；Güulc Ehre et al.，2015；Britz et al.，2017）或解码算法（Güulc Ehre et al.，2015；Dakwale and Monz，2017； 这两个类别的概述如图1所示。 注意，由于以模型为中心的方法也使用单语或平行语料库，这两类之间存在重叠。

![image-20200716144751506](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716144751.png)

​		本文的其余部分结构如下:我们首先简要介绍了NMT，并描述了NMT中低资源域和低资源语言差异的原因（第2节）； 接下来，我们简要回顾了SMT领域自适应技术的历史发展（第3节）； 在这些背景知识的基础上，详细介绍和比较了NMT领域自适应方法（第4节）； 然后，我们介绍了在真实单词场景中NMT的领域自适应，这对MT的实际应用至关重要（第5节）； 最后，我们对本领域未来的研究方向提出了自己的看法（第6节），并对本文进行了总结（第7节）。

# 2 Neural Machine Translation

![image-20200716144810366](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716144810.png)

​		NMT是一种用于从一种语言翻译到另一种语言的端到端方法，它依赖于深度学习来训练翻译模型（Cho et al.，2014；Sutskever et al.，2014；Bahdanau et al.，2015）。 具有注意力的编码器-解码器模型（Bahdanau等人，2015）是最常用的NMT架构。 这种模式也被称为RNNSearch。 图2描述了RNNsearch模型（Bahdanau等人，2015），它接受输入语句x={x1，。。。，x n}及其翻译y={y1，。。。，y m}。 转换生成为:

![image-20200526164307859](/Users/suntian/Library/Application Support/typora-user-images/image-20200526164307859.png)

​		其中？是一组参数，m是y中的全部字数，y j是当前预测字，y<j是先前预测字。 假设我们有一个平行语料库C，由一组平行句对（x，y）组成。 训练目标是最小化互熵损失L W.R.T？:

![image-20200716144631212](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716144631.png)		该模型由三个主要部分组成，即编码器，解码器和注意模型。 编码器使用嵌入机制将单词转换为它们的连续空间表示。 这些嵌入本身并不包含单词之间的关系和它们在句子中的位置的信息。 使用递归神经网络(RNN)层，在这种情况下是门控递归单元(GRU)，可以实现这一点。 一个RNN保持一个隐藏状态（也称为记忆或历史），这允许它为一个单词生成一个连续的空间表示，给定所有已经看到的过去的单词。 有两个GRU层，分别编码前向和后向信息。 每个字x i通过将前向隐藏状态h i和后向隐藏状态h i级联为h i=[h i；h i]来表示。 这样，源句子x={x1，。。。，x n}可以表示为h={h1，。。。，h n}。 通过使用前向和后向递归信息，在给定单词前后的所有单词的情况下，获得单词的连续空间表示。

​		解码器在概念上是一个RNN语言模型(RNNLM)，具有自己的嵌入机制，一个GRU层用来记忆先前生成的单词，一个softmax层用来预测一个目标单词。 编码器和解码器通过使用关注机制来耦合，该关注机制计算由编码器产生的反复表示的加权平均值，从而充当软对准机制。 这个加权平均向量，也称为上下文或注意力向量，与先前预测的单词一起被馈送到解码器GRU以产生一个表示，该表示被传递到softmax层以预测下一个单词。 在等式中，用于解码器的时间j的RNN隐藏状态s j计算如下:

![image-20200716144647241](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716145204.png)

​		其中f是GRU的激活函数，sj-1是前一RNN隐藏状态，yj-1是前一字，cj是上下文向量。 c j通过使用对准权重a ji计算为编码器隐藏状态h={h1，。。。，h n}的加权和:

![image-20200716144658420](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716144658.png)

​		其中a是对位置i周围的输入和位置J处的输出的匹配水平进行评分的对准模型。 softmax层包含maxout层，它是具有max池的前馈层。 maxout层采用解码器GRU生成的循环隐藏状态，前一个字和上下文向量来计算一个整体表示，并将其馈送到一个简单的softmax层:

![image-20200716144710001](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716144710.png)

​		由于编码器，解码器和注意力模型中的大量参数，需要大量的并行语料库来训练NMT系统以避免过度筛选。 这是低资源域和语言NMT的主要瓶颈。

# 3 Domain Adaptation for SMT

​		在SMT中，人们提出了许多领域自适应方法，以克服在特定领域和语言中缺乏大量数据的问题。 大多数SMT域适配方法可以大致分为两大类:

## 3.1 Data Centric

此类别侧重于使用现有的域内数据选择或生成与域相关的数据。

i）当存在来自其他领域的有效并行语料库时，主要思想是使用从域内和域外数据训练的模型对域外数据进行评分，并使用所得评分的截止阈值从域外数据中选择训练数据。LMs（Moore and Lewis，2010； Axelrod等人，2011年； Duh等人，2013年），以及联合模型（Hoang和Simaan，2014年； Durrani等人，2015），以及更近的卷积神经网络(CNN)模型（Chen等人，2016）可以用来给句子打分。

ii）当平行语料不足时，也有研究使用信息检索（Utiyama和Isahara，2003），自我增强（Lambert等人，2011）或平行词嵌入（Marie和Fujita）生成伪平行句子。 ，2017）。 除了句子生成之外，还有一些研究生成单语言n-gram（Wang等，2014）和并行短语对（Chu，2015； Wang等，2016）。

SMT中大多数基于数据中心的方法都可以直接应用于NMT。 但是，这些方法大多数都采用与NMT不相关的数据选择或生成标准。 因此，这些方法只能在NMT方面实现适度的改进（Wang等，2017a）。

## 3.2 Model Centric

此类别重点在于对来自不同域的模型进行插值。

i）模型级插值。训练了分别对应于每个语料库的几种SMT模型，例如LM，翻译模型和重新排序模型。然后将这些模型组合以获得最佳性能（Foster和Kuhn，2007; Bisazza等人，2011; Niehues和Waibel，2012; Sennrich等人，2013; Durrani等人，2015; Imamura和Sumita，2016） 。

ii）实例级别插值。实例加权已应用于几种自然语言处理（NLP）域适应任务（Jiang和Zhai，2007），尤其是SMT（Matsoukas等，2009； Foster等，2010； Shah等，2012； Mansour和Ney，2012； Zhou等，2015）。他们首先通过使用规则或统计方法为权重对每个实例/域评分，然后通过为每个实例/域赋予权重来训练SMT模型。另一种方法是通过数据重新采样对语料库加权（Shah等人，2010； Rousseau等人，2011）。

对于NMT，像SMT一样，已经提出了几种插值模型/数据的方法。对于模型级插值，最相关的NMT技术是模型集成（Jean等，2015）。对于实例级插值，最相关的方法是在NMT目标函数中分配权重（Chen等，2017a; Wang等，2017b）。但是，SMT和NMT的模型结构完全不同。 SMT是几种独立模型的组合；相比之下，NMT本身就是一个不可或缺的模型。因此，这些方法大多数不能直接应用于NMT。

# 4 Domain Adaptation for NMT

## 4.1 Data Centric

### 4.1.1. Using Monolingual Corpora

​		与SMT不同的是，域内单语数据不能直接作为常规NMT的LM，为此已进行了许多研究。 Güulccoverehre等人。 （2015年）在单语数据上训练RNNLM，并融合RNNLM和NMT模型。 科里等人。 （2017）将目标单语数据复制到源端，并将复制的数据用于训练NMT。 Domhan和Hieber(2017)提出将目标单语数据用于具有LM和NMT多任务学习的解码器。 Zhang和Zong(2016b)使用源端单语数据通过多任务学习来增强NMT编码器，用于预测翻译和重新排序的源句子。 Cheng等人。 （2016）通过使用NMT作为自动编码器重建单语数据，将源单语数据和目标单语数据同时用于NMT。

### 4.1.2 Synthetic Parallel Corpora Generation

​		由于NMT本身具有学习LMs的能力，目标单语数据也可以用于NMT系统在反译目标句子后加强译码器以生成合成的平行语料库（Sennrich et al.，2016b)。 图3显示了该方法的流程图。 还表明，合成数据生成对于使用目标侧单语数据（Sennrich等人，2016c)，源侧单语数据（Zhang和Zong，2016b)或两者（Park等人，2017）的域适应非常有效。

![image-20200716144827897](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716144827.png)

### 4.1.3 Using Out-of-Domain Parallel Corpora

​		采用域内和域外并行语料库，训练一个既能提高域内翻译质量又不降低域外翻译质量的混合域机器翻译系统是理想的。 我们将这些努力归类为多域方法，这些方法已经成功地为NMT开发。 此外，还提出了面向NMT的SMT数据选择思想。

​		Chu等人的多域方法。 （2017）最初是由Sennrich等人提出的。 (2016a)，它使用标记来控制NMT的礼貌性。 该方法的概述如图6中虚线部分所示。 在这种方法中，多个域的语料库通过两个小的修改连接起来:

​		•在各语料库的源句中添加域标记“<2domain>”。 这使NMT解码器为特定域生成句子。

​		•对较小语料库进行过采样，以便培训程序对每个领域给予同等重视。

​		Sajjad等人。 （2017）进一步比较训练一个多领域系统的不同方法。 特别地，他们比较简单地连接多领域语料库的连接，在每个领域语料库上迭代训练NMT系统的staking，选择与领域内数据接近的一组领域外数据的selection，以及集成独立训练的多个NMT模型的ensemble。 他们认为，在域内数据上调整级联系统可获得最佳性能。 布里茨等人。 （2017年）将多域法与一种判别法进行比较（详见4.2.2节）。 结果表明，判别法的性能优于多域法。

​		数据选择正如SMT一节（第3.1节）中提到的，SMT中的数据选择方法可以略微提高NMT的性能，因为它们的数据选择标准与NMT不是很相关（Wang et al.，2017a)。 为了解决这个问题，Wang等人。 (2017a)利用NMT中源句子的内部嵌入，利用句子嵌入相似度从域外数据中选择与域内数据接近的句子（图4）。 Van der Wees等人。 （2017）提出了一种动态数据选择方法，其中他们在不同的训练历元之间改变所选择的训练数据子集用于NMT。 结果表明，基于域内相似度的训练数据逐渐减少的方法具有最好的性能。

![image-20200716144857822](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716144857.png)

​		虽然所有的以数据为中心的NMT方法在原理上是相互补充的，但目前还没有尝试将这些方法结合起来的研究，这被认为是未来的一个方向。

## 4.2 Model Centric

### 4.2.1 Training Objective Centric

​		本节中的方法改变了获得最优域内训练目标的训练函数或过程。

​			实例/代价权重NMT中实例权重的主要挑战在于NMT不是线性模型或线性模型的组合，这意味着实例权重不能直接集成到NMT中。 在NMT中只有一个关于实例加权的工作（Wang et al.，2017b)。 他们为目标函数设置了一个权重，这个权重通过一个域内LM和一个域外LM从交叉熵中学习（Axelrod等人，2011）（图5）。 Chen et al。 (2017a)使用域分类器修改NMT成本函数。 域分类器的输出概率转换为域权重。 此分类器使用开发数据进行训练。 近日，王等人。 （2018）为NMT提出了一个句子选择和加权的联合框架。

![image-20200716144915149](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716145447.png)

​		微调微调是域适应的常规方式（Luong和Manning，2015；Sennrich等，2016b；Servan等，2016；Freitag和Al-Onaizan，2016）。 在该方法中，在资源丰富的域外语料库上训练NMT系统直到收敛，然后在资源贫乏的域内语料库上调整其参数。 通常，优化应用于域内并行语料库。 Varga等人。 （2017）将其应用于从可比语料库中提取的平行句上。 通过从可比语料库中提取平行数据，可比语料库已被广泛用于SMT（Chu，2015）。 为了防止域内数据调优后的域外转换退化，Dakwale和Monz(2017)提出了一种基于知识蒸馏的域外模型分布的扩展调优（辛顿等人，2015）。

![image-20200716144933626](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716145456.png)

​		混合微调该方法是多域和非域调谐方法的组合（图6）。 培训程序如下:

​		1.在域外数据上训练NMT模型，直到收敛。

​		2.在域内和域外数据的混合上（通过对域内数据进行过采样）恢复从步骤1开始的NMT模型的训练，直到收敛。

​		混合调谐解决了域内数据小而导致的过度调谐问题。 与训练多域模型相比，用域外数据训练一个好的模型更容易。 一旦我们获得了良好的模型参数，我们就可以使用这些参数对混合域数据进行优化，从而为域内模型获得更好的性能。 此外，混合调优比多域调优更快，因为训练域外模型比训练多域模型收敛更快，而多域模型在混合域数据调优时收敛也非常快。 Chu等人。 （2017年）表明，混合调优比多域调优和混合调优效果更好。 此外，混合调谐具有与Dakw和Monz(2017)中的集成方法类似的效果，不会降低域外转换性能。

​		正则化Barone等人。 （2017年）还解决了调谐期间的过度调谐问题。 他们解决这一问题的策略是探索正则化技术，如辍学和L2正则化。 此外，他们还提出调谐，这是一个变体的退出，以正规化。 我们认为混合调谐和正则化技术是相辅相成的。

### 4.2.2 Architecture Centric

​		本节中的方法更改NMT体系结构以适应域。

​		深度融合与域内单语数据自适应的一种技术是训练用于NMT解码器的域内RNNLM，并将其与NMT模型相结合（也称为融合）(Güulcco/Ehre et al.，2015）。 融合可以是浅的，也可以是深的。 形式上，深度融合指示LM和NMT被集成为单个解码器（即，将RNNLM集成到NMT架构中）。 浅融合表示LM和NMT的分数被一起考虑（即，用RNNLM模型重新记录NMT模型）。
​		在深度融合中，RNNLM和NMT的译码器通过级联它们的隐藏状态来集成。 在计算下一个字的输出概率时，模型会调整为使用RNNLM和NMT模型的隐藏状态。 Domhan和Hieber(2017)提出了一种与深度聚变方法类似的方法(Güulccoverehre等人，2015）。 然而，不同于单独训练RNNLM和NMT模型(Güulccoverehre et al.，2015），Domhan和Hieber(2017)联合训练RNNLM和NMT模型。

​		利用多领域语料库中信息多样性的领域鉴别器，Britz等人。 （2017年）提出一种判别方法。 在他们的判别方法中，他们在编码器的基础上增加了一个前馈网络(FFNN)作为判别器，利用注意力来预测源句子的域。 鉴别器与NMT网络联合优化。 图7显示了该方法的概述。

​		域控制除了使用域令牌来控制域之外，Kobus等人。 （2016）提出在NMT的嵌入层添加词级特征来控制域。 特别是，他们给每个单词附加了一个域标记。 他们还提出了一种基于术语频率-逆文档频率（tf-idf）的方法来预测输入句子的域标记。

![image-20200716145014990](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716145507.png)

### 4.2.3 Decoding Centric

​		以译码为中心的方法关注于域自适应译码算法，与其他以模型为中心的方法本质上是互补的。

​		浅融合浅融合是一种在大型单语语料库上训练LM，然后将它们与先前训练过的NMT模型相结合的方法(Güulccoverehre et al.，2015）。 在浅融合(Güulccoehre et al.，2015）中，NMT模型生成的下一个单词假设由NMT和RNNLM概率的加权和重新编码（图8）。

​		集成Freitag和Al-Onaizan(2016)提出集成域外模型和调整后的域内模型。 他们的动机与Dakwale和Monz(2017)的工作完全相同，即防止域内数据调优后的域外转换退化。

​		神经格搜索。 （2017）提出了一种基于堆栈的字格解码算法，而字格是由SMT生成的（Dyer et al.，2008）。 在他们的域自适应实验中，他们表明基于堆栈的译码比常规译码要好。

# 5 Domain Adaptation in Real-World Scenarios

​		需要根据特定的场景采用领域自适应方法。 例如，当域外数据中存在一些伪并行的域内数据时，首选选句； 当只有额外的单语数据时，可以采用LM和NMT融合。 在许多情况下，域外并行数据和单语域内数据都可用，使得不同方法的组合成为可能。 Chu等人。 （2018年）开展一项研究，将混合调谐（Chu等人，2017年）应用于合成并行数据（Sennrich等人，2016b)，该研究显示出比任何一种方法都更好的性能。 因此，我们在本文中不推荐任何特定的技术，而是建议读者为自己的场景选择最好的方法。

​		上述领域自适应研究大多假设数据的领域是给定的。 然而，在诸如在线翻译引擎之类的实用视图中，用户输入的句子的域并没有给出。 在这种情况下，预测输入句子的域对于良好的翻译是至关重要的。 为了解决这个问题，SMT中的一种常用方法是对领域进行高成本的分类，然后使用相应的模型在分类领域中翻译输入句子（Huck等人，2015）。 Xu等人。 （2007）为汉英翻译任务进行领域分类。 分类器使用LM插值和词汇相似性，对整个文档进行操作，而不是对单个句子进行操作。 哈克等人。 （2015）扩展了Xu等人的工作。 （2007年）。 它们使用LMs和最大熵分类器来预测目标域。 Banerjee等人。 （2010）使用tf-idf特征构建支持向量机分类器。 分类是在单个句子的水平上进行的。 Wang等人。 （2012）依赖于具有各种基于短语的特征的平均感知器分类器。

​		对于NMT，Kobus等人。 （2016）提出了一种NMT域控制方法，通过在NMT的词嵌入层添加域标签或特征。 它们采用内部分类器来区分域信息。 Li等人。 （2016）建议使用测试语句作为查询在训练数据中搜索相似语句，然后使用检索到的训练语句调整NMT模型以翻译测试语句。 Farajian等人。 （2017）遵循Li等人的策略。 （2016），但提出基于输入句子和检索到的句子的相似度动态设置学习算法的超参数（即学习速率和历元数），用于更新NMT模型。 图9显示了输入域未知场景中MT的域适配的概述。

![image-20200716145037266](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200716145037.png)

# 6 Future Directions

## 6.1 Domain Adaptation for State-of-art NMT Architectures

​		自从基于RNN的NMT的成功（Cho et al.，2014；Sutskever et al.，2014；Bahdanau et al.，2015）以来，NMT的其他架构已经被开发。 一个代表性的架构是基于CNN的NMT（Gehring等人，2017）。 与基于RNN的模型相比，基于CNN的模型可以在训练过程中完全并行计算，并且更易于优化。 另一个代表性的架构是Transformer，它仅基于注意力（Vaswani等人，2017）。 研究表明，基于CNN的NMT和变压器的性能显著优于Wu等人的基于RNN的NMT模型。 （2016年）在翻译质量和速度两个方面。 然而，目前大多数针对NMT的域适应研究都是基于RNN模型（Bahdanau等人，2015）。 针对这些最新的NMT模型的域自适应技术的研究显然是未来的一个重要方向。

## 6.2 Domain Specific Dictionary Incorporation

​		如何利用词典，知识库等外部知识进行外语教学是一个很大的研究问题。 在领域适配中，使用领域规范词典是一个非常关键的问题。 从实践的角度来看，许多翻译公司已经创建了领域规范词典，但没有创建领域规范语料库。 如果我们能够研究一种使用领域规范词典的好方法，将会极大地促进MT的实际应用。 有一些研究尝试使用词典进行NMT，但其用法仅限于帮助低频度或罕见词的翻译（Arthur et al.，2016；Zhang and Zong，2016a)。 Arcan和Buitelaar(2017)使用领域规范词典进行术语翻译，但它们只是应用了Luong等人提出的未知词替换方法。 （2015年），它遭受嘈杂的关注。

## 6.3 Multilingual and Multi-Domain Adaptation

​		在同一语言对中使用域外平行语料库可能并不总是可能的，因此使用其他语言的数据很重要（Johnson et al.，2016）。 这种学习方法被称为跨语迁移学习，即在多种语言之间传递NMT模型参数。 多语言模型依赖于参数共享，有助于提高低资源语言的翻译质量，尤其是在目标语言相同的情况下（Zoph et al.，2016）。 有一些研究训练了多语言（Firat et al.，2016；Johnson et al.，2017）或多域模型（Sajjad et al.，2017），但没有一个研究试图将多个语言对和多个域打包成一个单一的翻译系统。 即使存在同一语言对中的域外数据，同时使用多语言和多域数据也有可能提高翻译性能。 因此，我们认为针对NMT的多语言，多领域适配可以是未来的另一个方向。 Chu和Dabre（2018）针对这一课题进行了初步研究。

## 6.4 Adversarial Domain Adaptation and Domain Generation

​		生成对抗网络是一类用于无监督机器学习的人工智能算法，由Goodfellow等人（2014）提出。 对抗性方法已经在域自适应中变得流行（Ganin等人，2016），其通过相对于域鉴别器的对抗性目标最小化近似域差异距离（Tzeng等人，2017）。 它们已被应用于计算机视觉和机器学习中的领域自适应任务（Tzeng et al.，2017；Motiian et al.，2017；Volpi et al.，2017；Zhao et al.，2017；Pei et al.，2018）。 最近，一些对抗方法开始被引入到一些NLP任务（Liu et al.，2017；Chen et al.，2017b)和NMT（Britz et al.，2017）中。

​		现有的大多数方法侧重于从通用域到特定域的适应。 在实际场景中，训练数据和测试数据具有不同的分布，目标域有时是看不到的。 欧文等人。 （2013）分析此类情景下的翻译错误。 领域泛化旨在将从标记的源领域获得的知识应用到看不见的目标领域（Li et al.，2018）。 它提供了一种在真实世界MT中匹配训练数据和测试数据分布的方式，这可能是NMT领域适配的一个未来趋势。

# 7 Conclusion

​		NMT领域适配是一个较新但又非常重要的研究课题，以促进MT实用化。 在本文中，我们对主要在过去两年中开发的技术进行了首次全面的回顾。 本文比较了NMT的域适配技术和SMT的域适配技术，后者是近二十年来的主要研究领域。 此外，展望了未来的研究方向。 将NMT中的领域自适应技术与一般的NLP，计算机视觉和机器学习技术相结合是我们今后的工作。 我们希望这篇调查论文能够对NMT领域适配的研究起到重要的推动作用。

# 8 References

Mihael Arcan and Paul Buitelaar. 2017. Translating domain-speciﬁc expressions in knowledge bases with neural machine translation. CoRR, abs/1709.02184.

Philip Arthur, Graham Neubig, and Satoshi Nakamura. 2016. Incorporating discrete translation lexicons into neural machine translation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1557–1567, Austin, Texas, November. Association for Computational Linguistics.

Amittai Axelrod, Xiaodong He, and Jianfeng Gao. 2011. Domain adaptation via pseudo in-domain data selection.

In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing, pages 355–362, Edinburgh, Scotland, U.K.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In In Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015), San Diego, USA, May. International Conference on Learning Representations.

Pratyush Banerjee, Jinhua Du, Baoli Li, Sudip Naskar, Andy Way, and Josef Genabith. 2010. Combining multidomain statistical machine translation models using automatic classiﬁers. In The Ninth Conference of the Association for Machine Translation in the Americas, Denver, Colorado.

Arianna Bisazza, Nick Ruiz, and Marcello Federico. 2011. Fill-up versus interpolation methods for phrase-based SMT adaptation. In IWSLT, pages 136–143. ISCA.

Ondˇrej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Shujian Huang, Matthias Huck, Philipp Koehn, Qun Liu, Varvara Logacheva, Christof Monz, Matteo Negri, Matt Post, Raphael Rubino, Lucia Specia, and Marco Turchi. 2017. Findings of the 2017 conference on machine translation (WMT17). In Proceedings of the Second Conference on Machine Translation, pages 169–214, Copenhagen, Denmark, September. Association for Computational Linguistics.

Denny Britz, Quoc Le, and Reid Pryzant. 2017. Effective domain mixing for neural machine translation. In Proceedings of the Second Conference on Machine Translation, pages 118–126, Copenhagen, Denmark, September. Association for Computational Linguistics.

M Cettolo, J Niehues, S St¨uker, L Bentivogli, R Cattoni, and M Federico. 2015. The iwslt 2015 evaluation campaign. In Proceedings of the Twelfth International Workshop on Spoken Language Translation (IWSLT).

Boxing Chen, Roland Kuhn, George Foster, Colin Cherry, and Fei Huang. 2016. Bilingual methods for adaptive training data selection for machine translation. In The Twelfth Conference of The Association for Machine Translation in the Americas, pages 93–106, Austin, Texas.

Boxing Chen, Colin Cherry, George Foster, and Samuel Larkin. 2017a. Cost weighting for neural machine translation domain adaptation. In Proceedings of the First Workshop on Neural Machine Translation, pages 40–46, Vancouver.

Xinchi Chen, Zhan Shi, Xipeng Qiu, and Xuanjing Huang. 2017b. Adversarial multi-criteria learning for chinese word segmentation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1193–1203, Vancouver, Canada. Association for Computational Linguistics.

Yong Cheng, Wei Xu, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. Semi-supervised learning for neural machine translation. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1965–1974, Berlin, Germany, August. Association for Computational Linguistics.

Kyunghyun Cho, Bart van Merri¨enboer, C¸alar G¨ulc¸ehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using rnn encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734, Doha, Qatar, October. Association for Computational Linguistics.

Chenhui Chu and Raj Dabre. 2018. Multilingual and multi-domain adaptation for neural machine translation. In Proceedings of the 24st Annual Meeting of the Association for Natural Language Processing (NLP 2018), pages 909–912, Okayama, Japan, Match.

Chenhui Chu, Raj Dabre, and Sadao Kurohashi. 2017. An empirical comparison of domain adaptation methods for neural machine translation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, Vancouver, Canada, July. Association for Computational Linguistics.

Chenhui Chu, Raj Dabre, and Sadao Kurohashi. 2018. A comprehensive empirical comparison of domain adaptation methods for neural machine translation. Journal of Information Processing (JIP), 26(1):1–10.

Chenhui Chu. 2015. Integrated parallel data extraction from comparable corpora for statistical machine translation. Doctoral Thesis, Kyoto University.

Gabriela Csurka.

2017.

Domain adaptation for visual applications:

A comprehensive survey.

CoRR,

abs/1702.05374.

Anna Currey, Antonio Valerio Miceli Barone, and Kenneth Heaﬁeld. 2017. Copied monolingual data improves low-resource neural machine translation. In Proceedings of the Second Conference on Machine Translation, pages 148–156, Copenhagen, Denmark, September. Association for Computational Linguistics.

Praveen Dakwale and Christof Monz. 2017. Fine-tuning for neural machine translation with limited degradation across in- and out-of-domain data. In Proceedings of the 16th Machine Translation Summit (MT-Summit 2017), pages 156–169.

Tobias Domhan and Felix Hieber. 2017. Using target-side monolingual data for neural machine translation through multi-task learning. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1500–1505, Copenhagen, Denmark, September. Association for Computational Linguistics.

Kevin Duh, Graham Neubig, Katsuhito Sudoh, and Hajime Tsukada. 2013. Adaptation data selection using neural language models: Experiments in machine translation. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 678–683, Soﬁa, Bulgaria, August.

Nadir Durrani, Hassan Sajjad, Shaﬁq Joty, Ahmed Abdelali, and Stephan Vogel. 2015. Using joint models for domain adaptation in statistical machine translation. In Proceedings of MT Summit XV, pages 117–130, Miami, FL, USA.

Christopher Dyer, Smaranda Muresan, and Philip Resnik. 2008. Generalizing word lattice translation. In Proceedings of ACL-08: HLT, pages 1012–1020, Columbus, Ohio, June. Association for Computational Linguistics.

M. Amin Farajian, Marco Turchi, Matteo Negri, and Marcello Federico. 2017. Multi-domain neural machine translation through unsupervised adaptation. In Proceedings of the Second Conference on Machine Translation, pages 127–137, Copenhagen, Denmark, September. Association for Computational Linguistics.

Orhan Firat, Kyunghyun Cho, and Yoshua Bengio. 2016. Multi-way, multilingual neural machine translation with a shared attention mechanism. In NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016, pages 866–875.

George Foster and Roland Kuhn. 2007. Mixture-model adaptation for smt. In Proceedings of the Second Workshop on Statistical Machine Translation, StatMT ’07, pages 128–135, Stroudsburg, PA, USA. Association for Computational Linguistics.

George Foster, Cyril Goutte, and Roland Kuhn. 2010. Discriminative instance weighting for domain adaptation in statistical machine translation. In Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing, pages 451–459, Cambridge, MA.

Markus Freitag and Yaser Al-Onaizan.

2016.

Fast domain adaptation for neural machine translation. arXiv

preprint arXiv:1612.06897.

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Franc¸ois Laviolette, Mario Marchand, and Victor Lempitsky. 2016. Domain-adversarial training of neural networks. Journal of Machine Learning Research, 17(59):1–35.

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. Convolutional sequence to sequence learning. CoRR, abs/1705.03122.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. In Advances in neural information processing systems, pages 2672–2680.

Isao Goto, Ka-Po Chow, Bin Lu, Eiichiro Sumita, and Benjamin K. Tsou. 2013. Overview of the patent machine translation task at the ntcir-10 workshop. In Proceedings of the 10th NTCIR Conference, pages 260–286, Tokyo, Japan, June. National Institute of Informatics (NII).

C¸aglar G¨ulc¸ehre, Orhan Firat, Kelvin Xu, Kyunghyun Cho, Lo¨ıc Barrault, Huei-Chi Lin, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2015. On using monolingual corpora in neural machine translation. CoRR, abs/1503.03535.

Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. In NIPS Deep Learning and Representation Learning Workshop.

Cuong Hoang and Khalil Sima’an. 2014. Latent domain translation models in mix-of-domains haystack. In Proceedings of the 25th International Conference on Computational Linguistics: Technical Papers, pages 19281939, Dublin, Ireland.

Matthias Huck, Alexandra Birch, and Barry Haddow. 2015. Mixed-domain vs. multi-domain statistical machine translation. Proceedings of MT Summit XV, 1:240–255.

Kenji Imamura and Eiichiro Sumita. 2016. Multi-domain adaptation for statistical machine translation based on feature augmentation. In Proceedings of the 12th Conference of the Association for Machine Translation in the Americas, Austin, Texas, USA.

Ann Irvine, John Morgan, Marine Carpuat, Hal Daume III, and Dragos Munteanu. 2013. Measuring machine translation errors in new domains. Transactions of the Association for Computational Linguistics, 1:429–440. S´ebastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. On using very large target vocabulary for neural machine translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1–10, Beijing, China, July. Association for Computational Linguistics.

Jing Jiang and ChengXiang Zhai. 2007. Instance weighting for domain adaptation in NLP. In Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics, pages 264–271, Prague, Czech Republic. Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda B. Vi´egas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. Google’s multilingual neural machine translation system: Enabling zero-shot translation. CoRR, abs/1611.04558. Melvin Johnson, Mike Schuster, Quoc Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernand a Vigas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2017. Google’s multilingual neural machine translation system: Enabling zero-shot translation. Transactions of the Association for Computational Linguistics, 5:339–351.

Huda Khayrallah, Gaurav Kumar, Kevin Duh, Matt Post, and Philipp Koehn. 2017. Neural lattice search for domain adaptation in machine translation. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 20–25, Taipei, Taiwan, November. Asian Federation of Natural Language Processing.

Catherine Kobus, Josep Crego, and Jean Senellart. 2016. Domain control for neural machine translation. arXiv preprint arXiv:1612.06140.

Philipp Koehn and Rebecca Knowles. 2017. Six challenges for neural machine translation. In Proceedings of the First Workshop on Neural Machine Translation, pages 28–39, Vancouver, August. Association for Computational Linguistics.

Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, Chris Dyer, Ondrej Bojar, Alexandra Constantin, and Evan Herbst. 2007. Moses: Open source toolkit for statistical machine translation. In Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics Companion Volume Proceedings of the Demo and Poster Sessions, pages 177–180, Prague, Czech Republic, June. Association for Computational Linguistics. Philipp Koehn. 2017. Neural machine translation. CoRR, abs/1709.07809.

Patrik Lambert, Holger Schwenk, Christophe Servan, and Sadaf Abdul-Rauf. 2011. Investigations on translation model adaptation using monolingual data. In Proceedings of the Sixth Workshop on Statistical Machine Translation, WMT ’11, pages 284–293, Stroudsburg, PA, USA. Association for Computational Linguistics. Xiaoqing Li, Jiajun Zhang, and Chengqing Zong. 2016. One sentence one model for neural machine translation.

CoRR, abs/1609.06490.

Ya Li, Mingming Gong, Xinmei Tian, Tongliang Liu, and Dacheng Tao. 2018. Domain generalization via conditional invariant representations. In The Thirty-Second AAAI Conference on Artiﬁcial Intelligence.

Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2017. Adversarial multi-task learning for text classiﬁcation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1–10, Vancouver, Canada. Association for Computational Linguistics.

Minh-Thang Luong and Christopher D Manning. 2015. Stanford neural machine translation systems for spoken language domains. In Proceedings of the 12th International Workshop on Spoken Language Translation, pages 76–79, Da Nang, Vietnam, December.

Thang Luong, Ilya Sutskever, Quoc Le, Oriol Vinyals, and Wojciech Zaremba. 2015. Addressing the rare word problem in neural machine translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 11–19, Beijing, China, July. Association for Computational Linguistics.

Saab Mansour and Hermann Ney. 2012. A simple and effective weighted phrase extraction for machine translation adaptation. In The 9th International Workshop on Spoken Language Translation, Hong Kong.

Benjamin Marie and Atsushi Fujita. 2017. Efﬁcient extraction of pseudo-parallel sentences from raw monolingual data using word embeddings. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 392–398, Vancouver, Canada, July. Association for Computational Linguistics.

Spyros Matsoukas, Antti-Veikko I. Rosti, and Bing Zhang. 2009. Discriminative corpus weight estimation for machine translation. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 708–717, Singapore.

Antonio Valerio Miceli Barone, Barry Haddow, Ulrich Germann, and Rico Sennrich. 2017. Regularization techniques for ﬁne-tuning in neural machine translation. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1489–1494, Copenhagen, Denmark, September. Association for Computational Linguistics.

Robert C Moore and William Lewis. 2010. Intelligent selection of language model training data. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 220–224, Uppsala, Sweden.

Saeid Motiian, Quinn Jones, Seyed Mehdi Iranmanesh, and Gianfranco Doretto. 2017. Few-shot adversarial domain adaptation. CoRR, abs/1711.02536.

Toshiaki Nakazawa, Shohei Higashiyama, Chenchen Ding, Hideya Mino, Isao Goto, Hideto Kazawa, Yusuke Oda, Graham Neubig, and Sadao Kurohashi. 2017. Overview of the 4th workshop on asian translation. In Proceedings of the 4th Workshop on Asian Translation (WAT2017), pages 1–54, Taipei, Taiwan, November. Asian Federation of Natural Language Processing.

Graham Neubig. 2017. Neural machine translation and sequence-to-sequence models: A tutorial. CoRR, abs/1703.01619.

Jan Niehues and Alex H. Waibel. 2012. Detailed analysis of different strategies for phrase table adaptation in smt. In Proceedings of the Conference of the Association for Machine Translation in the Americas (AMTA), San Diego, US-CA.

Sinno Jialin Pan and Qiang Yang. 2010. A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10):1345–1359, October.

Jaehong Park, Jongyoon Song, and Sungroh Yoon. 2017. Building a neural machine translation system using only synthetic parallel data. CoRR, abs/1704.00253.

Zhongyi Pei, Zhangjie Cao, Mingsheng Long, and Jianmin Wang. 2018. Multi-adversarial domain adaptation. Anthony Rousseau, Fethi Bougares, Paul Del´eglise, Holger Schwenk, and Yannick Est`eve. 2011. Liums systems for the iwslt 2011 speech translation tasks. In International Workshop on Spoken Language Translation, San Francisco, USA.

Hassan Sajjad, Nadir Durrani, Fahim Dalvi, Yonatan Belinkov, and Stephan Vogel. 2017. Neural machine translation training in a multi-domain scenario. In Proceedings of the Twelfth International Workshop on Spoken Language Translation (IWSLT), Tokyo, Japan.

Rico Sennrich, Holger Schwenk, and Walid Aransa. 2013. A multi-domain translation model framework for statistical machine translation. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 832–840, Soﬁa, Bulgaria.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016a. Controlling politeness in neural machine translation via side constraints. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 35–40, San Diego, California, June. Association for Computational Linguistics.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016b. Improving neural machine translation models with monolingual data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 86–96, Berlin, Germany, August. Association for Computational Linguistics. Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016c. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany, August. Association for Computational Linguistics.

Christophe Servan, Josep Crego, and Jean Senellart. 2016. Domain specialization: a post-training domain adaptation for neural machine translation. arXiv preprint arXiv:1612.06141.

Kashif Shah, Lo¨ıc Barrault, and Holger Schwenk. 2010. Translation model adaptation by resampling. In Proceedings of the Joint Fifth Workshop on Statistical Machine Translation and MetricsMATR, pages 392–399. Kashif Shah, Lo¨ıc Barrault, and Holger Schwenk. 2012. A general framework to weight heterogeneous parallel data for model adaptation in statistical machine translation. In Proceedings of the Conference of the Association for Machine Translation in the Americas (AMTA), San Diego, US-CA.

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to sequence learning with neural networks. In Proceedings of the 27th International Conference on Neural Information Processing Systems, pages 3104–3112, Cambridge, MA, USA. MIT Press.

Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. 2017. Adversarial discriminative domain adaptation.

CoRR, abs/1702.05464.

Masao Utiyama and Hitoshi Isahara. 2003. Reliable measures for aligning japanese-english news articles and sentences. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics, pages 72–79, Sapporo, Japan, July. Association for Computational Linguistics.

Marlies van der Wees, Arianna Bisazza, and Christof Monz. 2017. Dynamic data selection for neural machine translation. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1400–1410, Copenhagen, Denmark, September. Association for Computational Linguistics.

Adam Csaba Varga. 2017. Domain adaptation for multilingual neural machine translation. Master Thesis, Saarlandes University.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 5998–6008. Curran Associates, Inc.

Riccardo Volpi, Pietro Morerio, Silvio Savarese, and Vittorio Murino. 2017. Adversarial feature augmentation for unsupervised domain adaptation. CoRR, abs/1711.08561.

Wei Wang, Klaus Macherey, Wolfgang Macherey, Franz Och, and Peng Xu. 2012. Improved domain adaptation for statistical machine translation. In Proceedings of AMTA, San Diego, California, USA.

Rui Wang, Hai Zhao, Bao-Liang Lu, Masao Utiyama, and Eiichiro Sumita. 2014. Neural network based bilingual language model growing for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 189–195, Doha, Qatar, October. Association for Computational Linguistics.

Rui Wang, Hai Zhao, Bao-Liang Lu, Masao Utiyama, and Eiichiro Sumita. 2016. Connecting phrase based statistical machine translation adaptation. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 3135–3145, Osaka, Japan, December. The COLING 2016 Organizing Committee.

Rui Wang, Andrew Finch, Masao Utiyama, and Eiichiro Sumita. 2017a. Sentence embedding for neural machine translation domain adaptation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 560–566, Vancouver, Canada, July. Association for Computational Linguistics.

Rui Wang, Masao Utiyama, Lemao Liu, Kehai Chen, and Eiichiro Sumita. 2017b. Instance weighting for neural machine translation domain adaptation. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1482–1488, Copenhagen, Denmark.

Rui Wang, Masao Utiyama, Andrew Finch, Lemao Liu, Kehai Chen, and Eiichiro Sumita. 2018. Sentence selection and weighting for neural machine translation domain adaptation. IEEE/ACM Transactions on Audio, Speech, and Language Processing.

Karl Weiss, Taghi M. Khoshgoftaar, and DingDing Wang. 2016. A survey of transfer learning. Journal of Big Data, 3(1):9, May.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. CoRR, abs/1609.08144.

Jia Xu, Yonggang Deng, Yuqing Gao, and Hermann Ney. 2007. Domain dependent statistical machine translation.

In MT Summit, Copenhagen, Denmark.

Jiajun Zhang and Chengqing Zong. 2016a. Bridging neural machine translation and bilingual dictionaries. CoRR, abs/1610.07272.

Jiajun Zhang and Chengqing Zong. 2016b. Exploiting source-side monolingual data in neural machine translation.

In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 15351545, Austin, Texas, November. Association for Computational Linguistics.

Han Zhao, Shanghang Zhang, Guanhang Wu, Jo˜ao P. Costeira, Jos´e M. F. Moura, and Geoffrey J. Gordon. 2017.

Multiple source domain adaptation with adversarial training of neural networks. CoRR, abs/1705.09684.

Xinpeng Zhou, Hailong Cao, and Tiejun Zhao. 2015. Domain adaptation for SMT using sentence weight. In Chinese Computational Linguistics and Natural Language Processing Based on Naturally Annotated Big Data, pages 153–163, Guangzhou, China.

Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. 2016. Transfer learning for low-resource neural machine translation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016, pages 1568–1575.