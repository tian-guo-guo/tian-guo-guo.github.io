---
layout:     post           # 使用的布局（不需要改）
title:      2018 Alibaba Submission to the WMT18 Parallel Corpus Filtering Task             # 标题 
subtitle:   2018 Alibaba Submission to the WMT18 Parallel Corpus Filtering Task   #副标题
date:       2020-06-25             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 技术


---

# 2018 Alibaba Submission to the WMT18 Parallel Corpus Filtering Task

![image-20200625152422085](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200720170730.jpg)

# Abstract

​		本文介绍了阿里巴巴机器翻译小组向WMT 2018并行语料库过滤共享任务提交的内容。 在评估平行语料库的质量时，研究了语料库的三个特征，即1）双语/翻译质量，2）单语质量和3）语料库多样性。 基于规则的方法和基于模型的方法都适用于对平行句子对进行评分。 最后的并行语料库过滤系统可靠，易于构建并适用于其他语言对。

# 1 Introduction

​		并行语料库是机器翻译和多语言自然语言处理的重要资源。除了数量和领域外，并行语料库的质量在MT系统培训中也非常重要（Koehn和Knowles，2017年; Khaylarllah和Koehn，2018年）。互联网包含大量的多语言资源，包括平行和可比较的句子（Resnik和Smith，2003年）。许多成功的机器翻译系统都是使用从网上抓取的语料库构建的。但是实际上，这种并行语料库可能非常嘈杂。并行语料库过滤的任务解决了清理嘈杂的并行语料库的问题。

​		在此任务中，我们可以将语料库清理任务分为三部分。首先，高质量的并行句子对应具有其目标句子精确翻译源句子的特性，反之亦然。在此任务中，我们尝试量化翻译对的质量（也称为双语评分）和句子对的准确性。其次，还应评估平行语料库的目标句子和/或源句子的质量。在这项工作中，目标附带句子在NMT中的重要性备受关注。

​		第三，如并行语料库过滤任务所述，参与者不应关注域相关性。我们需要关注所有领域，以便最终的MT系统能够得到广泛使用。因此，应在对并行语料库进行二次采样时评估多样性。最后，将并行语料库的三个特征组合起来以构建最终的干净语料库。

​		本文的结构如下：第2节介绍了我们在并行语料库过滤中使用的方法。第三部分详细说明了实验和结果。本节还详细介绍了用于构建基于模型的方法的数据集。结论在第4节中得出。

# 2 Parallel Sentence Pairs Scoring Methods

​		在本节中，将详细介绍三种评分/过滤方法。

## 2.1 Bilingual Quality Evaluation

​		在这里，我们描述了嘈杂的语料库过滤规则和两种翻译质量评估方法：（1）基于单词对齐的双语评分和（2）基于Bitoken CNN分类器的双语评分（Chen et al。，2016）。

**基于规则的过滤**

一系列启发式规则适用于过滤不良句子对。它们简单但有效，如下所述。

•源句子与目标句子的长度比。句子长度计算为记号/单词的数量。在我们的系统中，该比率设置为0.4到2.5。

•源令牌序列和目标令牌序列之间的编辑距离。较小的编辑距离表示源句子和目标句子非常相似。这种语料库极大地损害了NMT系统的性能（Khayrallah and Koehn，2018）。此外，编辑距离可以通过源和目标句子长度的平均长度进行归一化，这表示编辑距离比率。编辑距离和编辑距离比率都用于过滤源句子和目标句子相似的句子对。在我们的系统中，如果句子对的编辑距离小于2或编辑距离比小于0.1，则将删除该句子对。

•特殊令牌的一致性（Taghipour等，2010）。例如，高质量句子对在源句子和目标句子中都应包含相同的电子邮件地址（如果存在）。在此任务中，特殊令牌是电子邮件地址，URL和大阿拉伯数字
**基于单词对齐的双语评分**

单词对齐模型可用于评估双语句子对的翻译质量（Khadivi和Ney，2005； Taghipour等，2010； Ambati，2011）。受（Khadivi and Ney，2005）的启发，我们简化了原始算法，句子对的翻译分数如下：

![image-20200625152738504](/Users/suntian/Library/Application Support/typora-user-images/image-20200625152738504.png)

​		在等式（1）中，s和t分别表示源句子和目标句子，p（w 1 | w 2）表示单词翻译概率，a s2t表示源单词与目标单词对齐，m和n是源长度 和目标句子。

​		在此任务中，在WMT18新翻译任务提供的干净并行语料库上训练单词对齐模型。 我们使用快速对齐工具包（Dyer等，2013）来训练模型，并获得正向和反向单词翻译概率表。

​		该模型也称为对齐评分模型。

**基于Bitoken CNN分类器的双语评分**

​		在（Chen et al。，2016）的工作之后，建立了一个基于CNN的计分模型来评估翻译质量。
​		在此模型中，bitokens是从对齐的句子对中提取的。图1显示了如何从单词对齐的句子对中获得一个被咬住的序列。序列中的每个bitoken都被视为一个单词，而每个bitoken序列均被视为一个普通句子。然后将这些被咬的句子输入CNN分类器，以建立双语评分模型。对于每个候选句子对，此模型将给出两个概率：p pos和p neg，并且质量得分被视为得分bitoken = p pos-p neg。对于火车数据集，将从高质量语料库中获得的bitoken序列标记为正。对于负数训练数据，我们会基于干净数据手动构建一些嘈杂的数据（Lample等人，2017年），例如，保留干净并行语料库的目标副词，或随机删除源句或目标句的单词。因此，可以从这个不平行的语料库获得负的bitoken序列。

此评分模型也可以称为bitoken CNN评分模型。

![image-20200625152847768](https://tva1.sinaimg.cn/large/007S8ZIlly1gg4khamcz3j30eq0b6di0.jpg)

## 2.2 Monolingual Quality Evaluation Rule based Filtering

**基于规则的过滤**

一些规则适用于过滤源或目标端不好的句子对。这些规则是：

•太短（≤2个单词）或太长（> 80个单词）的句子长度将被删除。
•有效标记的比率与句子的长度成正比。在这里，有效令牌是包含相应语言字母的令牌。例如，有效的英语令牌应包含英文字母。在我们的系统中，如果句子的有效令牌率小于0.2，则过滤该句子。

•语言过滤。对于德语-英语平行语料库，源和目标句子的语言应为英语和德语。我们可以使用我们开发的语言检测工具1来检测句子的语言。如果源和目标方的语言不是德语和英语，则会过滤句子对。

​		语言模型评分我们使用语言模型来评估句子的质量。语言模型已成功用于选择域相关语料库（Yasuda等，2008； Moore和Lewis，2010）。此外，语言模型还可以用于过滤非语法数据（Denkowski等，2012； Allauzen等，2011），这适用于此任务。

​		在我们的语料库过滤系统中，我们专注于目标句子（即英语句子）的质量，因为它们在NMT中更为重要。首先，在WMT18提供的所有可用英语单语语料库的基础上，建立了一个大型语言模型。训练语料库使用上述一些规则进行清理。然后，标准化长度语言模型得分可以被视为单语质量得分。但是在实践中，此方法有一个缺点：对于包含稀有单词的好句子，它的得分较低。需要概括训练语料库以克服这种不足，例如，我们可以将LM训练语料库中出现次数少于10次的单词替换为其语音标签（Axelrod等人，2015）。最后，在通用语料库上重建语言模型。

## 2.3 Corpus Diversity

​		基于规则的过滤我们可以使用一个简单的规则来减少相似句子对的数量。 首先，应概括源句和目标句。 在我们的实验中，对于英语句子，通过删除除英语字母以外的所有字符来进行概括。 同样，进行了类似的操作以概括德语句子。 此后，如果某些句子对具有相同的广义源或目标句子，则将选择质量得分最高的句子对。

​		基于N-gram的多样性评分在此方法中，我们旨在子选择包含各种N-gram的语料库。 这样的语料库被认为是高度多样性的。 我们遵循（Ambati，2011;Bic¸ici和Yuret，2011）的工作，其动机是为n-gram权重引入特征衰减函数。 在我们的系统中，选择子集S 1 j−1后，下一个句子s j的多样性得分为：

![image-20200625153108739](https://tva1.sinaimg.cn/large/007S8ZIlly1gg4kjqkw15j30eg058aag.jpg)

​		其中，S 1 j-1代表所选句子的集合，其中包含第1到第（j-1）个句子，S是要选择的整个句子池。

​		f（s j | S 1 j-1）是在选择了语料库S 1 j-1的情况下句子s j的多样性分数。

​	NG（s j，n）是句子s j中大小为n的所有n-gram。 | NG（s j，n）|是NG（s j，n）的大小。

​	norm（s j）是感度s j的归一化因子，等于n = 1 N | NG（s j，n）|。

​		Freq（ng，S）是选择数据S中n-gram的频率。

​		λ是指数衰减超参数，在我们的实验中λ= 1。

​		等式（2）指示n元语法在池集合S和选择的集合S 1 j-1中通过其频率加权。所选集中ngram的频率越高，权重越低；池集中n-gram的频率越高，权重也越高。在实践中，首先，池S中的句子对按其质量得分（由双语和单语得分组合）以降序排序。然后，在双语主体的目标侧执行上述选择方法。

​		并行短语多样性评分基于N-gram的多样性评分通常用于选择具有高度多样性的单语句子。在这里，我们的目标是选择包含多种并行短语的双语语料库。通过这种语料库，MT模型将学习更多的翻译知识。

​		首先，我们使用快速对齐工具包来训练单词对齐模型。然后，可以使用Moses工具包提取语料库的短语表。接下来，我们可以使用最大匹配的方法从短语表中获得每个句子对的平行短语对。最后，按照基于N-gram的分集计分部分所述的方法，将相同的选择过程（其中N-gram被短语对代替）用于句子对的评分。在我们的系统中，词组长度小于7时效果最佳。

## 2.4 Methods Combination and corpus sampling

​		在我们的语料库过滤系统中，所有方法都组合到一条管道中。

​		首先，我们将所有双语和单语规则应用于过滤非常嘈杂的句子对。 然后，可以由上述相应模型产生两个双语得分和目标辅助语言模型得分。 将这三个分数分别归一化，然后线性组合以生成单个质量分数。 在这里，这些分数的权重是通过网格搜索方法选择的（Hsu et al。，2003）。 之后，我们按其对应的质量分数从高到低对句子对进行排序。 然后使用分集方法对语料库进行重新评分/重新排序。 最后，我们选择两套前N个句子对，它们总共包含1000万个单词和1亿个单词。

# 3 Experiments and Results

​		在本节中，我们指定实验设置并执行语料库筛选任务。

## 3.1 Corpus and Settings

​		选择数据池2由WMT18语料库筛选任务提供，其中包含约1亿个句子对。很吵。任务的参与者被要求从单词对中选择（a）一亿个单词和（b）一千万个单词的句子对。 3结果子集的质量取决于统计机器翻译（基于单词的摩西）和基于此数据训练的神经机器翻译系统（玛丽安）的BLEU分数。在我们的SMT和NMT实验中，我们使用了任务组织者4提供的SMT和NMT配置以及开发和测试集。

​		在建立比对评分模型时，在使用双语和单语过滤规则之后，从WMT18 EnglishGermanGermanGermanGerman新闻翻译任务提供的语料库中选择了4,337,154个句子对。接下来，使用快速对齐工具在干净的语料库上构建单词对齐模型，然后我们可以获得正向和反向单词翻译概率表。

​		在构建双向CNN评分模型时，将构建20,000个正标记的Bitoken序列和20,000个负标记的Bitoken序列。快速对齐工具包也在此使用。然后，我们使用CONTEXT 5工具包来训练CNN模型。 bitokens的嵌入向量由word2vec 6训练，每个向量的大小设置为200。

​		对于目标句子的质量评估，我们使用KenLM（Heaeld等人，2013）工具包来训练正常的和广义的LM。干净的训练语料库包含6000万个英语句子，这些句子是从WMT18新闻翻译任务提供的语料库中选择的。

## 3.2 Experimental Results

​		首先，通过训练SMT和NMT系统对包含大约1亿对句子的整个语料库进行了评估。最终的BLEU分数分别为21.21和7.8。这个实验表明整个语料库确实很吵。

![image-20200625154926352](https://tva1.sinaimg.cn/large/007S8ZIlly1gg4l2rm69gj30te0doq64.jpg)

​		其他实验结果详见表1。随机选择的语料表现也很差。 sys 1系统使用双语/单语规则和对齐方式评分，效果更好。我们用bitoken CNN方法代替对齐方式评分方法，然后构建sys 2系统。我们发现在句子对评分中，比对评分法和bitoken CNN方法非常相似。结果，两种方法都选择了很多句子对（子集中大约70％）。这两种方法在sys 3中结合使用，有一些改进。合并时，将原始分数标准化为间隔[0，1]，然后使用线性模型生成新分数。在sys 3系统中，比对得分和双位CNN得分的权重分别为0.4和0.6。

​		sys 4在sys 3的基础上引入了语言模式得分。对齐得分，bitcon CNN得分和语言模型得分的权重分别为0.4、0.6和0.8。它表明语言模型对于选择整洁的句子对很有用。

​		最后，基于sys 4，在sys 5中引入了语料库多样性过滤规则和评分。我们发现，多样性方法（在sys 5系统中仅使用并行短语多样性评分）可以很好地选择较小的子集语料库，例如。一千万个单词的语料库。对于大型子集语料库选择，几乎没有任何改善。我们将此归因于较大子集语料库的足够高的多样性。

# 4 Conclusions

​		在本文中，我们介绍了用于WMT 2018语料库筛选任务的语料库筛选系统。 在我们的系统中，从三个方面评估句子对：（1）双语翻译质量；（2）源句子和目标句子的单语质量；（3）子语料库的多样性。 我们的实验表明，所有方法都有助于建立更清晰的并行语料库。

# References

Alexandre Allauzen, H´elene Bonneau-Maynard, HaiSon Le, Aur´elien Max, Guillaume Wisniewski, Franc¸ois Yvon, Gilles Adda, Josep M Crego, Adrien Lardilleux, Thomas Lavergne, et al. 2011. Limsi@ wmt11. In Proceedings of the Sixth Workshop on Statistical Machine Translation, pages 309–315. Association for Computational Linguistics.

Vamshi Ambati. 2011. Active learning and crowdsourcing for machine translation in low resource scenarios. Ph.D. thesis, University of Southern California.

Amittai Axelrod, Philip Resnik, Xiaodong He, and Mari Ostendorf. 2015. Data selection with fewer words. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 58–65, Lisbon, Portugal. Association for Computational Linguistics.

Ergun Bic¸ici and Deniz Yuret. 2011. Instance selection for machine translation using feature decay algorithms. In Proceedings of the Sixth Workshop on Statistical Machine Translation, pages 272–283. Association for Computational Linguistics.

Boxing Chen, Roland Kuhn, George Foster, Colin Cherry, and Fei Huang. 2016. Bilingual methods for adaptive training data selection for machine translation. In Proc. of AMTA, pages 93–103.

Michael Denkowski, Greg Hanneman, and Alon Lavie. 2012. The cmu-avenue french-english translation system. In Proceedings of the Seventh Workshop on Statistical Machine Translation, pages 261–266. Association for Computational Linguistics.

Chris Dyer, Victor Chahuneau, and Noah A Smith. 2013. A simple, fast, and effective reparameterization of ibm model 2. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 644–648.

Alexandre Allauzen, H´elene Bonneau-Maynard, HaiSon Le, Aur´elien Max, Guillaume Wisniewski, Franc¸ois Yvon, Gilles Adda, Josep M Crego, Adrien Lardilleux, Thomas Lavergne, et al. 2011. Limsi@ wmt11. In Proceedings of the Sixth Workshop on Statistical Machine Translation, pages 309–315. Association for Computational Linguistics.

Vamshi Ambati. 2011. Active learning and crowdsourcing for machine translation in low resource scenarios. Ph.D. thesis, University of Southern California.

Amittai Axelrod, Philip Resnik, Xiaodong He, and Mari Ostendorf. 2015. Data selection with fewer words. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 58–65, Lisbon, Portugal. Association for Computational Linguistics.

Ergun Bic¸ici and Deniz Yuret. 2011. Instance selection for machine translation using feature decay algorithms. In Proceedings of the Sixth Workshop on Statistical Machine Translation, pages 272–283. Association for Computational Linguistics.

Boxing Chen, Roland Kuhn, George Foster, Colin Cherry, and Fei Huang. 2016. Bilingual methods for adaptive training data selection for machine translation. In Proc. of AMTA, pages 93–103.

Michael Denkowski, Greg Hanneman, and Alon Lavie. 2012. The cmu-avenue french-english translation system. In Proceedings of the Seventh Workshop on Statistical Machine Translation, pages 261–266. Association for Computational Linguistics.

Chris Dyer, Victor Chahuneau, and Noah A Smith. 2013. A simple, fast, and effective reparameterization of ibm model 2. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 644–648.