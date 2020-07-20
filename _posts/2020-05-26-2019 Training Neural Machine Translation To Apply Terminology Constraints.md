---
layout:     post           # 使用的布局（不需要改）
title:      2019 Training Neural Machine Translation To Apply Terminology Constraints           # 标题 
subtitle:   2019 Training Neural Machine Translation To Apply Terminology Constraints          #副标题
date:       2020-05-26             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 术语

---

# 2019 Training Neural Machine Translation To Apply Terminology Constraints

![image-20200526095616272](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200720164847.jpg)

# Abstract

​		本文提出了一种**在运行时将自定义术语注入神经机器翻译的新颖方法**。This paper proposes a novel method to inject custom terminology into neural machine translation at run time. 以前的工作主要是**对解码算法提出修改，以约束输出，使其包含运行时提供的目标项**。Previous works have mainly proposed modiﬁcations to the decoding algorithm in order to constrain the output to include run-time-provided target terms. 这些受限解码方法虽然很有效，但在推理步骤中增加了显著的计算开销，而且如本文所示，在实际条件下测试时可能会变得脆弱。While being effective, these constrained decoding methods add, however, signiﬁcant computational overhead to the inference step, and, as we show in this paper, can be brittle when tested in realistic conditions.  在本文中，我们通过训练一个神经MT系统来学习当提供输入时如何使用自定义术语来解决这个问题。 对比实验表明，我们的方法不仅比一种最先进的约束译码实现更有效，而且与无约束译码同样快。In this paper we approach the problem by training a neural MT system to learn how to use custom terminology when provided with the input. Comparative experiments show that our method is not only more effective than a state-of-the-art implementation of constrained decoding, but is also as fast as constraint-free decoding.

# 1 Introduction

​		尽管如今神经机器翻译(NMT)达到了很高的质量，但它的输出仍然不能满足翻译行业日常处理的许多特定领域。 **虽然NMT已经证明，可以从域内并行或单语数据的可用性中受益于学习领域规范术语（Farajian et al.，2018），但它并不是一个普遍适用的解决方案，因为一个领域通常可能太窄，并且缺乏数据，因此这种自举技术无法发挥作用**。While NMT has shown to beneﬁt from the availability of in-domain parallel or monolingual data to learn domain speciﬁc terms (Farajian et al., 2018), it is not a universally applicable solution as often a domain may be too narrow and lacking in data for such bootstrapping techniques to work. 因此，大多数多语言内容提供商为其所有域维护由语言专家创建的术语。For this reason, most multilingual content providers maintain terminologies for all their domains which are created by language specialists. 例如，为了表示输入的《大白鲨》是一部恐怖电影，会有一个条目，如《大白鲨(en)Lo Squalo(it)》，应该翻译为Lo Squalo e`unfilm Pauroso。 虽然翻译存储器可以被看作是NMT领域适应的现成的训练数据，但术语数据库（短期基础）更难处理，而且在提出在运行时将领域术语集成到NMT的方法方面已经有了重要的工作。

​		**约束译码是解决这一问题的主要方法。 简而言之，它使用源端与输入端匹配的术语条目的目标端作为解码时间约束。 Chatterjee等人提出了约束译码和各种改进**。 Constrained decoding is the main approach to this problem. In short, it uses the target side of terminology entries whose source side match the input as decoding-time constraints.（2017），Hasler等人。 （2018），Hokamp和Liu（2017）等。 Hokamp和Liu（2017）最近介绍了网格**波束搜索(GBS)算法，该算法对每个提供的词法约束使用单独的波束**。 the grid beam search (GBS) algorithm which uses a separate beam for each supplied lexical constraint.然而，该解决方案在约束的数量上指数地增加解码过程的运行时间复杂度。 Post和Vilar(2018)最近建议使用**动态波束分配(DBA)技术，该技术将计算开销减少到恒定因子，与约束的数量无关**。dynamic beam allocation (DBA) technique that reduces the computational overhead to a constant factor, independent from the number of constraints. 在实践中，Post和Vilar(2018)中报道的结果表明，使用DBA的约束解码是有效的，但当使用波束大小为5时，仍然会导致翻译时间增加3倍。

​		**在本文中，我们把约束译码问题看作是在训练时间学习术语的复制行为的问题。 通过修改神经MT的训练过程，完全消除了推理时的任何计算开销**。 In this paper we address the problem of constrained decoding as that of learning a copy behaviour of terminology at training time. By modifying the training procedure of neural MT we are completely eliminating any computational overhead at inference time. **具体地说，NMT模型经过训练，学习如何使用作为源句子附加输入的术语条目。 术语翻译作为内联注释插入，附加的输入流（所谓的源因子）用于通知运行文本和目标术语之间的切换。 我们给出了从两个术语词典中抽取的术语进行英德翻译的实验**。Speciﬁcally, the NMT model is trained to learn how to use terminology entries when they are provided as additional input to the source sentence. Term translations are inserted as inline annotations and additional input streams (so called source factors) are used to signal the switch between running text and target terms. 由于我们不假设术语在训练时间是可用的，所以我们的所有测试都是在零镜头设置下执行的，即使用未见的术语术语。 我们将我们的方法与Post和Vilar(2018)提出的带DBA的约束解码的有效实现方法进行了比较。

​		而我们的目标与Gu等人的目标相似。 （2017）（教NMT使用翻译记忆）和Pham等人的。 （2018）（探索实施复制行为的网络架构），**我们提出的方法与标准transformer NMT模型（Vaswani等人，2017）一起工作，该模型提供了包含运行文本和内联注释的混合输入。 这将术语功能与NMT体系结构解耦，当最新的体系结构不断变化时，这一点尤为重要**。the method we propose works with a standard transformer NMT model (Vaswani et al., 2017) which is fed a hybrid input containing running text and inline annotations. This decouples the terminology functionality from the NMT architecture, which is particularly important as the state-of-the-art architectures are continuously changing.

# 2 Model

​		**我们提出了一种集成的方法，在这种方法中，当输入中提供了目标术语时，MT模型在训练时学习如何使用术语。 特别是，模型应该学会偏向翻译以包含所提供的术语，即使它们在训练数据中没有被观察到**。 We propose an integrated approach in which the MT model learns, at training time, how to use terminology when target terms are provided in input. In particular, the model should learn to bias the translation to contain the provided terms, even if they were not observed in the training data.**我们对传统的MT输入进行了扩充，以包含一个源句子以及一个针对该句子触发的术语条目列表，特别是那些源边与句子匹配的术语条目**。 We augment the traditional MT input to contain a source sentence as well as a list of terminology entries that are triggered for that sentences, speciﬁcally those whose source sides match the sentence.**虽然人们已经探索了许多不同的方法来增加翻译输入的额外信息，但是我们选择在源句子中集成术语信息作为内联注释，或者将目标术语添加到源句子中，或者直接用目标术语替换原术语**。 While many different ways have been explored to augment MT input with additional information, we opt here for integrating terminology information as inline annotations in the source sentence, by either appending the target term to its source version, or by directly replacing the original term with the target one.**我们在源句中添加一个额外的并行流来表示这种“语码转换”。 当附加翻译时，此流有三个可能的值:0表示源单词（默认值），1表示源术语，2表示目标术语。 两个被测试的变体，一个保留了术语的源端，另一个丢弃了源端，用表1中的一个例子说明了这两个变体**。We add an additional parallel stream to signal this “code-switching” in the source sentence. When the translation is appended this stream has three possible values: 0 for source words (default), 1 for source terms, and 2 for target terms. The two tested variants, one in which the source side of the terminology is retained and one in which it is discarded, are illustrated with an example in Table 1.

![image-20200526102926158](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200720165017.jpg)

表1:用于生成sourcetarget训练数据的两种替代方式，包括源中的目标术语以及指示源词（0），源术语（1）和目标术语（2）的因素。Table 1: The two alternative ways used to generate sourcetarget training data, including target terms in the source and factors indicating source words (0), source terms (1), and target terms (2).

## 2.1 Training data creation训练数据创建

​		由于我们不修改原始的sequence-tosequence NMT体系结构，网络可以从训练数据的扩充中学习术语的使用。 **我们假设当一个术语条目（t s，t t）在源中被注释时，目标方t t出现在参考文献中时，模型将在训练时学会使用所提供的术语**。We hypothesize that the model will learn to use the provided terminology at training time if it holds true that when a terminology entry (t s , t t ) is annotated in the source, the target side t t is present in the reference. **因此，我们仅对满足此标准的术语对进行标注**。 For this reason we annotate only terminology pairs that ﬁt this criterion.**实验中使用的术语库相当大，并且对所有匹配项进行注释导致了大多数包含术语注释的句子。 由于我们希望模型在基线，无约束条件下表现得同样好，所以我们通过随机忽略一些匹配来限制注释的数量**。The term bases used in the experiments are quite large and annotating all matches leads to most of the sentences containing term annotations. Since we want to model to perform equally well in a baseline, constraint-free condition, we limit the number of annotations by randomly ignoring some of the matches.

​		**一个句子s可能包含来自一个术语库的多个匹配项，但是我们在源术语重叠的情况下保留最长的匹配项**。 A sentence s may contain multiple matches from a term base, but we keep the longest match in the case of overlapping source terms.**此外，当检查句子中的术语匹配时，我们应用近似匹配来允许术语中的一些形态变化**。Moreover, when checking for matches of a term inside a sentence, we apply approximate matching to allow for some morphological variations in the term **在我们当前的实现中，我们使用一个简单的字符序列匹配，例如，允许将基本单词形式视为匹配，即使它们在影响中或作为化合物的一部分**。In our current implementation, we use a simple character sequence match, allowing for example for base word forms to be considered matches even if they are inﬂected or as part of compounds.

# 3 Experiments

## 3.1 Evaluation setting

​		**并行数据和NMT架构**   我们在WMT 2018英德新闻翻译任务1上测试我们的方法，通过在Europarl和新闻评论数据上训练模型，**总计220万句。 基线按原样使用此训练数据**。We test our approach on the WMT 2018 English-German news translation tasks 1 , by training models on Europarl and news commentary data, for a total 2.2 million sentences. The baselines use this train data as is. **对于其他条件，包含术语注释的句子被添加，大约相当于原始数据的10%**。For the other conditions sentences containing term annotations are added amounting to approximately 10% of the original data. **我们限制了添加的数据量（通过随机忽略一些匹配的术语），因为我们希望模型在没有提供术语作为输入时同样工作良好。 注意，这些句子来自原始数据池，因此没有引入实际的新数据**。We limit the amount of data added (by randomly ignoring some of the matched terms) as we want the model to work equally well when there are no terms provided as input. Note that these sentences are from the original data pool and therefore no actual new data is introduced.

​		我们使用**Moses**（Koehn et al.，2007）对语料库进行标记化，并对**32k**个标记进行**联合源和目标BPE编码**（Sennrich et al.，2016）。 **我们使用上一节中描述的源因子流，这些源因子流以简单的方式从字流广播到BPE流**。We use the source factor streams described in the previous section which are broadcast from word streams to BPE streams in a trivial way. **我们将这个附加流的三个值嵌入到大小为16的向量中，并将它们串联到相应的子字嵌入中**。We embed the three values of this additional stream into vectors of size 16 and concatenate them to the corresponding sub-word embeddings. 我们使用具有**两个编码层和两个解码层的变压器网络**（Vaswani等人，2017年），共享源和目标嵌入，并使用**Sockeye工具包**（Hieber等人，2018年）训练所有模型（参见附录中的完整训练配置）。 WMT newstest 2013开发集用于计算停止标准，**所有模型都经过至少50个，最多100个历元的训练**。 在相同的条件下，**使用5的波束大小**，我们将我们提出的两种方法，**即逐附加的训练和逐替换的训练train-by-appending and train-by-replace**，**与Sockeye中可用的Post和Vilar(2018)的约束译码算法进行了比较**。

​		**术语数据库**      我们提取了两个可公开使用的术语库，Wiktionary和IATE的英语-德语部分。 2为了避免虚假匹配，我们筛选出出现在前500个最常见英语单词中的条目以及单个字符条目。 通过确保在源端没有重叠，我们将术语基分成训练和测试列表。We extracted the English-German portions of two publicly available term bases, Wiktionary and IATE. 2 In order to avoid spurious matches, we ﬁltered out entries occurring in the top 500 most frequent English words as well as single character entries. We split the term bases into train and test lists by making sure there is no overlap on the source side.

## 3.2结果

​		我们分别对WMT newstest 2013/2017 as开发(dev)和测试集进行评估，并使用Wiktionary和IATE的测试部分对测试集进行注释。 3我们选择参考文献中使用该术语的句子，因此复制行为是正确的。 用维基词库提取的**测试集包含727个句子和884个词条**，用维基词库提取的测试集包含**414个句子和452个词条**。

​		表2显示了结果。 我们**报告解码速度，BLEU分数，以及术语使用率**，计算为术语翻译在输出中产生的次数占术语注释总数的百分比。

![image-20200526104644947](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200720165034.jpg)

​		表2:提供了正确的术语条目的系统的术语使用率百分比和BLEU分数，这些系统与源和目标完全匹配。 我们还提供了P99延迟数（即99%的翻译是在给定的秒数内完成的）。 并且在p值小于0.05时表示比基线系统明显更好和更差的系统。

​		**术语使用率和解码速度**    我们的第一个观察结果是，<u>基线模型已经以76%的高比率使用术语翻译。 按追加训练设置达到约90%的术语使用率，而按替换训练达到更高的使用率（93%，94%），表明完全消除源术语会更加强烈地强制执行复制行为</u>。 **所有这些都优于约束译码，约束译码在Wiktionary上达到99%，而在IATE上仅为82%**。 4

​		**第二，我们的两个设置的解码速度都与基线的解码速度相当，因此比约束解码(CD)的平移速度快三倍**。 这是一个重要的区别，因为解码时间增加三倍会阻碍术语在latencycritical应用程序中的使用。 注意，解码时间是通过在单个GPU P3 AWS实例上运行批处理大小为1的实验来测量的。 5

​		**翻译质量**    **令人惊讶的是，我们观察到显著的差异W.R.T BLEU评分。 请注意，术语只影响句子的一小部分，并且大多数时候基线已经包含了所需的术语，因此在此测试集中不可能出现高的BLEU变化。 约束译码不会导致BLEU的任何变化，除了在具有5的小波束尺寸的情况下在IATE上的减小**。 Surprisingly, we observe signiﬁcant variance w.r.t BLEU scores. Note that the terminologies affect only a small part of a sentence and most of the times the baseline already contains the desired term, therefore high BLEU variations are impossible on this test set. Constrained decoding does not lead to any changes in BLEU, other than a decrease on IATE with a small beam size of 5.然而，所有逐次训练模型都显示BLEU增加(+0.2到+0.9)，特别是术语使用率较低的逐次训练模型。 当检查这些方法的错误时，我们观察到这样的情况，即约束解码改变翻译以适应一个术语，即使该术语的变体已经在翻译中，如表3的Festzunehmen/Festnahme示例（并且有时即使已经使用了相同的术语）。 仔细观察以前的约束译码文献，可以发现大多数的评价都与本文不同:数据集只包含参考文献中包含术语的句子，而且基线也不能产生术语。 这是一个理想的设置，我们相信很少，如果有，模拟现实世界的应用程序。

​		**在我们的方法中，我们观察到了另一个令人惊讶的积极行为，而约束解码不能处理:在某些情况下，我们的模型生成术语库提供的术语翻译的形态变体。 在此之后，我们设置了一个额外的实验，通过扩展前一个实验集来包括目标侧的近似匹配（与2.1节中解释的训练中的近似匹配相同）**。We observed an additional surprisingly positive behavior with our approach which constrained decoding does not handle: in some cases, our models generate morphological variants of terminology translations provided by the term base. Following up on this we set up an additional experiment by extending the previous set to also include approximate matches on the target side (identical to the approximate match in training explained in Section 2.1).

![image-20200526110644580](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200720164928.jpg)

表4:提供术语条目的系统的机器翻译结果，显示精确的源匹配和近似的参考匹配。 当P值小于0.05时，表示系统明显比基线差。

​		表4显示了这些结果。 我们注意到，这个测试用例对于约束解码和按替换训练来说已经更加困难，这很可能是因为删除了原始源端内容。 另一方面，trainby-append的表现仍优于基线，而受限解码显示BLEU分数显著降低0.9-1.3个BLEU点。 表3中的Humanithumanit示例代表了在源匹配项（其目标侧需要受到影响）的情况下，由约束解码引入的错误。

# 4 Conclusion

​		虽然以往的神经MT研究大多致力于术语与约束译码的集成，但我们提出了一种黑箱方法，即直接训练一个通用的神经MT体系结构来学习如何使用运行时提供的外部术语。 我们在零镜头设置下进行了实验，表明复制行为是在测试时触发的，而这些词在训练中从未见过。 与约束解码相反，我们还观察到该方法展示了术语的灵活使用，因为在某些情况下，术语以其提供的形式使用，而在其他情况下则执行灵活。

​		据我们所知，在神经MT的约束译码算法空间中，没有任何现有的工作比我们的方法具有更好的速度与性能权衡，我们相信这使得它特别适合于生产环境。

# References

Rajen Chatterjee, Matteo Negri, Marco Turchi, Marcello Federico, Lucia Specia, and Fr´ed´eric Blain. 2017. Guiding neural machine translation decoding with external knowledge. In Proceedings of the Second Conference on Machine Translation, pages 157–168, Copenhagen, Denmark. Association for Computational Linguistics.

Josep Maria Crego, Jungi Kim, Guillaume Klein, Anabel Rebollo, Kathy Yang, Jean Senellart, Egor Akhanov, Patrice Brunelle, Aurelien Coquard, Yongchao Deng, Satoshi Enoue, Chiyo Geiss, Joshua Johanson, Ardas Khalsa, Raoum Khiari, Byeongil Ko, Catherine Kobus, Jean Lorieux, Leidiana Martins, Dang-Chuan Nguyen, Alexandra Priori, Thomas Riccardi, Natalia Segal, Christophe Servan, Cyril Tiquet, Bo Wang, Jin Yang, Dakun Zhang, Jing Zhou, and Peter Zoldan. 2016. Systran’s pure neural machine translation systems. CoRR, abs/1610.05540.

M. Amin Farajian, Nicola Bertoldi, Matteo Negri, Marco Turchi, and Marcello Federico. 2018. Evaluation of Terminology Translation in Instance-Based Neural MT Adaptation. In Proceedings of the 21st Annual Conference of the European Association for Machine Translation, pages 149–158, Alicante, Spain. European Association for Machine Translation.

Jiatao Gu, Yong Wang, Kyunghyun Cho, and Victor O. K. Li. 2017. Search engine guided nonparametric neural machine translation. In Proceedings of the Thirty-Second AAAI Conference on Artiﬁcial Intelligence, pages 5133–5140, New Orleans, Louisiana, USA. Association for the Advancement of Artiﬁcial Intelligence.

Eva Hasler, Adri`a Gispert, Gonzalo Iglesias, and Bill Byrne. 2018. Neural machine translation decoding with terminology constraints. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 506–512. Association for Computational Linguistics.

Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton, and Matt Post. 2018. The sockeye neural machine translation toolkit at AMTA 2018. In Proceedings of the 13th Conference of the Association for Machine Translation in the Americas (Volume 1: Research Papers), pages 200–207. Association for Machine Translation in the Americas.

Chris Hokamp and Qun Liu. 2017. Lexically constrained decoding for sequence generation using grid beam search. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 August 4, Volume 1: Long Papers, pages 1535–1546.

Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, Chris Dyer, Ondrej Bojar, Alexandra Constantin, and Evan Herbst. 2007. Moses: Open source toolkit for statistical machine translation. In Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics Companion Volume Proceedings of the Demo and Poster Sessions, pages 177–180. Association for Computational Linguistics.

Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective approaches to attention-based neural machine translation. CoRR, abs/1508.04025.

Ngoc-Quan Pham, Jan Niehues, and Alexander H.

Waibel. 2018. Towards one-shot learning for rareword translation with external experts. In Proceedings of the 2nd Workshop on Neural Machine Translation and Generation, NMT@ACL 2018, Melbourne, Australia, July 20, 2018, pages 100–109.

Matt Post and David Vilar. 2018. Fast lexically constrained decoding with dynamic beam allocation for neural machine translation. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers), pages 1314–1324.

Rico Sennrich, Barry Haddow, and Alexandra Birch.

2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 17151725. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 6000–6010.

# NMT Sockeye train parameters

encoder-config:

​		act_type: relu 

​		attention_heads: 8 

​		conv_config: null 

​		dropout_act: 0.1 

​		dropout_attention: 0.1 

​		dropout_prepost: 0.1 

​		dtype: float32 

​		feed_forward_num_hidden: 2048 

​		lhuc: false 

​		max_seq_len_source: 101 

​		max_seq_len_target: 101 

​		model_size: 512 

​		num_layers: 2 

​		positional_embedding_type: fixed 

​		postprocess_sequence: dr 

​		preprocess_sequence: n 

​		use_lhuc: false



decoder config:

​		act_type: relu 

​		attention_heads: 8 

​		conv_config: null 

​		dropout_act: 0.1 

​		dropout_attention: 0.1 

​		dropout_prepost: 0.1 

​		dtype: float32 

​		feed_forward_num_hidden: 2048 

​		max_seq_len_source: 101 

​		max_seq_len_target: 101 

​		model_size: 512 

​		num_layers: 2 

​		positional_embedding_type: fixed 

​		postprocess_sequence: dr 

​		preprocess_sequence: n



config_loss: !LossConfig

​		label_smoothing: 0.1 

​		name: cross-entropy 

​		normalization_type: valid 

​		vocab_size: 32302



config_embed_source: !

​		EmbeddingConfig 

​		dropout: 0.0 

​		dtype: float32 

​		factor_configs: null 

​		num_embed: 512

​		num_factors: 1 

​		vocab_size: 32302



config_embed_target: !

​		EmbeddingConfig 

​		dropout: 0.0 

​		dtype: float32 

​		factor_configs: null 

​		num_embed: 512 

​		num_factors: 1 

​		vocab_size: 32302