---
layout:     post           # 使用的布局（不需要改）
title:      2019 BART- Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension             # 标题 
subtitle:   BART Pretraining model   #副标题
date:       2020-05-23             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 预训练模型


---

# 2019 BART- Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

# Abstract

​		我们提出了**BART，一种用于预训练序列到序列模型的去噪自动编码器**。 <u>BART的训练方法是:（1）用任意的噪声函数破坏文本corrupting text with an arbitrary noising function；（2）学习一个模型来重建原始文本learning a model to reconstruct the original text</u>。 它使用了一个标准的基于转换器的神经机器翻译体系结构，尽管它很简单，但它可以被看作是对BERT（由于双向编码器），GPT（带有从左到右解码器）和许多其他较新的预训练方案的推广。 我们评估了许多去噪方法，通过随机调整原始句子的顺序和使用一种新颖的填充方案（用单个掩码标记替换文本）来获得最佳性能。 BART在文本生成时特别有效，但在理解任务中也很有效。 它与RoBERTa的表现相匹配，在GLUE和SQuAD上提供了相当的培训资源，在一系列抽象对话，问题回答和总结任务上实现了新的最先进的结果，获得了高达6 Rouge的收益。 **BART还为机器翻译提供了比反向翻译系统增加1.1BLEU的功能，只需进行目标语言预训练**。 我们还报告了在BART框架内复制其他预训练方案的消融实验，以更好地测量哪些因素最影响Fluffence结束任务的表现。

# 1 Introduction

​		**自监督方法**在广泛的NLP任务中取得了显著的成功（Mikolov et al.，2013；Peters et al.，2018；Devlin et al.，2019；Joshi et al.，2019；Yang et al.，2019；Liu et al.，2019）。 最成功的方法是掩蔽语言模型的变体，它是去噪自动编码器，训练它重建文本，其中随机子集的单词已经被掩蔽。 最近的工作表明，通过改进屏蔽令牌的分布（Joshi et al.，2019），预测屏蔽令牌的顺序（Yang et al.，2019）以及替换屏蔽令牌的可用上下文（Dong et al.，2019），取得了收益。 然而，这些方法典型地集中于特定类型的结束任务（例如跨度预测，生成等），限制了它们的适用性。

​		**在本文中，我们提出了BART，它预训练一个双向和自回归相结合的模型。 BART是一个去噪自动编码器，构建了一个序列到序列模型，适用于非常广泛的结束任务。 预训练分为两个阶段:（1）用任意的噪声函数对文本进行破坏；（2）学习序列到序列模型来重建原始文本。 BART使用一个标准的基于转换器的神经机器翻译架构，尽管它很简单，但可以被看作是BERT（由于双向编码器），GPT（带有从左到右解码器）和许多其他较新的预训练方案的推广（参见图1）**。

![image-20200523172340594](https://tva1.sinaimg.cn/large/007S8ZIlly1gf2icrfgkrj31340ps45q.jpg)

(a)BERT:随机令牌被掩码替换，文档被双向编码。 缺失的令牌是独立预测的，因此BERT不能容易地用于生成。

(b)GPT:代币是自回归预测的，意味着GPT可用于生成。 然而，词汇只能以左向语境为条件，无法学习双向交互。

(c)BART:编码器的输入不需要与解码器的输出对齐，允许任意的噪声变换。 在这里，文档由于用掩码符号替换文本的跨距而被破坏。 用双向模型对损坏的文档（左）进行编码，然后用自回归解码器计算原始文档（右）的似然度。 为了进行优化，编码器和解码器都要输入未损坏的文档，我们使用解码器内部隐藏状态的表示形式。

​		<u>这种设置的一个关键优势是噪声灵活性； 可以对原始文本应用任意转换，包括更改其长度。 我们评估了许多去噪方法，通过随机调整原始句子的顺序和使用一种新颖的填充方案（其中任意长度的文本跨度（包括零长度）被单个掩码标记替换）来获得最佳性能。 该方法通过强迫模型更多地考虑整体句子长度并对输入进行更长范围的变换，从而推广了BERT中的原始单词掩蔽和下一个句子预测目标。</u>

​		BART在文本生成时特别有效，但在理解任务中也很有效。 它与RoBERTa（Liu et al.，2019）的表现相匹配，与GLUE（Wang et al.，2018）和SQuAD（Rajpurkar et al.，2016）的类似训练资源相匹配，并在一系列抽象对话，问题回答和总结任务上取得了新的最先进的结果。 例如，它比以前在XSum上的工作提高了6 ROUGE的性能（Narayan等人，2018）。
​		**BART还开辟了新的方法来思考优化问题。 本文提出了一种新的机器翻译方案，将一个BART模型堆叠在几个附加的转换层之上。 通过BART的传播，训练这些层基本上将外语翻译成带噪声的英语，从而使用BART作为预先训练的目标方语言模型。 在WMT罗马尼亚语-英语基准测试中，这种方法比强反译MT基线的性能提高了1.1BLEU.**

​		**BART also opens up new ways of thinking about ﬁne tuning. We present a new scheme for machine translation where a BART model is stacked above a few additional transformer layers. These layers are trained English, by propagation through BART, thereby using BART as a pre-trained target-side language model. This approach improves performance over a strong back-translation MT baseline by 1.1 BLEU on the WMT Romanian-English benchmark.**。

​		为了更好地理解这些影响，我们还报告了一项消融分析，该分析复制了最近提出的其他训练目标。 这项研究允许我们仔细控制多个因素，包括数据和优化参数，这些因素已被表明对整体性能的重要性不亚于训练目标的选择（Liu et al.，2019）。 我们发现，BART在我们考虑的所有任务中都表现出最稳定的强劲性能。

# 2 Model

​		**BART是一个去噪自动编码器，它将损坏的文档映射到它的原始文档。 它被实现为序列到序列模型，带有一个基于损坏文本的双向编码器和一个从左到右的自回归解码器。 对于预训练，我们优化了原始文档的负对数似然**

​		**BART is a denoising autoencoder that maps a corrupted document to the original document it was derived from. It is implemented as a sequence-to-sequence model with a bidirectional encoder over corrupted text and a left-to-right autoregressive decoder. For pre-training, we optimize the negative log likelihood of the original document.。**

## 2.1 Architecture

​		BART使用（Vaswani等人，2017年）的标准序列到序列转换器架构，但在GPT之后，我们将ReLU激活函数修改为GeLUs(Hendrycks&Gimpel，2016年），并从N(0，0.02）初始化参数。 对于我们的基础模型，我们在编码器和解码器中使用6层，对于我们的大模型，我们在每一个中使用12层。 该体系结构与BERT中使用的体系结构密切相关，但有以下不同之处:（1）解码器的每一层额外地在编码器的最大隐层上执行交叉关注（如变压器序列到序列模型中的那样）； （2）在词预测之前，BERT使用了一个额外的前馈网络，而BART没有这样做。 总的来说，BART比同等大小的BERT模型包含大约多10%的参数。

## 2.2 Pre-training BART

​		**BART是通过破坏文档然后优化重建损失--解码器输出和原始文档之间的交叉熵来训练的 BART is trained by corrupting documents and then optimizing a reconstruction loss—the cross-entropy between the decoder’s output and the original document.**。 与针对特定噪声方案量身定制的现有去噪自动编码器不同，BART允许我们应用任何类型的文档损坏。 在关于源的所有信息都丢失的极端情况下，BART相当于一个语言模型。

​		我们对以前提出的几种新的转换进行了实验，但我们相信开发其他新的替代方案具有巨大的潜力。 下面总结了我们使用的转换，示例如图2所示。

![image-20200523173428603](https://tva1.sinaimg.cn/large/007S8ZIlly1gf2inyhghtj30rk07mq4a.jpg)

​		图2:我们实验的输入噪声转换。 这些转换可以组合。

​		**令牌掩蔽Token Masking**  继BERT（Devlin等人，2019）之后，随机令牌被采样并替换为[MASK]元素。

​		**令牌删除Token Deletion**   随机令牌从输入中删除。 与令牌掩蔽相反，模型必须决定哪些位置缺少输入。
​		**文本在填充中Text Infilling**   对多个文本跨度进行采样，跨度长度取自泊松分布。 每个跨度都用单个[MASK]令牌替换。 0-长度跨度对应于[MASK]令牌的插入。 Filling中的文本受SpanBERT（Joshi et al.，2019）的启发，但SpanBERT从不同的（固定几何）分布中采样跨距长度，并用完全相同长度的[MASK]令牌序列替换每个跨距。 filling中的文本教模型预测一个跨度中缺少多少令牌。

​		**句子排列Sentence Permutation**   将文档按句号划分成句子，这些句子按随机顺序排列。

​		**文档旋转Document Rotation**   随机均匀地选择标记，并旋转文档，使其以该标记开始。 此任务训练模型以识别文档的开始。

# 3 Fine-tuning BART

​		BART生成的表示可以用多种方式用于下游应用程序。

## 3.1 Sequence Classification Tasks

​		对于序列分类任务，相同的输入馈入编码器和解码器，而解码器令牌的基本隐藏状态馈入新的多类线性分类器。 这种方法与BERT中的CLS令牌有关； 然而，我们将附加的令牌添加到末尾，以便令牌在解码器中的表示可以关注来自完整输入的解码器状态（图3A)。

![image-20200523173647262](https://tva1.sinaimg.cn/large/007S8ZIlly1gf2iqd6rfzj310i0c842b.jpg)

(a)若要使用BART解决分类问题，则向编码器和解码器输入相同的输入，并使用来自最终输出的表示。
(b)对于机器翻译，我们学习了一个额外的小编码器，它取代了BART中的单词embeddings。 新的编码器可以使用不相交的词汇表。

## 3.2 Token Classification Tasks

​		对于令牌分类任务，例如SQuAD的应答端点分类，我们将完整的文档输入编码器和解码器，并使用解码器的顶部隐藏状态作为每个单词的表示。 此表示用于对令牌进行分类。

## 3.3 Sequence Generation Tasks

​		由于BART具有自回归解码器，因此可以直接对其进行调整，以执行序列生成任务，例如抽象问题回答和总结。 在这两个任务中，信息都是从输入中复制但被操纵的，这与去噪预训练目标密切相关。 这里，编码器输入是输入序列，解码器自回归地产生输出。

## 3.4 Machine Translation

​		我们还探讨了使用BART来改进机器翻译译码器来翻译成英语。 以前的工作Edunov等人。 （2019）已经表明，通过加入预训练的编码器可以改进模型，但是在解码器中使用预训练的语言模型的收益是有限的。 通过添加从bitext学习的一组新的编码器参数（参见图3b)，我们可以将整个BART模型（编码器和解码器）用作单个预训练解码器用于机器翻译。

​		We also explore using BART to improve machine translation decoders for translating into English. Previous work Edunov et al. (2019) has shown that models can be improved by incorporating pre-trained encoders, but gains from using pre-trained language models in decoders have been limited. We show that it is possible to use the entire BART model (both encoder and decoder) as a single pretrained decoder for machine translation, by adding a new set of encoder parameters that are learned from bitext (see Figure 3b).

​		更准确地说，我们用一个新的随机初始化的编码器代替Bart的编码器嵌入层。 该模型是端到端训练的，它训练新的编码器将外来词映射到BART可以去噪到英语的输入中。 新的编码器可以使用与原始BART模型不同的词汇表。

​		More precisely, we replace BART’s encoder embedding layer with a new randomly initialized encoder. The model is trained end-to-end, which trains the new encoder to map foreign words into an input that BART can de-noise to English. The new encoder can use a separate vocabulary from the original BART model.

​		我们分两个步骤训练信源编码器，在两种情况下都是从BART模型的输出反向传播交叉熵损失。 在第一步，我们冻结大部分BART参数，只更新随机初始化的信源编码器，BART位置嵌入以及BART编码器第一层的自注意输入投影矩阵。 在第二步，我们训练所有的模型参数进行少量的迭代。

​		We train the source encoder in two steps, in both cases backpropagating the cross-entropy loss from the output of the BART model. In the ﬁrst step, we freeze most of BART parameters and only update the randomly initialized source encoder, the BART positional embeddings, and the self-attention input projection matrix of BART’s encoder ﬁrst layer. In the second step, we train all model parameters for a small number of iterations.

# 4 Comparing Pre-training Objectives

​		BART在预训练过程中支持比以往工作更广泛的噪声方案。 我们使用基本尺寸模型（6个编码器和6个解码器层，隐藏尺寸为768）对一系列选项进行了比较，这些模型是在§5中的完整大规模实验中考虑的具有代表性的任务子集上进行评估的。

## 4.1 Comparison Objectives

虽然提出了许多训练前目标，但这些目标之间的公平比较却很难实现，至少部分原因是训练数据，训练资源，模型之间的体系结构差异以及优化程序存在差异。 我们重新实现了最近提出的用于鉴别和生成任务的强预训练方法。 我们的目标是尽可能地控制与预训练目标无关的差异。 然而，为了提高性能，我们确实对学习率和层规范化的使用做了一些小的改变（针对每个目标分别调整这些）。 作为参考，我们将我们的实现与BERT发布的数字进行比较，BERT也是在书籍和Wikipedia数据的组合上训练100万步的。 我们比较以下几种方法:

**语言模型Language Model**   类似于GPT（Radford et al.，2018），我们训练了一个从左到右的Transformer语言模型。 该模型相当于BART解码器，没有交叉注意。

**基于XLNet的置换语言模型Permuted Language Model**（Yang et al.，2019），我们对1/6的标记进行采样，并以随机顺序自回归的方式生成它们。 为了与其他模型保持一致，我们没有实现XLNET中跨段的相对位置嵌入或注意力。

**掩蔽语言模型遵循BERT Masked Language Model**（Devlin et al.，2019），我们用[MASK]符号替换15%的令牌，训练模型独立预测原始令牌。

**多任务掩蔽语言模型 Multitask Masked Language Model**   正如在UniLM（Dong et al.，2019）中一样，我们用额外的自我注意掩蔽训练一个掩蔽语言模型。 自注意屏蔽按以下比例随机选择:1/6从左到右，1/6从右到左，1/3未屏蔽，1/3为前50%的令牌未屏蔽，其余部分为从左到右屏蔽。

**Masked Seq-to-Seq**   受MASS（Song et al.，2019）的启发，我们屏蔽了一个包含50%代币的跨度，并训练一个序列到序列模型来预测被屏蔽的代币。

​		对于置换LM，屏蔽LM和多任务屏蔽LM，我们使用双流注意（Yang et al.，2019）来有效地计算序列输出部分的可能性（在输出上使用对角线自我注意掩码来从左到右预测单词）。
​		我们尝试（1）将任务视为标准的序列到序列问题，其中编码器的源输入和目标是解码器的输出；或（2）将源作为解码器中目标的前置，只损失序列的目标部分。 我们发现前者适用于BART模型，后者适用于其他模型。

​		为了最直接地比较我们的模型对其优化目标（人类文本的对数似然）建模的能力，我们在表1中报告了困惑。

## 4.2 Tasks

**SQuAD**（Rajpurkar等人，2016）一个关于维基百科段落的提取性问题回答任务。 答案是从给定文档上下文中提取的文本跨度。 与BERT（Devlin et al.，2019）类似，我们使用串联的问题和上下文作为BART编码器的输入，并额外地将它们传递给解码器。 该模型包括分类器，用于预测每个令牌的起始和结束指数。

**MNLI**（Williams et al.，2017），一个双文本分类任务，用于预测一个句子是否包含另一个句子。 经过优化的模型将这两个句子与附加的EOS令牌连接起来，并将它们传递给BART编码器和解码器。 与BERT不同的是，EOS标记的表示被用来对句子关系进行分类。

**ELI5**(Fan et al.，2019），一个长形式的抽象问答数据集。 模型根据问题和支持文档的串联来生成答案。

**XSum**（Narayan et al.，2018），一个具有高度抽象摘要的新闻摘要数据集。

**ConvAI2**（Dinan et al.，2019）是一种基于语境和角色的对话反应生成任务。

**CNN/DM**（Hermann等，2015），一个新闻摘要数据集。 这里的摘要通常与源句密切相关。

## 4.3 Results

​		结果如表1所示。 有几个趋势是明确的:

![image-20200523174653489](https://tva1.sinaimg.cn/large/007S8ZIlly1gf2j0vlae9j311y0hetds.jpg)

表1:培养前目标比较。 所有模型的规模相当，都是在书籍和维基百科数据的组合上进行100万步的训练。 底部两个模块中的条目使用相同的代码库对相同的数据进行训练，并使用相同的过程进行优化。 第二块中的条目受先前工作中提出的预培训目标的启发，但已被简化为集中于评估目标（见§4.1)。 不同任务的性能差异很大，但带有文本的BART模型表现出最稳定的强劲性能。

**预训练方法的表现在不同的任务中有显著的差异Performance of pre-training methods varies significantly across tasks**   预训练方法的有效性高度依赖于任务。 例如，一个简单的语言模型实现了最好的ELI5性能，但却得到了最差的SQUAD结果。

**标记掩蔽是关键的Token masking is crucial**  预训练目标，基于旋转文档或排列句子的单独训练效果较差。 成功的方法要么使用令牌删除或掩蔽，要么使用自我注意掩蔽。 在生成任务中，删除似乎优于屏蔽。

**从左到右预训练改进了生成Left-to-right pre-training improves generation**   掩码语言模型和置换语言模型在生成时表现不如其他模型，并且是我们认为在预训练期间唯一不包括从左到右自回归语言模型的模型。

**双向编码器对SQuADBidiractional encoders are crucial for SQuAD**   至关重要正如以前的工作（Devlin et al.，2019）所指出的那样，仅仅从左到右的解码器在SQuAD上表现较差，因为未来的上下文在分类决策中至关重要。 然而，BART仅用一半的双向层数就实现了类似的性能。

**预训练目标并不是唯一的重要因素The pre-training objective is not the only important factor**，我们的置换语言模型表现不如XLNet（Yang et al.，2019）。 其中一些差异可能是由于没有包括其他体系结构改进，例如相对位置嵌入或段级重复。
**纯语言模型在ELI5上表现最好Pure language models perform best on ELI5**   ELI5数据集是一个离群值，其困惑程度比其他任务高得多，并且是其他模型唯一表现优于BART的生成任务。 纯语言模型的表现最好，这表明当输出只受到输入的松散约束时，BART的效率较低。

**BART实现了最稳定的强劲性能BART achieves the most consistently strong performance**。 除了ELI5，使用文本填充的BART模型在所有任务上都表现良好。

# 5 Large-scale Pre-training Experiments

​		最近的工作表明，当预训练规模扩大到大批量时，下游性能可以显著提高（Yang et al.，2019；Liu et al.，2019）和语料库。 为了测试BART在该机制中的表现，并为下游任务创建一个有用的模型，我们使用与RoBERTa模型相同的规模对BART进行了训练。

## 5.1 Experimental Setup

​		我们预先训练了一个大型模型，在编码器和解码器中各有12层，隐藏大小为1024。 继RoBERTa（Liu et al.，2019）之后，我们使用8000的批处理规模，并对模型进行500000步的训练。 文档使用与GPT-2相同的字节对编码进行标记化（Radford等人，2019）。 基于§4节中的结果，我们使用了文本填充和句子排列的组合。 我们在每个文档中屏蔽30%的标记，并排列所有句子。 尽管句子排列在CNN/DM总结数据集上仅显示出显著的附加增益，但我们假设较大的预训练模型可能更能从这项任务中学习。 为了帮助模型更好地处理数据，我们在最后10%的训练步骤中禁用了退出。 我们使用与Liu等人相同的预训练数据。 （2019），包含160GB的新闻，书籍，故事和网络文本。

## 5.2 Discriminative Tasks

​		表2比较了BART和最近几种方法在经过充分研究的SQuAD和GLUE任务上的表现（Warstadt et al.，2018；Socher et al.，2013；Dolan&Brockett，2005；Agirre et al.，2007；Williams et al.，2018；Dagan et al.，2006；Levesque et al.，2011）。

​		最直接可比的基线是RoBERTa，它是用相同的资源预先训练的，但目标不同。 总体而言，BART的表现类似，在大多数任务上模型之间只有很小的差异。 表明BART对生成任务的改进并不以牺牲分类性能为代价。

## 5.3 Generation Tasks

​		我们还实验了几个文本生成任务。 从输入到输出文本，BART被调优为标准序列到序列模型。 在调整期间，我们使用标签平滑交叉熵损失（Pereyra等人，2017年），平滑参数设置为0.1。 在生成过程中，我们将波束大小设置为5，在波束搜索中移除重复的三元图，并在验证集上用最小-LEN，最大-LEN，长度惩罚来调谐模型（Fan et al.，2017）。

​		**摘要Summarization**   为了与现有的摘要数据进行比较，我们给出了CNN/DailyMail和XSum两个具有不同性质的摘要数据集的结果。

​		CNN/DailyMail的摘要往往与原文相似。 提取模型在这里做得很好，甚至前三个源句的基线也很有竞争力。 尽管如此，巴特的表现胜过所有现有的工作。

​		相比之下，XSum是高度抽象的，提取模型的性能较差。 BART在所有ROUGE指标上的表现都比以前最好的，利用了BERT的作品高出了大约6.0分--这表明在这个问题上的表现有了显著的进步。 从质量上看，样品质量较高（见§6)。

​		**对话Dialogue**   我们评估了C ONVAI2上的对话响应生成（Dinan et al.，2019），在这种情况下，代理必须根据先前的上下文和文本指定的角色生成响应。 BART在两个自动化度量方面的表现优于以前的工作。
​		**抽象QA Abstraction QA**   我们使用最近提出的ELI5数据集来测试模型生成长自由形式答案的能力。 我们发现，BART的表现比以前最好的作品高出1.2ROUGE-L，但数据集仍然具有挑战性，因为问题只对答案进行了微弱的限定。

## 5.4 Translation

​		我们还评估了WMT16罗马尼亚英语的表现，并用Sennrich等人的回译数据进行了补充。 （2016年）。 我们使用6层变压器源编码器将罗马尼亚语映射成BART能够去噪成英语的表示，遵循§3.4中介绍的方法。 实验结果如表6所示。 我们将我们的结果与具有Transformerlarge设置（基线行）的基线Transformer架构（Vaswani等人，2017）进行比较。 我们展示了我们模型的两个步骤在填充BART和调谐BART行中的性能。 对于每一行，我们在原始的WMT16罗马尼亚语-英语中进行实验，并增加了反翻译数据。 我们使用的束宽为5，长度惩罚为a=1。 初步结果表明，在没有反向翻译数据的情况下，我们的方法效果较差，而且容易出现过度的情况--未来的工作应该探索更多的正则化技术。

​		We also evaluated performance on WMT16 RomanianEnglish, augmented with back-translation data from Sennrich et al. (2016). We use a 6-layer transformer source encoder to map Romanian into a representation that BART is able to de-noise into English, following the approach introduced in §3.4. Experiment results are presented in Table 6. We compare our results against a baseline Transformer architecture (Vaswani et al., 2017) with Transformerlarge settings (the baseline row). We show the performance of both steps of our model in the ﬁxed BART and tuned BART rows. For each row we experiment on the original WMT16 Romanian-English augmented with back-translation data. We use a beam width of 5 and a length penalty of α = 1. Preliminary results suggested that our approach was less effective without back-translation data, and prone to overﬁtting—future work should explore additional regularization techniques.

![image-20200523174955228](/Users/suntian/Library/Application Support/typora-user-images/image-20200523174955228.png)

表6:WMT'16 RO-EN上基线和BART的性能(BLEU)与反向翻译数据增强。 通过使用单语英语预训练，BART比强反译(BT)基线提高。e

# 6 Qualitative Analysis

​		BART在总结度量方面显示了很大的改进，比以前的技术水平提高了6个百分点。 为了理解BART超越自动化度量的性能，我们定性地分析了它的世代。
​		表7显示了BART生成的示例摘要。 示例取自在创建预训练语料库之后发布的维基新闻文章，以消除描述的事件出现在模型的训练数据中的可能性。 继Narayan等人之后。 （2018年），在总结文章之前，我们删除了文章的第一句，因此没有简单的摘要。

​		不出所料，模型输出是流畅的，语法的英语。 然而，模型输出也是高度抽象的，从输入中复制的短语很少。 输出通常也是真实准确的，并且集成了来自输入文档的支持证据和背景知识（例如，正确填写姓名，或推断PG&E在加利福尼亚运营）。 在第一个例子中，要推断出“鱼”正在保护珊瑚礁免受全球变暖的影响，就需要从文本中进行非同寻常的推断。 然而，该工作发表在《科学》上的说法没有得到消息来源的支持。
​		这些样本表明BART预训练学习到了自然语言理解和生成的强大组合。

# 7 Related Work

​		早期的预训练方法是基于语言模型的。 GPT（Radford et al.，2018）只对左向语境进行建模，这对于某些任务来说是有问题的。 ELMo（Peters et al.，2018）将仅左和仅右表征串联起来，但并不预先训练这些特征之间的交互。 Radford等人。 （2019年）论证了超大型语言模型可以充当无监督多任务模型。
​		BERT（Devlin et al.，2019）引入了掩蔽语言建模，它允许预训练学习左右语境词之间的交互。 最近的工作表明，通过更长时间的训练（Liu et al.，2019），通过跨层绑定参数（Lan et al.，2019）以及通过掩蔽跨度而不是单词（Joshi et al.，2019），可以实现非常强的性能。 预测不是自回归的，降低了BERT对于生成任务的有效性。

​		UniLM（Dong et al.，2019年）用一组掩码为BERT调音，其中一些掩码只允许向左的上下文。 与BART一样，这使得UniLM既可以用于生成性任务，也可以用于鉴别性任务。 不同之处在于，UniLM预测是条件独立的，而BART预测是自回归的。 BART减少了预训练和生成任务之间的不匹配，因为解码器总是在未损坏的上下文上训练的。

​		**MASS（Song等人，2019年）或许是与BART最相似的模型。 将连续的令牌跨度被屏蔽的输入序列映射到由丢失的令牌组成的序列。 MASS对于鉴别任务的效率较低，因为不相交的令牌集被馈送到编码器和解码器。**

​		MASS (Song et al., 2019) is perhaps the most similar model to BART. An input sequence where a contiguous span of tokens is masked is mapped to a sequence consisting of the missing tokens. MASS is less effective for discriminative tasks, because disjoint sets of tokens are fed into the encoder and decoder.

​		XL-Net（Yang等人，2019）通过以置换顺序自回归地预测屏蔽令牌来扩展BERT。 该目标允许预测同时基于左上下文和右上下文。 相比之下，BART解码器在预训练期间从左到右工作，与生成期间的设置相匹配。

​		有几篇论文探讨了使用预先训练的表示来改进机器翻译。 最大的改进来自对源语言和目标语言的预培训（Song et al.，2019；Lample&Conneau，2019），但这需要对所有感兴趣的语言进行预培训。 其他工作已经表明，可以使用预先训练的表示来改进编码器（Edunov等人，2019），但在解码器中的增益更加有限。 我们展示了如何使用BART来改进机器翻译解码器。

# 8 Conclusions

​		我们引入了BART，这是一种学习将损坏文档映射到原始文档的预训练方法。 BART在区分任务上取得了与RoBERTa相似的性能，同时在许多文本生成任务上取得了新的最新结果。 未来的工作应该探索新的方法来破坏培训前的文档，也许可以根据具体的最终任务对其进行调整。