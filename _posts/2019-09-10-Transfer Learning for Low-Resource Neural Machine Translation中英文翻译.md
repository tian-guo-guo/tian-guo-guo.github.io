---
layout:     post                    # 使用的布局（不需要改）
title:      Transfer Learning for Low-Resource Neural Machine Translation中英文翻译             # 标题 
subtitle:   迁移学习在MT的应用 #副标题
date:       2019-09-10              # 时间
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

## Transfer Learning for Low-Resource Neural Machine Translation Barret

### Abstract
The encoder-decoder framework for neural machine translation (NMT) has been shown effective in large data scenarios, but is much less effective for low-resource languages. We present a transfer learning method that signifi- cantly improves BLEU scores across a range of low-resource languages. Our key idea is to first train a high-resource language pair (the parent model), then transfer some of the learned parameters to the low-resource pair (the child model) to initialize and constrain training. Using our transfer learning method we improve baseline NMT models by an av- erage of 5.6 BLEU on four low-resource lan- guage pairs. Ensembling and unknown word replacement add another 2 BLEU which brings the NMT performance on low-resource ma- chine translation close to a strong syntax based machine translation (SBMT) system, exceed- ing its performance on one language pair. Ad- ditionally, using the transfer learning model for re-scoring, we can improve the SBMT sys- tem by an average of 1.3 BLEU, improving the state-of-the-art on low-resource machine translation.

用于神经机器翻译(NMT)的编码器-解码器框架已被证明在大数据场景中是有效的，但对于低资源语言的效果要差得多。 我们提出了一种迁移学习方法，它能显著提高低资源语言的BLEU分数。 我们的核心思想是首先训练一个高资源语言对（父模型），然后将一些学习到的参数传递给低资源语言对（子模型）来初始化和约束训练。 利用我们的迁移学习方法，我们对四个低资源语言对的基线NMT模型进行了5.6BLEU的改进。 集成和未知词替换增加了2个BLEU，使得低资源机器翻译的NMT性能接近于一个强的基于语法的机器翻译系统(SBMT)，超过了它在一个语言对上的性能。 另外，利用迁移学习模型对SBMT系统进行重新评分，平均提高了1.3bleu，改善了低资源机器翻译的现状。 

### 1 Introduction
Neural machine translation (NMT) (Sutskever et al., 2014) is a promising paradigm for extracting translation knowledge from parallel text. NMT systems have achieved competitive accuracy rates under large-data training conditions for language pairs such as English–French. However, neural methods are data-hungry and learn poorly from low-count events. This behavior makes vanilla NMT a poor choice for low-resource languages, where parallel data is scarce. Table 1 shows that for 4 low-resource languages, a standard string-to-tree statistical MT system (SBMT) (Galley et al., 2004; Galley et al., 2006) strongly outperforms NMT, even when NMT uses the state-of-the-art local attention plus feed- input techniques from Luong et al. (2015a).

神经机器翻译（Neural Machine Translation，NMT）（Sutskever et al.，2014）是一种很有前途的从平行文本中提取翻译知识的范式。 NMT系统在英法等语言对的大数据训练条件下取得了有竞争力的准确率。 然而，神经方法对数据的需求很大，从低计数事件中学习的能力也很差。 这种行为使得Vanilla NMT在并行数据稀缺的低资源语言中是一个糟糕的选择。 表1显示，对于4种低资源语言，标准的String-to-Tree统计MT系统(SBMT)（Galley等人，2004；Galley等人，2006）的性能强于NMT，即使NMT使用Luong等人的最新的本地注意加提要输入技术。 (2015a)。
![190910-Table1.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table1.png)

In this paper, we describe a method for substantially improving NMT results on these languages. Our key idea is to first train a high-resource language pair, then use the resulting trained network (the parent model) to initialize and constrain training for our low-resource language pair (the child model). We find that we can optimize our results by fixing certain parameters of the parent model and letting the rest be fine-tuned by the child model. We re- port NMT improvements from transfer learning of 5.6 BLEU on average, and we provide an analysis of why the method works. The final NMT system approaches strong SBMT baselines in all four language pairs, and exceeds SBMT performance in one of them. Furthermore, we show that NMT is an exceptional re-scorer of ‘traditional’ MT output; even NMT that on its own is worse than SBMT is consistently able to improve upon SBMT system output when incorporated as a re-scoring model.

在本文中，我们描述了一种在这些语言上实质上改进NMT结果的方法。 我们的关键思想是首先训练一个高资源语言对，然后使用得到的训练网络（父模型）初始化和约束我们的低资源语言对（子模型）的训练。 我们发现，我们可以通过固定父模型的某些参数并让其馀的参数由子模型微调来优化我们的结果。 我们从平均5.6BLEU的迁移学习中报告了NMT的改进，并分析了该方法的工作原理。 最终的NMT系统在所有四种语言对中都接近于强大的SBMT基线，其中一种语言对的性能超过了SBMT。 此外，我们还证明了NMT是“传统”MT输出的一个特殊的重得分者； 即使NMT本身比SBMT差，作为一个重新评分模型，它也能够持续改进SBMT系统的输出。

We provide a brief description of our NMT model in Section 2. Section 3 gives some background on transfer learning and explains how we use it to improve machine translation performance. Our main experiments translating Hausa, Turkish, Uzbek, and Urdu into English with the help of a French–English parent model are presented in Section 4. Section 5 explores alternatives to our model to enhance under- standing. We find that the choice of parent language pair affects performance, and provide an empirical upper bound on transfer performance using an artificial language. We experiment with English-only language models, copy models, and word-sorting models to show that what we transfer goes beyond monolingual information and that using a translation model trained on bilingual corpora as a parent is essential. We show the effects of freezing, fine- tuning, and smarter initialization of different components of the attention-based NMT system during transfer. We compare the learning curves of transfer and no-transfer models, showing that transfer solves an overfitting problem, not a search problem. We summarize our contributions in Section 6.

我们在第2节中简要描述了我们的NMT模型。 第三节介绍了迁移学习的背景知识，并说明了如何利用迁移学习来提高机器翻译的性能。 第四节介绍了我们在法英父母模式的帮助下，将豪萨语、土耳其语、乌兹别克语和乌尔都语翻译成英语的主要实验。 第5节探讨了我们的模式的替代办法，以提高人们的认识。 我们发现，父母语言对的选择会影响学习成绩，并提供了一个使用人工语言的迁移成绩的经验上界。 我们用纯英语语言模型、复制模型和单词排序模型进行了实验，以表明我们传递的信息不仅仅是单一语言信息，使用一个经过双语语料库训练的翻译模型作为父语言是非常必要的。 我们展示了在转移期间冻结、微调和更智能地初始化基于注意的NMT系统的不同组件的效果。 我们比较了转移模型和无转移模型的学习曲线，结果表明转移模型解决的是一个过拟合问题，而不是一个搜索问题。 我们在第6节中总结了我们的贡献。

### 2 NMT Background
In the neural encoder-decoder framework for MT (Neco and Forcada, 1997; Casta˜no and Casacuberta, 1997; Sutskever et al., 2014; Bahdanau et al., 2014; Luong et al., 2015a), we use a recurrent neural network (encoder) to convert a source sen- tence into a dense, fixed-length vector. We then use another recurrent network (decoder) to convert that vector to a target sentence. In this paper, we use a two-layer encoder-decoder system (Figure 1) with long short-term memory (LSTM) units (Hochreiter and Schmidhuber, 1997). The models were trained to optimize maximum likelihood (via a softmax layer) with back-propagation through time (Werbos, 1990). Additionally, we use an attention mechanism that allows the target decoder to look back at the source encoder, specifically the local attention model from Luong et al. (2015a). In our model we also use the feed-input input connection from Luong et al. (2015a) where at each timestep on the decoder we feed in the top layer’s hidden state into the lowest layer of the next timestep.

在用于MT的神经编码器-解码器框架（Neco和Forcada，1997；Casta~NO和Casacuberta，1997；Sutskever等人，2014；Bahdanau等人，2014；Luong等人，2015a)中，我们使用递归神经网络（编码器）将源感测转换为密集的固定长度向量。 然后，我们使用另一个递归网络（解码器）将该向量转换为目标语句。 在本文中，我们使用了一个具有长短时记忆(LSTM)单元的两层编码器-解码器系统（图1）（Hochreiter和Schmidhuber，1997）。 这些模型被训练成通过时间反向传播优化最大似然（通过SoftMax层）（Werbos，1990）。 此外，我们使用了一种关注机制，允许目标解码器回顾源编码器，特别是Luong等人的局部关注模型。 (2015a)。 在我们的模型中，我们还使用了Luong等人的feed-input输入连接。 (2015a)其中在解码器上的每个时间步，我们将顶层的隐藏状态馈送到下一个时间步的最低层。 

### 3 Transfer Learning
Transfer learning uses knowledge from a learned task to improve the performance on a related task, typically reducing the amount of required training data (Torrey and Shavlik, 2009; Pan and Yang, 2010). In natural language processing, transfer learning methods have been successfully applied to speech recognition, document classification and sentiment analysis (Wang and Zheng, 2015). Deep learning models discover multiple levels of representation, some of which may be useful across tasks, which makes them particularly suited to transfer learning (Bengio, 2012). For example, Cires¸an et al. (2012) use a convolutional neural network to recognize handwritten characters and show positive effects of transfer between models for Latin and Chinese characters. Ours is the first study to apply transfer learning to neural machine translation. 

转移学习利用从学习任务中获得的知识来提高相关任务的绩效，通常减少所需的培训数据量（Torrey和Shavlik，2009年；Pan和Yang，2010年）。 在自然语言处理中，迁移学习方法已经成功地应用于语音识别、文档分类和情感分析（Wang和Zheng，2015）。 深度学习模型发现了多个表征层次，其中一些可能在任务之间有用，这使得它们特别适合于迁移学习（Bengio，2012）。 例如，Cires等人。 （2012）使用卷积神经网络识别手写体字符，并显示拉丁文和中文字符模型之间转移的积极效果。 我们的研究是第一个将迁移学习应用于神经机器翻译的研究。

![190910-fugure1.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-figure1.png)

There has also been work on using data from multiple language pairs in NMT to improve performance. Recently, Dong et al. (2015) showed that sharing a source encoder for one language helps performance when using different target decoders for different languages. In that paper the authors showed that using this framework improves performance for low-resource languages by incorporating a mix of low-resource and high-resource languages. Firat et al. (2016) used a similar approach, employ- ing a separate encoder for each source language, a separate decoder for each target language, and a shared attention mechanism across all languages. They then trained these components jointly across multiple different language pairs to show improve- ments in a lower-resource setting.

在NMT中，还研究了如何使用来自多个语言对的数据来提高性能。 最近，Dong等人。 （2015）表明，当对不同语言使用不同的目标解码器时，共享一种语言的源编码器有助于提高性能。 在这篇文章中，作者展示了使用该框架通过合并低资源和高资源语言的混合来提高低资源语言的性能。 Firat等人 （2016）使用了类似的方法，为每种源语言使用单独的编码器，为每种目标语言使用单独的解码器，以及跨所有语言的共享注意机制。 然后，他们在多个不同的语言对中联合培训这些组件，以在资源较少的环境中显示改进。

There are a few key differences between our work and theirs. One is that we are working with truly small amounts of training data. Dong et al. (2015) used a training corpus of about 8m English words for the low-resource experiments, and Firat et al. (2016) used from 2m to 4m words, while we have at most 1.8m words, and as few as 0.2m. Additionally, the aforementioned previous work used the same domain for both low-resource and high-resource languages, while in our case the datasets come from vastly different domains, which makes the task much harder and more realistic. Our approach only requires using one additional high-resource language, while the other papers used many. Our approach also allows for easy training of new low- resource languages, while Dong et al. (2015) and Fi- rat et al. (2016) do not specify how a new language should be added to their pipeline once the models are trained. Finally, Dong et al. (2015) observe an aver- age BLEU gain on their low-resource experiments of +1.16, and Firat et al. (2016) obtain BLEU gains of +1.8, while we see a +5.6 BLEU gain. 

我们的工作和他们的工作有几个主要的不同之处。 其一，我们正在处理真正少量的训练数据。 Dong等人。 （2015）使用了大约800万个英语单词的训练语料库进行低资源实验，以及Firat等人。 （2016年）使用了200万至400万字，而我们最多只有180万字，最少只有20万字。 此外，前面提到的工作对低资源和高资源语言使用相同的域，而在我们的示例中，数据集来自截然不同的域，这使得任务变得更加困难和现实。 我们的方法只需要使用一种额外的高资源语言，而其他的文件使用了很多。 我们的方法还允许容易地训练新的低资源语言，而Dong等人。 （2015年）和Fi-Rat等人。 （2016）不具体说明一旦培训了模型，应如何将新语言添加到他们的管道中。 最后，Dong等人。 （2015年）在低资源实验中观察到平均BLEU增益为+1.16，Firat等人。 （2016）获得+1.8的BLEU增益，而我们看到+5.6的BLEU增益。

The transfer learning approach we use is simple and effective. We first train an NMT model on a large corpus of parallel data (e.g., French–English). We call this the parent model. Next, we initialize an NMT model with the already-trained parent model. This new model is then trained on a very small par- allel corpus (e.g., Uzbek–English). We call this the child model. Rather than starting from a random position, the child model is initialized with the weights from the parent model.

我们使用的迁移学习方法简单有效。 我们首先在大量的平行数据（例如，法语-英语）上训练一个NMT模型。 我们称之为父模型。 接下来，我们用已经训练好的父模型初始化一个NMT模型。 然后，在一个非常小的并行语料库（如乌兹别克语-英语）上训练这个新模型。 我们把这叫做儿童模型。 子模型不是从随机位置开始，而是用父模型的权重初始化。

A justification for this approach is that in scenarios where we have limited training data, we need a strong prior distribution over models. The parent model trained on a large amount of bilingual data can be considered an anchor point, the peak of our prior distribution in model space. When we train the child model initialized with the parent model, we fix parameters likely to be useful across tasks so that they will not be changed during child model train- ing. In the French–English to Uzbek–English ex- ample, as a result of the initialization, the English word embeddings from the parent model are copied, but the Uzbek words are initially mapped to random French embeddings. The parameters of the English embeddings are then frozen, while the Uzbek em- beddings’ parameters are allowed to be modified, i.e. fine-tuned, during training of the child model. Freezing certain transferred parameters and fine tun- ing others can be considered a hard approximation to a tight prior or strong regularization applied to some of the parameter space. We also experiment with ordinary L2 regularization, but find it does not significantly improve over the parameter freezing described above. 

这种方法的一个理由是，在我们的训练数据有限的情况下，我们需要模型上的强先验分布。 在大量双语数据上训练的父模型可以看作锚点，是模型空间中先验分布的峰值。 当我们训练用父模型初始化的子模型时，我们修正了可能在任务之间有用的参数，以便它们在子模型训练期间不会被更改。 在法语-英语到乌兹别克语-英语的例子中，由于初始化，从父模型复制英语单词嵌入，但是乌兹别克语单词最初映射到随机法语嵌入。 然后冻结英文嵌入的参数，而允许在训练子模型期间修改乌兹别克嵌入的参数，即微调。 冻结某些传递的参数和微调其他参数可以被认为是对应用于一些参数空间的紧先验或强正则化的硬近似。 我们还对普通的L2正则化进行了实验，但发现它与上面描述的参数冻结相比没有明显的改善。

Our method results in large BLEU increases for a variety of low resource languages. In one of the four language pairs our NMT system using transfer beats a strong SBMT baseline. Not only do these transfer models do well on their own, they also give large gains when used for re-scoring n-best lists (n = 1000) from the SBMT system. Section 4 de- tails these results.

我们的方法导致各种低资源语言的BLEU大幅增加。 在四种语言对中的一种语言对中，我们的NMT系统使用的是迁移，而不是强大的SBMT基线。 这些转移模型不仅本身表现良好，而且在用于从SBMT系统中重新评分n-最佳列表(n=1000)时也会带来很大的收益。 第四节对这些结果进行了分析。 

### 4 Experiments
To evaluate how well our transfer method works we apply it to a variety of low-resource languages, both stand-alone and for re-scoring a strong SBMT base- line. We report large BLEU increases across the board with our transfer method. 

为了评估我们的迁移方法的工作情况，我们将其应用于各种低资源语言，包括独立语言和用于重新评分强大的SBMT基线的语言。 我们报告使用我们的转移方法全面大幅增加BLEU。 

For all of our experiments with low-resource languages we use French as the parent source language and for child source languages we use Hausa, Turkish, Uzbek, and Urdu. The target language is al- ways English. Table 1 shows parallel training data set sizes for the child languages, where the language with the most data has only 1.8m English tokens. For comparison, our parent French–English model uses a training set with 300 million English tokens and achieves 26 BLEU on the development set. Table 1 also shows the SBMT system scores along with the NMT baselines that do not use transfer. There is a large gap between the SBMT and NMT systems when our transfer method is not used. 

对于所有低资源语言的实验，我们使用法语作为父源语言，对于子源语言，我们使用豪萨语、土耳其语、乌兹别克语和乌尔都语。 目的语是通用英语。 表1显示了子语言的并行训练数据集大小，其中数据最多的语言只有180万个英语标记。 相比之下，我们的母公司法文-英文模式使用了一套有3亿个英文标记的培训集，并在开发集上实现了26个BLEU。 表1还显示了SBMT系统得分以及不使用传输的NMT基线。 SBMT系统和NMT系统在不采用我们的传输方法时存在很大的差距。

![190910-Table1.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table1.png)

The SBMT system used in this paper is a string-to-tree statistical machine translation system (Galley et al., 2006; Galley et al., 2004). In this system there are two count-based 5-gram language models. One is trained on the English side of the WMT 2015 English–French dataset and the other is trained on the English side of the low-resource bi- text. Additionally, the SBMT models use thousands of sparsely-occurring, lexicalized syntactic features (Chiang et al., 2009). 

本文使用的SBMT系统是一个字符串到树的统计机器翻译系统（Galley等人，2006；Galley等人，2004）。 在该系统中，有两个基于计数的5-克语言模型。 一个在2015年WMT英法数据集的英文部分接受培训，另一个在资源不足的双语文本的英文部分接受培训。 此外，SBMT模型使用数千个稀疏出现的词汇化句法特征（Chiang et al.，2009）。

For our NMT system, we use development sets for Hausa, Turkish, Uzbek, and Urdu to tune the learning rate, parameter initialization range, dropout rate, and hidden state size for all the experiments. For training we use a minibatch size of 128, hidden state size of 1000, a target vocabulary size of 15K, and a source vocabulary size of 30K. The child models are trained with a dropout probability of 0.5, as in Zaremba et al. (2014). The common parent model is trained with a dropout probability of 0.2. The learning rate used for both child and parent mod- els is 0.5 with a decay rate of 0.9 when the de- velopment perplexity does not improve. The child models are all trained for 100 epochs. We re-scale the gradient when the gradient norm of all param- eters is greater than 5. The initial parameter range is [-0.08, +0.08]. We also initialize our forget-gate biases to 1 as specified by J´ozefowicz et al. (2015) and Gers et al. (2000). For decoding we use a beam search of width 12.

对于我们的NMT系统，我们使用Hausa、Turkish、Uzbek和Urdu的开发集来调整所有实验的学习速率、参数初始化范围、辍学率和隐藏状态大小。 对于训练，我们使用小批大小128，隐藏状态大小1000，目标词汇大小15K，源词汇大小30K。 如Zaremba等人所述，儿童模型的辍学率为0.5。 （2014年）。 公共父模型以0.2的辍学率进行训练。 在发展困惑没有改善的情况下，儿童和家长模式的学习率为0.5，衰减率为0.9。 这些儿童模特都经过了100年的训练。 当所有参数的梯度范数大于5时，我们重新缩放梯度。 初始参数范围为[-0.08，+0.08]。 我们还将遗忘栅偏置初始化为1，如Józefowicz等人所述。 （2015年）和Gers等人。 （2000年）。 对于解码，我们使用宽度为12的波束搜索。 

### 4.1 Transfer Results
The results for our transfer learning method applied to the four languages above are in Table 2. The parent models were trained on the WMT 2015 (Bojar et al., 2015) French–English corpus for 5 epochs. Our baseline NMT systems (‘NMT’ row) all receive a large BLEU improvement when using the transfer method (the ‘Xfer’ row) with an average BLEU improvement of 5.6. Additionally, when we use un- known word replacement from Luong et al. (2015b) and ensemble together 8 models (the ‘Final’ row) we further improve upon our BLEU scores, bringing the average BLEU improvement to 7.5. Overall our method allows the NMT system to reach competitive scores and outperform the SBMT system in one of the four language pairs.

我们的迁移学习方法应用于以上四种语言的结果见表2。 母模型在2015年WMT（Bojar等人，2015年）法语-英语语料库上进行了5个时期的训练。 我们的基线NMT系统（“NMT”行）在使用传输方法（“XFER”行）时都得到了很大的BLEU改进，平均BLEU改进为5.6。 另外，当我们使用来自Luong等人的未知词替换时。 (2015b)和集成8个模型（“最终”行），我们进一步改善了我们的BLEU得分，使平均BLEU改善到7.5。 总的来说，我们的方法允许NMT系统在四种语言对中的一种中达到竞争性的分数并且优于SBMT系统。

![190910-Table2.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table2.png)

### 4.2 Re-scoring Results
We also use the NMT model with transfer learn- ing as a feature when re-scoring output n-best lists (n = 1000) from the SBMT system. Table 3 shows the results of re-scoring. We compare re-scoring with transfer NMT to re-scoring with baseline (i.e. non-transfer) NMT and to re-scoring with a neural language model. The neural language model is an LSTM RNN with 2 layers and 1000 hidden states. It has a target vocabulary of 100K and is trained using noise-contrastive estimation (Mnih and Teh, 2012; Vaswani et al., 2013; Baltescu and Blunsom, 2015; Williams et al., 2015). Additionally, it is trained us- ing dropout with a dropout probability of 0.2 as suggested by Zaremba et al. (2014). Re-scoring with the transfer NMT model yields an improvement of 1.1– 1.6 BLEU points above the strong SBMT system; we find that transfer NMT is a better re-scoring feature than baseline NMT or neural language models. 

在对SBMT系统输出的N-Best列表(n=1000)进行重新评分时，我们还使用了以迁移学习为特征的NMT模型。 表3显示了重新评分的结果。 我们比较了重评分与迁移NMT，重评分与基线（即非迁移）NMT和重评分与神经语言模型。 该神经语言模型是一个具有2层1000个隐态的LSTM神经网络。 它的目标词汇量为100K，使用噪声对比估计进行训练（MNIH和TEH，2012年；Vaswani等人，2013年；Baltescu和Blunsom，2015年；Williams等人，2015年）。 此外，还按照Zaremba等人的建议，对辍学者进行了辍学概率为0.2的培训。 （2014年）。 用转移NMT模型进行重新评分，比强SBMT系统提高了1.1-1.6个BLEU点； 我们发现，迁移NMT比基线NMT或神经语言模型具有更好的再评分特征。 

![190910-Table3.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table3.png)

In the next section, we describe a number of additional experiments designed to help us understand the contribution of the various components of our transfer model.

在下一节中，我们描述了一些额外的实验，这些实验旨在帮助我们理解传输模型的各个组件的贡献。

### 5 Analysis
We analyze the effects of using different parent models, regularizing different parts of the child model, and trying different regularization techniques.

我们分析了使用不同的父模型、对子模型的不同部分进行正则化以及尝试不同的正则化技术的效果。 

### 5.1 Different Parent Languages
In the above experiments we use French–English as the parent language pair. Here, we experiment with different parent languages. In this set of experiments we use Spanish–English as the child language pair. A description of the data used in this section is presented in Table 4. 

在上述实验中，我们使用法语和英语作为母体语言对。 在这里，我们用不同的父语言进行实验。 在这组实验中，我们使用西班牙语和英语作为子语言对。 表4说明了本节使用的数据。

![190910-Table4.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table4.png)

Our experimental results are shown in Table 5, where we use French and German as parent languages. If we just train a model with no transfer on a small Spanish–English training set we get a BLEU score of 16.4. When using our transfer method we get Spanish–English BLEU scores of 31.0 and 29.8 via French and German parent languages, respectively. As expected, French is a better parent than German for Spanish, which could be the result of the parent language being more similar to the child language. We suspect using closely-related parent language pairs would improve overall quality.

我们的实验结果如表5所示，其中我们使用法语和德语作为父语言。 如果我们只是在一个小的西班牙-英语训练集上训练一个没有转会的模特，我们得到了16.4的BLEU分数。 当使用我们的迁移方法时，我们通过法语和德语的父语言分别得到31.0和29.8的西班牙语-英语BLEU分数。 正如预期的那样，法语比德语更适合西班牙语，这可能是因为父语言与子语言更相似。 我们怀疑使用密切相关的父语言对会提高整体质量。

![190910-Table5.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table5.png)

### 5.2 Effects of having Similar Parent Language
Next, we look at a best-case scenario in which the parent language is as similar as possible to the child language. 

接下来，我们看一个最好的情况，在这种情况下，父语言与子语言尽可能相似。

Here we devise a synthetic child language (called French?) which is exactly like French, except its vocabulary is shuffled randomly. (e.g., “internationale” is now “pomme,” etc). This language, which looks unintelligible to human eyes, nevertheless has the same distributional and relational properties as actual French, i.e. the word that, prior to vocabulary reassignment, was ‘roi’ (king) is likely to share distributional characteristics, and hence embedding similarity, to the word that, prior to reassignment, was ‘reine’ (queen). French should be the ideal par- ent model for French?.

在这里，我们设计了一种合成的儿童语言（称为法语？） 这和法语一模一样，只不过它的词汇是随机排列的。 （例如，“internationale”现在是“pomme”等）。 这一语言在人眼看来难以理解，但却具有与实际法语相同的分布和关系属性，即在词汇重新分配之前为“roi”（国王）的单词可能具有分布特征，因此嵌入了与重新分配之前为“reine”（皇后）的单词的相似性。 法语应该是法语的理想典范。 

The results of this experiment are shown in Table 6. We get a 4.3 BLEU improvement with an unrelated parent (i.e. French–parent and Uzbek– child), but we get a 6.7 BLEU improvement with a ‘closely related’ parent (i.e. French–parent and French?–child). We conclude that the choice of par- ent model can have a strong impact on transfer mod- els, and choosing better parents for our low-resource languages (if data for such parents can be obtained) could improve the final results.

实验结果见表6。 我们得到一个4.3 BLEU改进与一个无关的父母（即法国父母和乌兹别克孩子），但我们得到一个6.7 BLEU改进与一个‘密切相关’的父母（即法国父母和法国孩子）。 我们的结论是，Parent模型的选择对迁移模式有很大的影响，为我们的低资源语言选择更好的父语言（如果可以获得这样的父语言的数据）可以改善最终的结果。

![190910-Table6.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table6.png)

### 5.3 Ablation Analysis
In all the above experiments, only the target input and output embeddings are fixed during training. In this section we analyze what happens when different parts of the model are fixed, in order to determine the scenario that yields optimal performance. Figure 2 shows a diagram of the components of a sequence- to-sequence model. Table 7 shows the effects of al- lowing various components of the child NMT model to be trained. We find that the optimal setting for transferring from French–English to Uzbek–English in terms of BLEU performance is to allow all of the components of the child model to be trained except for the input and output target embeddings. 

在所有上述实验中，只有目标输入和输出嵌入在训练过程中是固定的。 在本节中，我们分析当模型的不同部分固定时会发生什么，以确定产生最佳性能的场景。 图2显示了序列到序列模型的组件图。 表7显示了要训练的儿童NMT模型的各个组成部分的效果。 我们发现，从BLEU性能的角度来看，从法语-英语转换到乌兹别克语-英语的最佳设置是允许训练除输入和输出目标嵌入之外的子模型的所有组件。

![190910-figure2.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-figure2.png)

![190910-Table7.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table7.png)

Even though we use this setting for our main experiments, the optimum setting is likely to be language- and corpus-dependent. For Turkish, experiments show that freezing attention parameters as well gives slightly better results. For parent-child models with closely related languages we expect freezing, or strongly regularizing, more components of the model to give better results.

即使我们使用这个设置为我们的主要实验，最优的设置可能是依赖于语言和语料库。 对于土耳其语，实验表明，冻结注意参数也会给出稍好的结果。 对于具有密切相关语言的父子模型，我们希望冻结或强正则化模型的更多组件，以获得更好的结果。 

### 5.4 Learning Curve
In Figure 3 we plot learning curves for both a transfer and a non-transfer model on training and development sets. We see that the final training set perplexities for both the transfer and non-transfer model are very similar, but the development set perplexity for the transfer model is much better. 

在图3中，我们绘制了培训和开发集上的迁移和非迁移模型的学习曲线。 我们发现，迁移模型和非迁移模型的最终训练集困惑非常相似，但迁移模型的发展集困惑要好得多。

![190910-figure3.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-figure3.png)

### 5.4 Learning Curve
In Figure 3 we plot learning curves for both a transfer and a non-transfer model on training and development sets. We see that the final training set perplexities for both the transfer and non-transfer model are very similar, but the development set perplexity for the transfer model is much better. 

在图3中，我们绘制了培训和开发集上的迁移和非迁移模型的学习曲线。 我们发现，迁移模型和非迁移模型的最终训练集困惑非常相似，但迁移模型的发展集困惑要好得多。 

![190910-figure3.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-figure3.png)

The fact that the two models start from and converge to very different points, yet have similar train- ing set performances, indicates that our architecture and training algorithm are able to reach a good minimum of the training objective regardless of the initialization. However, the training objective seems to have a large basin of models with similar performance and not all of them generalize well to the development set. The transfer model, starting with and staying close to a point known to perform well on a related task, is guided to a final point in the weight space that generalizes to the development set much better.

这两种模型从不同的点开始，收敛到不同的点，但训练集性能相似，表明无论初始化与否，我们的结构和训练算法都能很好地达到训练目标的最小值。 然而，培训目标似乎有一大盆性能相似的模型，并不是所有的模型都能很好地推广到开发集。 转移模型从一个已知在相关任务上表现良好的点开始并保持在该点附近，被引导到权重空间中的最后一点，该点更好地推广到开发集。 

### 5.5 Dictionary Initialization 
Using the transfer method, we always initialize input language embeddings for the child model with randomly-assigned embeddings from the par- ent (which has a different input language). A smarter method might be to initialize child embeddings with similar parent embeddings, where similarity is mea- sured by word-to-word t-table probabilities. To get these probabilities we compose Uzbek–English and English–French t-tables obtained from the Berkeley Aligner (Liang et al., 2006). We see from Figure 4 that this dictionary-based assignment results in faster improvement in the early part of the train- ing. However the final performance is similar to our standard model, indicating that the training is able to untangle the dictionary permutation introduced by randomly-assigned embeddings.

使用传输方法，我们总是使用来自角色（具有不同输入语言）的随机分配的嵌入来初始化子模型的输入语言嵌入。 一种更聪明的方法可能是用类似的父嵌入来初始化子嵌入，其中相似性由字到字的t表概率来度量。 为了得到这些概率，我们编写了从Berkeley Aligner获得的乌兹别克语-英语和英语-法语T表（Liang等人，2006年）。 从图4中我们可以看到，这种基于字典的赋值在训练的早期会导致更快的改进。 然而，最终的性能与我们的标准模型相似，这表明训练能够解开由随机分配的嵌入所引入的字典排列。 

![190910-figure4.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-figure4.png)

### 5.6 Different Parent Models
In the above experiments, we use a parent model trained on a large French–English corpus. One might hypothesize that our gains come from exploiting the English half of the corpus as an additional language model resource. Therefore, we explore transfer learning for the child model with parent models that only use the English side of the French– English corpus. We consider the following parent models in our ablative transfer learning scenarios:

在上述实验中，我们使用了一个在大型法语英语语料库上训练的父模型。 有人可能会假设，我们的收获来自于利用语料库的英语部分作为额外的语言模型资源。 因此，我们采用只使用英法语料库中英语部分的父母模式来探索儿童模式的迁移学习。 在消融性迁移学习情境中，我们考虑了以下父母模型: 

• A true translation model (French–English Par- ent)   
• A word-for-word English copying model (English–English Parent)  
• A model that unpermutes scrambled English (EngPerm–English Parent)  
• (The parameters of) an RNN language model (LM Xfer)  
• 真正的翻译模式（法英部分）   
• 逐字英文复制模式（英文-英文家长）   
• 不置乱英语的模式（engperm-english parent）   
•（的参数）RNN语言模型（LM xFER）   

The results, in Table 8, show that transfer learning does not simply import an English language model, but makes use of translation parameters learned from the parent’s large bilingual text.

表8中的结果表明，迁移学习不是简单地导入英语语言模型，而是利用从父母的大型双语文本中学习到的翻译参数。

![190910-Table8.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190910-Table8.png)

### 6 Conclusion
Overall, our transfer method improves NMT scores on low-resource languages by a large margin and al- lows our transfer NMT system to come close to the performance of a very strong SBMT system, even exceeding its performance on Hausa–English. In addition, we consistently and significantly improve state-of-the-art SBMT systems on low-resource languages when the transfer NMT system is used for re- scoring. Our experiments suggest that there is still room for improvement in selecting parent languages that are more similar to child languages, provided data for such parents can be found.

总体而言，我们的迁移方法大大提高了低资源语言的NMT成绩，使我们的迁移NMT系统接近于一个非常强大的SBMT系统的性能，甚至超过了Hausa-English系统的性能。 此外，在使用迁移NMT系统进行重新评分时，我们一致且显著地改进了低资源语言上的最新SBMT系统。 我们的实验表明，在选择更类似于子语言的父语言方面仍有改进的余地，前提是可以找到这类父语言的数据。