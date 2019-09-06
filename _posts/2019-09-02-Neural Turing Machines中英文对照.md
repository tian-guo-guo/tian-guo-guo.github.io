---
layout:     post                    # 使用的布局（不需要改）
title:      Neural Turing Machines中英文对照              # 标题 
subtitle:   NTM的学习 #副标题
date:       2019-09-02              # 时间
author:     甜果果                      # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 学习日记
    - paper
    - nlp
    - 机器翻译
---

> A. Graves et al., “Neural Turing Machines Alex,” Nat. Publ. Gr., vol. 538, no. 7626, pp. 471–476, 2014.

# Neural Turing Machines

## Abstract
We extend the capabilities of neural networks by coupling them to external memory re- sources, which they can interact with by attentional processes. The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-to- end, allowing it to be efficiently trained with gradient descent. Preliminary results demon- strate that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples.

我们通过将神经网络与外部记忆资源耦合来扩展神经网络的能力，神经网络可以通过注意过程与外部记忆资源交互。 该组合系统类似于图灵机或冯·诺伊曼结构，但是是可区分的端到端的，允许它以梯度下降的方式被有效地训练。 初步结果表明，神经图灵机可以从输入和输出示例中推断出简单的算法，如复制、排序和联想回忆。 

## 1 Introduction
Computer programs make use of three fundamental mechanisms: elementary operations (e.g., arithmetic operations), logical flow control (branching), and external memory, which can be written to and read from in the course of computation (Von Neumann, 1945). De- spite its wide-ranging success in modelling complicated data, modern machine learning has largely neglected the use of logical flow control and external memory.

计算机程序使用三种基本机制:基本运算（例如算术运算）、逻辑流控制（分支）和外部存储器，它们可以在计算过程中写入和读取（von Neumann，1945）。 尽管现代机器学习在复杂数据建模方面取得了广泛的成功，但它在很大程度上忽视了逻辑流量控制和外部存储器的使用。

Recurrent neural networks (RNNs) stand out from other machine learning methods for their ability to learn and carry out complicated transformations of data over extended periods of time. Moreover, it is known that RNNs are Turing-Complete (Siegelmann and Sontag, 1995), and therefore have the capacity to simulate arbitrary procedures, if properly wired. Yet what is possible in principle is not always what is simple in practice. We therefore enrich the capabilities of standard recurrent networks to simplify the solution of algorithmic tasks. This enrichment is primarily via a large, addressable memory, so, by analogy to Turing’s enrichment of finite-state machines by an infinite memory tape, we dub our device a “Neural Turing Machine” (NTM). Unlike a Turing machine, an NTM is a differentiable computer that can be trained by gradient descent, yielding a practical mechanism for learning programs.

递归神经网络（RecurrentNeuralNetworks，RNNs）以其在较长时间内学习和执行复杂的数据转换的能力而从其它机器学习方法中脱颖而出。 此外，已知RNN是Turing-Complete（Siegelmann和Sontag，1995），因此，if properly wired，RNN具有模拟任意过程的能力。 然而，原则上可能的事情并不总是在实践中简单的事情。 因此，我们丰富了标准递归网络的能力，简化了算法任务的求解。 这种充实主要是通过一个大的、可寻址的存储器，因此，通过类比图灵通过一个无限存储带对有限状态机的充实，我们将我们的设备称为“神经图灵机器”(NTM)。 与图灵机器不同，NTM是一种可微分的计算机，可以通过梯度下降来训练，从而产生一种实用的程序学习机制。

In human cognition, the process that shares the most similarity to algorithmic operation is known as “working memory.” While the mechanisms of working memory remain some- what obscure at the level of neurophysiology, the verbal definition is understood to mean a capacity for short-term storage of information and its rule-based manipulation (Badde- ley et al., 2009). In computational terms, these rules are simple programs, and the stored information constitutes the arguments of these programs. Therefore, an NTM resembles a working memory system, as it is designed to solve tasks that require the application of approximate rules to “rapidly-created variables.” Rapidly-created variables (Hadley, 2009) are data that are quickly bound to memory slots, in the same way that the number 3 and the number 4 are put inside registers in a conventional computer and added to make 7 (Minsky, 1967). An NTM bears another close resemblance to models of working memory since the NTM architecture uses an attentional process to read from and write to memory selectively. In contrast to most models ofworking memory, our architecture can learn to use its working memory instead of deploying a fixed set of procedures over symbolic data. 

在人类认知中，与算法操作最相似的过程被称为“工作记忆”。虽然工作记忆的机制在神经生理学层面上仍然有些模糊，但口头定义被理解为意味着信息的短期存储能力及其基于规则的操纵（Badde-ley等人，2009年）。 在计算术语中，这些规则是简单的程序，并且存储的信息构成这些程序的参数。 因此，NTM类似于工作存储器系统，因为它被设计为解决需要对“快速创建的变量”应用近似规则的任务。快速创建的变量（Hadley，2009）是快速绑定到存储器插槽的数据，就像数字3和数字4被放入常规计算机中的寄存器中并相加以生成7一样（Minsky，1967）。 NTM与工作记忆模型有另一个相似之处，因为NTM体系结构使用注意过程来选择性地读取和写入存储器。 与大多数工作内存模型不同，我们的体系结构可以学习使用它的工作内存，而不是在符号数据上部署一组固定的过程。

The organisation of this report begins with a brief review of germane research on working memory in psychology, linguistics, and neuroscience, along with related research in artificial intelligence and neural networks. We then describe our basic contribution, a mem- ory architecture and attentional controller that we believe is well-suited to the performance of tasks that require the induction and execution of simple programs. To test this architec- ture, we have constructed a battery of problems, and we present their precise descriptions along with our results. We conclude by summarising the strengths of the architecture.

本报告的组织首先简要回顾了心理学、语言学和神经科学中有关工作记忆的相关研究，以及人工智能和神经网络中的相关研究。 然后，我们描述了我们的基本贡献，一个内存结构和注意力控制器，我们认为它非常适合于执行需要归纳和执行简单程序的任务。 为了测试这种体系结构，我们构建了一组问题，并给出了它们的精确描述和结果。 最后，我们总结了该体系结构的优点。 

## 2 Foundational Research

### 2.1 Psychology and Neuroscience
The concept ofworking memory has been most heavily developed in psychology to explain the performance of tasks involving the short-term manipulation of information. The broad picture is that a “central executive” focuses attention and performs operations on data in a memory buffer (Baddeley et al., 2009). Psychologists have extensively studied the capacity limitations of working memory, which is often quantified by the number of “chunks” of information that can be readily recalled (Miller, 1956).1 These capacity limitations lead toward an understanding of structural constraints in the human working memory system, but in our own work we are happy to exceed them. 

工作记忆的概念在心理学中得到了很大的发展，用来解释涉及短期信息操纵的任务的执行情况。 总体情况是，“中央执行程序”集中注意力并对内存缓冲区中的数据执行操作（Baddeley等人，2009年）。 心理学家们对工作记忆的容量限制进行了广泛的研究，通常用容易回忆的信息“块”的数量来量化（Miller，1956）。1这些容量限制有助于理解人类工作记忆系统中的结构限制，但在我们自己的工作中，我们乐于超过它们。 

In neuroscience, the working memory process has been ascribed to the functioning of a system composed of the prefrontal cortex and basal ganglia (Goldman-Rakic, 1995). Typical experiments involve recording from a single neuron or group of neurons in prefrontal cortex while a monkey is performing a task that involves observing a transient cue, waiting through a “delay period,” then responding in a manner dependent on the cue. Certain tasks elicit persistent firing from individual neurons during the delay period or more complicated neural dynamics. A recent study quantified delay period activity in prefrontal cortex for a complex, context-dependent task based on measures of “dimensionality” of the population code and showed that it predicted memory performance (Rigotti et al., 2013). 

在神经科学中，工作记忆过程被认为是由前额叶皮层和基底神经节组成的系统的功能（Goldman-Rakic，1995）。 典型的实验包括从前额叶皮层的单个神经元或一组神经元进行记录，而猴子执行的任务包括观察一个短暂的线索，等待一段“延迟期”，然后以依赖于线索的方式作出反应。 某些任务在延迟期或更复杂的神经动力学过程中会引起单个神经元的持续性放电。 最近的一项研究基于群体编码的“维度”测量量化了复杂的上下文相关任务的前额叶皮层的延迟期活动，并表明它预测了记忆性能（Rigotti等人，2013年）。

Modeling studies of working memory range from those that consider how biophysical circuits could implement persistent neuronal firing (Wang, 1999) to those that try to solve explicit tasks (Hazy et al., 2006) (Dayan, 2008) (Eliasmith, 2013). Of these, Hazy et al.’s model is the most relevant to our work, as it is itself analogous to the Long Short-Term Memory architecture, which we have modified ourselves. As in our architecture, Hazy et al.’s has mechanisms to gate information into memory slots, which they use to solve a memory task constructed of nested rules. In contrast to our work, the authors include no sophisticated notion of memory addressing, which limits the system to storage and recall of relatively simple, atomic data. Addressing, fundamental to our work, is usually left out from computational models in neuroscience, though it deserves to be mentioned that Gallistel and King (Gallistel and King, 2009) and Marcus (Marcus, 2003) have argued that addressing must be implicated in the operation of the brain.

工作记忆的建模研究范围从考虑生物物理回路如何实现持续性神经元放电（Wang，1999）到试图解决显式任务（Hazy等人，2006）（Dayan，2008）（Eliasmith，2013）。 其中，Hazy等人的模型与我们的工作最相关，因为它本身类似于我们自己修改过的长期短期记忆体系结构。 与我们的体系结构一样，Hazy等人有机制将信息选通到内存插槽中，他们使用这些机制来解决由嵌套规则构建的内存任务。 与我们的工作相反，作者没有包含复杂的内存寻址概念，这限制了系统存储和调用相对简单的原子数据。 对我们的工作至关重要的寻址通常被排除在神经科学的计算模型之外，尽管值得一提的是，Gallistel and King（Gallistel and King，2009）和Marcus（Marcus，2003）认为，寻址必须与大脑的运作有关。 

### 2.2 Cognitive Science and Linguistics
Historically, cognitive science and linguistics emerged as fields at roughly the same time as artificial intelligence, all deeply influenced by the advent of the computer (Chomsky, 1956) (Miller, 2003). Their intentions were to explain human mental behaviour based on information or symbol-processing metaphors. In the early 1980s, both fields considered recursive or procedural (rule-based) symbol-processing to be the highest mark of cogni- tion. The Parallel Distributed Processing (PDP) or connectionist revolution cast aside the symbol-processing metaphor in favour of a so-called “sub-symbolic” description of thought processes (Rumelhart et al., 1986). 

从历史上看，认知科学和语言学几乎与人工智能同时作为一个领域出现，它们都深受计算机的出现的影响（Chomsky，1956年）（Miller，2003年）。 他们的意图是基于信息或符号处理隐喻来解释人类的心理行为。 在20世纪80年代早期，这两个领域都认为递归或过程（基于规则）符号处理是认知的最高标志。 并行分布处理（Parallel Distributed Processing，PDP）或连接主义革命抛弃了符号处理隐喻，转而采用所谓的“次符号”思维过程描述（Rumelhart等人，1986年）。

Fodor and Pylyshyn (Fodor and Pylyshyn, 1988) famously made two barbed claims about the limitations of neural networks for cognitive modeling. They first objected that connectionist theories were incapable of variable-binding, or the assignment of a particular datum to a particular slot in a data structure. In language, variable-binding is ubiquitous; for example, when one produces or interprets a sentence of the form, “Mary spoke to John,” one has assigned “Mary” the role of subject, “John” the role of object, and “spoke to” the role of the transitive verb. Fodor and Pylyshyn also argued that neural networks with fixed- length input domains could not reproduce human capabilities in tasks that involve process- ing variable-length structures. In response to this criticism, neural network researchers including Hinton (Hinton, 1986), Smolensky (Smolensky, 1990), Touretzky (Touretzky, 1990), Pollack (Pollack, 1990), Plate (Plate, 2003), and Kanerva (Kanerva, 2009) inves- tigated specific mechanisms that could support both variable-binding and variable-length structure within a connectionist framework. Our architecture draws on and potentiates this work.

Fodor和Pylyshyn（Fodor和Pylyshyn，1988）就神经网络用于认知建模的局限性提出了两个著名的尖锐的观点。 他们首先反对连接主义理论不能进行变量绑定，或者不能将特定的数据分配给数据结构中的特定槽。 在语言中，变量绑定无处不在； 例如，当一个人产生或解释一个句子的形式，“玛丽对约翰说话”，一个分配“玛丽”的角色的主语，“约翰”的角色的宾语，和“说话”的角色的及物动词。 Fodor和Pylyshyn还认为，具有固定长度输入域的神经网络不能在涉及处理可变长度结构的任务中再现人类的能力。 针对这一批评，包括辛顿（辛顿，1986年），斯摩棱斯基（斯摩棱斯基，1990年），图雷茨基（图雷茨基，1990年），波拉克（波拉克，1990年），普拉特（普拉特，2003年）和卡内尔瓦（卡内尔瓦，2009年）在内的神经网络研究人员研究了能够在连接论框架内支持可变绑定和可变长度结构的具体机制。 我们的架构借鉴并加强了这项工作。

Recursive processing of variable-length structures continues to be regarded as a hallmark of human cognition. In the last decade, a firefight in the linguistics community staked several leaders of the field against one another. At issue was whether recursive processing is the “uniquely human” evolutionary innovation that enables language and is specialized to language, a view supported by Fitch, Hauser, and Chomsky (Fitch et al., 2005), or whether multiple new adaptations are responsible for human language evolution and recursive processing predates language (Jackendoff and Pinker, 2005). Regardless of recursive process- ing’s evolutionary origins, all agreed that it is essential to human cognitive flexibility.

变长结构的递归处理一直被认为是人类认知的一个标志。 在过去的十年里，语言学界的一场交火使这一领域的几位领导人相互争斗。 争论的焦点是递归处理是“独特的人类”进化创新，它使语言成为可能，并且是语言的专门化，这一观点得到了惠誉、豪泽和乔姆斯基的支持（惠誉等人，2005年），还是多个新的适应是人类语言进化和递归处理早于语言的原因（Jackendoff和Pinker，2005年）。 不管递归过程的进化起源如何，所有人都认为递归过程对人类认知灵活性至关重要。 

### 2.3 Recurrent Neural Networks
Recurrent neural networks constitute a broad class of machines with dynamic state; that is, they have state whose evolution depends both on the input to the system and on the current state. In comparison to hidden Markov models, which also contain dynamic state, RNNs have a distributed state and therefore have significantly larger and richer memory and computational capacity. Dynamic state is crucial because it affords the possibility of context-dependent computation; a signal entering at a given moment can alter the behaviour of the network at a much later moment. 

递归神经网络是一类具有动态特性的机器； 也就是说，它们的状态的演化既取决于系统的输入，也取决于当前状态。 与隐马尔可夫模型相比，隐马尔可夫模型也包含动态，RNN具有分布式状态，因此具有更大和更丰富的内存和计算能力。 动态是至关重要的，因为它提供了上下文相关计算的可能性； 在给定时刻输入的信号可以在更晚的时刻改变网络的行为。

A crucial innovation to recurrent networks was the Long Short-Term Memory (LSTM) (Hochreiter and Schmidhuber, 1997). This very general architecture was developed for a specific purpose, to address the “vanishing and exploding gradient” problem (Hochreiter et al., 2001a), which we might relabel the problem of “vanishing and exploding sensitivity.” LSTM ameliorates the problem by embedding perfect integrators (Seung, 1998) for mem- ory storage in the network. The simplest example of a perfect integrator is the equation x(t + 1) = x(t) + i(t), where i(t) is an input to the system. The implicit identity matrix Ix(t) means that signals do not dynamically vanish or explode. If we attach a mechanism to this integrator that allows an enclosing network to choose when the integrator listens to inputs, namely, a programmable gate depending on context, we have an equation of the form x(t + 1) = x(t) + g(context)i(t). We can now selectively store information for an indefinite length of time. 

对循环网络的一个关键创新是长短时记忆(LSTM) （Hochreiter和Schmidhuber，1997年）。 这种非常通用的体系结构是为了解决“消失和爆炸梯度”问题（Hochreiter等人，2001a)而开发的，我们可以将“消失和爆炸灵敏度”问题重新命名为“消失和爆炸灵敏度”。LSTM通过在网络中嵌入用于记忆存储的完美积分器（Seung，1998）来改善该问题。 完美积分器最简单的例子是方程x(t+1)=x(t)+i(t)，其中i(t)是系统的输入。 隐式恒等式矩阵IX(t)意味着信号不会动态消失或爆炸。 如果我们在该积分器上附加一个机制，允许封闭网络选择积分器何时监听输入，即取决于上下文的可编程门，那么我们有一个形式为x(t+1)=x(t)+g(context)i(t)的公式。 我们现在可以有选择地将信息存储一段不确定的时间。

Recurrent networks readily process variable-length structures without modification. In sequential problems, inputs to the network arrive at different times, allowing variable- length or composite structures to be processed over multiple steps. Because they natively handle variable-length structures, they have recently been used in a variety of cognitive problems, including speech recognition (Graves et al., 2013; Graves and Jaitly, 2014), text generation (Sutskever et al., 2011), handwriting generation (Graves, 2013) and machine translation (Sutskever et al., 2014). Considering this property, we do not feel that it is ur- gent or even necessarily valuable to build explicit parse trees to merge composite structures greedily (Pollack, 1990) (Socher et al., 2012) (Frasconi et al., 1998). 

递归网络很容易处理变长结构而不需要修改。 在顺序问题中，网络的输入在不同的时间到达，允许可变长度或复合结构在多个步骤中处理。 由于它们本身处理可变长度结构，最近被用于各种认知问题，包括语音识别（Graves等人，2013年；Graves和Jaitly，2014年）、文本生成（Sutskever等人，2011年）、手写生成（Graves，2013年）和机器翻译（Sutskever等人，2014年）。 考虑到这一特性，我们认为构建显式解析树贪婪地合并复合结构并不迫切，甚至不一定有价值（Pollack，1990）（Socher等人，2012）（Frasconi等人，1998）。

Other important precursors to our work include differentiable models of attention (Graves, 2013) (Bahdanau et al., 2014) and program search (Hochreiter et al., 2001b) (Das et al., 1992), constructed with recurrent neural networks.

我们工作的其他重要前体包括可区分的注意力模型（Graves，2013年）（Bahdanau等人，2014年）和程序搜索（Hochreiter等人，2001年b）（Das等人，1992年），它们是用递归神经网络构建的。

![190902-fig1_Neural_Turing_Machine_Architecture.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig1_Neural_Turing_Machine_Architecture.jpg)

## 3 Neural Turing Machines
A Neural Turing Machine (NTM) architecture contains two basic components: a neural network controller and a memory bank. Figure 1 presents a high-level diagram of the NTM architecture. Like most neural networks, the controller interacts with the external world via input and output vectors. Unlike a standard network, it also interacts with a memory matrix using selective read and write operations. By analogy to the Turing machine we refer to the network outputs that parametrise these operations as “heads.” 

一种神经图灵机(NTM)结构包含两个基本组件:神经网络控制器和存储库。 图1给出了NTM体系结构的高级图。 与大多数神经网络一样，控制器通过输入和输出向量与外部世界交互。 与标准网络不同，它还使用选择性读写操作与存储器矩阵交互。 通过与图灵机器的类比，我们将参数化这些操作的网络输出称为“heads”。

Crucially, every component of the architecture is differentiable, making it straightforward to train with gradient descent. We achieved this by defining ‘blurry’ read and write operations that interact to a greater or lesser degree with all the elements in memory (rather than addressing a single element, as in a normal Turing machine or digital computer). The degree of blurriness is determined by an attentional “focus” mechanism that constrains each read and write operation to interact with a small portion of the memory, while ignoring the rest. Because interaction with the memory is highly sparse, the NTM is biased towards storing data without interference. The memory location brought into attentional focus is determined by specialised outputs emitted by the heads. These outputs define a normalised weighting over the rows in the memory matrix (referred to as memory “locations”). Each weighting, one per read or write head, defines the degree to which the head reads or writes at each location. A head can thereby attend sharply to the memory at a single location or weakly to the memory at many locations.

至关重要的是，体系结构的每个组件都是可区分的，这使得使用梯度下降进行训练变得非常简单。 我们通过定义“模糊”读写操作来实现这一点，这些操作或多或少地与内存中的所有元素进行交互（而不是像在普通的图灵机器或数字计算机中那样对单个元素进行寻址）。 模糊的程度由注意力“聚焦”机制决定，该机制约束每个读和写操作与存储器的一小部分交互，而忽略其余部分。 由于与存储器的交互非常稀疏，NTM倾向于无干扰地存储数据。 引起注意焦点的记忆位置由头部发出的特定输出决定。 这些输出定义对存储器矩阵中的行（称为存储器“位置”）的归一化加权。 每个加权（每个读写磁头一个）定义磁头在每个位置读写的程度。 因此，磁头可以在单个位置急剧地关注存储器，或者在许多位置微弱地关注存储器。 

### 3.1 Reading
Let Mt be the contents of the N × M memory matrix at time t, where N is the number of memory locations, and M is the vector size at each location. Let wt be a vector of weightings over the N locations emitted by a read head at time t. Since all weightings are normalised, the N elements wt(i) of wt obey the following constraints:

设mt是时刻t的n×m个存储矩阵的内容，其中n是存储位置的数目，m是每个位置处的向量大小。 设wt是在时刻t由读取头发射的n个位置上的权重的向量。 由于所有权重被归一化，WT的N个元素WT(i)遵守以下约束: 

![190902-equation1.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation1.png)

The length M read vector rt returned by the head is defined as a convex combination of the row-vectors Mt(i) in memory:

由磁头返回的长度M读取向量Rt被定义为存储器中的行向量Mt(i)的凸组合: 

![190902-equation2.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation2.png)

which is clearly differentiable with respect to both the memory and the weighting. 

这对于存储器和权重都是明显可微的。

### 3.2 Writing Taking
Taking inspiration from the input and forget gates in LSTM, we decompose each write into two parts: an erase followed by an add.  

从LSTM中的输入和遗忘门得到灵感，我们将每个写操作分解为两部分:擦除和添加。

Given a weighting wt emitted by a write head at time t, along with an erase vector et whose M elements all lie in the range (0, 1), the memory vectors Mt−1(i) from the previous time-step are modified as follows:

给定由写头在时间t发射的加权WT以及其M个元素都位于范围（0，1）中的擦除向量ET，将来自先前时间步长的存储器向量MT-1(i)修改如下:

![190902-equation3.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation3.png)

where 1 is a row-vector of all 1-s, and the multiplication against the memory location acts point-wise. Therefore, the elements of a memory location are reset to zero only if both the weighting at the location and the erase element are one; if either the weighting or the erase is zero, the memory is left unchanged. When multiple write heads are present, the erasures can be performed in any order, as multiplication is commutative. 

其中1是所有1-s的行向量，对存储器位置的乘法按点方向进行。 因此，只有当在位置处的加权和擦除元素都是1时，存储器位置的元素才被重置为零； 如果加权或擦除为零，则存储器保持不变。 当存在多个写入头时，可以以任何顺序执行擦除，因为乘法是可交换的。 

Each write head also produces a length M add vector at, which is added to the memoryafter the erase step has been performed: 

每个写入头还产生长度M相加向量AT，该长度M相加向量AT在执行擦除步骤之后被相加到存储器:

![190902-equation4.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation4.png)

Once again, the order in which the adds are performed by multiple heads is irrelevant. The combined erase and add operations of all the write heads produces the final content of the memory at time t. Since both erase and add are differentiable, the composite write operation is differentiable too. Note that both the erase and add vectors have M independent components, allowing fine-grained control over which elements in each memory location are modified.

同样，由多个头执行相加的顺序也是不相关的。 所有写入头的组合擦除和添加操作在时间t产生存储器的最终内容。 由于擦除和添加都是可微分的，所以复合写入操作也是可微分的。 注意，擦除向量和添加向量都有m个独立的组件，允许对每个内存位置中的哪些元素进行修改进行细粒度控制。 

![190902-fig2_Flow_Diagram_of_the_Addressing_Mechanism.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig2_Flow_Diagram_of_the_Addressing_Mechanism.jpg)

### 3.3 Addressing Mechanisms
Although we have now shown the equations of reading and writing, we have not described how the weightings are produced. These weightings arise by combining two addressing mechanisms with complementary facilities. The first mechanism, “content-based address- ing,” focuses attention on locations based on the similarity between their current values and values emitted by the controller. This is related to the content-addressing of Hopfield networks (Hopfield, 1982). The advantage of content-based addressing is that retrieval is simple, merely requiring the controller to produce an approximation to a part of the stored data, which is then compared to memory to yield the exact stored value. 

虽然我们现在已经展示了读和写的方程式，但我们还没有描述权重是如何产生的。 这些加权是通过将两个寻址机制与互补设施结合起来而产生的。 第一种机制“基于内容的寻址”基于它们的当前值和控制器发出的值之间的相似性将注意力集中在位置上。 这与Hopfield网络的内容寻址有关（Hopfield，1982）。 基于内容的寻址的优点是检索简单，仅需要控制器产生存储数据的一部分的近似值，然后将其与存储器进行比较以产生准确的存储值。

However, not all problems are well-suited to content-based addressing. In certain tasks the content of a variable is arbitrary, but the variable still needs a recognisable name or ad- dress. Arithmetic problems fall into this category: the variable x and the variable y can take on any two values, but the procedure f(x, y) = x × y should still be defined. A controller for this task could take the values of the variables x and y, store them in different addresses, then retrieve them and perform a multiplication algorithm. In this case, the variables are addressed by location, not by content. We call this form of addressing “location-based ad- dressing.” Content-based addressing is strictly more general than location-based addressing as the content of a memory location could include location information inside it. In our ex- periments however, providing location-based addressing as a primitive operation proved essential for some forms of generalisation, so we employ both mechanisms concurrently. 

然而，并非所有问题都非常适合基于内容的寻址。 在某些任务中，变量的内容是任意的，但是变量仍然需要一个可识别的名称或广告装扮。 算术问题属于这一类:变量x和变量y可以取任意两个值，但仍应定义过程f（x，y)=x×y。 该任务的控制器可以获取变量x和y的值，将它们存储在不同的地址中，然后检索它们并执行乘法算法。 在这种情况下，变量是按位置而不是按内容寻址的。 我们称这种寻址形式为“基于位置的处理”。基于内容的寻址比基于位置的寻址严格地更通用，因为存储器位置的内容可以包括其中的位置信息。 然而，在我们的实验中，提供基于位置的寻址作为一种原始操作被证明对于某些形式的概括是必不可少的，因此我们同时使用了这两种机制。

Figure 2 presents a flow diagram of the entire addressing system that shows the order of operations for constructing a weighting vector when reading or writing.

图2给出了整个寻址系统的流程图，其中显示了读取或写入时构造加权向量的操作顺序。

### 3.3.1 Focusing by Content
For content-addressing, each head (whether employed for reading or writing) first produces a length M key vector kt that is compared to each vector Mt(i) by a similarity measure K ? ·, · . The content-based system produces a normalised weighting wc ? t based on the similarity and a positive key strength, βt, which can amplify or attenuate the precision of the focus:

对于内容寻址，每个报头（无论是用于读还是写）首先产生长度M密钥向量Kt，该长度M密钥向量Kt通过相似性度量K？与每个向量Mt(i)进行比较。 ·，·。 基于内容的系统产生标准化加权WC？。 t基于相似性和正键强度βt，βt可以放大或衰减焦点的精度: 

![190902-equation5.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation5.png)

In our current implementation, the similarity measure is cosine similarity:

在我们当前的实现中，相似性度量是余弦相似性:

![190902-equation6.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation6.png)

### 3.3.2 Focusing by Location
The location-based addressing mechanism is designed to facilitate both simple iteration across the locations of the memory and random-access jumps. It does so by implementing a rotational shift of a weighting. For example, if the current weighting focuses entirely on a single location, a rotation of 1 would shift the focus to the next location. A negative shift would move the weighting in the opposite direction. 

基于位置的寻址机制被设计为便于跨越存储器位置的简单迭代和随机访问跳转。 它通过实现权重的旋转移位来实现这一点。 例如，如果当前权重完全集中在单个位置，则旋转1会将焦点转移到下一个位置。 一个负的变化将使权重向相反的方向移动。

Prior to rotation, each head emits a scalar interpolation gate gt in the range (0, 1). The value of g is used to blend between the weighting wt−1 produced by the head at the previous time-step and the weighting wct produced by the content system at the current time-step, yielding the gated weighting wgt :

在旋转之前，每个磁头发射范围（0，1）内的标量插值门GT。 g的值用于在前一时间步由头部产生的加权Wt−1和当前时间步由内容系统产生的加权Wct之间混合，产生选通加权Wgt:

![190902-equation7.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation7.png)

If the gate is zero, then the content weighting is entirely ignored, and the weighting from the previous time step is used. Conversely, if the gate is one, the weighting from the previous iteration is ignored, and the system applies content-based addressing. 

如果门为零，则完全忽略内容权重，并使用来自前一时间步骤的权重。 相反，如果门为1，则忽略来自上一次迭代的权重，并且系统应用基于内容的寻址。 

After interpolation, each head emits a shift weighting st that defines a normalised distribution over the allowed integer shifts. For example, if shifts between -1 and 1 are allowed, st has three elements corresponding to the degree to which shifts of -1, 0 and 1 are per- formed. The simplest way to define the shift weightings is to use a softmax layer of the appropriate size attached to the controller. We also experimented with another technique, where the controller emits a single scalar that is interpreted as the lower bound of a width one uniform distribution over shifts. For example, if the shift scalar is 6.7, then st(6) = 0.3, st(7) = 0.7, and the rest of st is zero.

在插值之后，每个磁头发出移位加权ST，该移位加权ST在允许的整数移位上定义归一化分布。 例如，如果允许-1和1之间的移位，则st具有与执行-1，0和1的移位的程度相对应的三个元素。 定义移位权重的最简单方法是使用连接到控制器的适当大小的SoftMax层。 我们还试验了另一种技术，其中控制器发出单个标量，该标量被解释为宽度的下界，宽度在移位上均匀分布。 例如，如果移位标量为6.7，则ST(6)=0.3，ST(7)=0.7，其余ST为0。 

If we index the N memory locations from 0 to N − 1, the rotation applied to wg t by st can be expressed as the following circular convolution: w˜t(i)

如果我们将N个存储器位置从0索引到N-1，则通过ST施加到WG T的旋转可以表示为以下循环卷积:

![190902-equation8.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation8.png)

where all index arithmetic is computed modulo N. The convolution operation in Equa- tion (8) can cause leakage or dispersion of weightings over time if the shift weighting is not sharp. For example, if shifts of -1, 0 and 1 are given weights of 0.1, 0.8 and 0.1, the rotation will transform a weighting focused at a single point into one slightly blurred over three points. To combat this, each head emits one further scalar γt ≥ 1 whose effect is to sharpen the final weighting as follows:

其中所有的指数算法都是模N计算的。如果移位加权不尖锐，则方程（8）中的卷积运算可能导致加权随时间的泄漏或分散。 例如，如果-1，0和1的移位被赋予0.1，0.8和0.1的权重，则旋转将把聚焦在单个点的权重转换为在三个点上稍微模糊的权重。 为了克服这一问题，每个磁头再发射一个标量γt≥1，其效果是使最终加权锐化，如下所示:

![190902-equation9.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation9.png)

The combined addressing system of weighting interpolation and content and location-based addressing can operate in three complementary modes. One, a weighting can be chosen by the content system without any modification by the location system. Two, a weighting produced by the content addressing system can be chosen and then shifted. This allows the focus to jump to a location next to, but not on, an address accessed by content; in computational terms this allows a head to find a contiguous block of data, then access a particular element within that block. Three, a weighting from the previous time step can be rotated without any input from the content-based addressing system. This allows the weighting to iterate through a sequence of addresses by advancing the same distance at each time-step.

加权内插和内容与基于位置的寻址的组合寻址系统可以以三种互补模式操作。 第一，可以由内容系统选择权重，而不需要位置系统进行任何修改。 第二，可以选择由内容寻址系统产生的权重，然后移位。 这允许焦点跳转到内容访问的地址旁边的位置，而不是跳转到内容访问的地址上； 在计算术语中，这允许头部查找连续的数据块，然后访问该块中的特定元素。 第三，可以在不需要来自基于内容的寻址系统的任何输入的情况下旋转来自先前时间步长的权重。 这允许加权通过在每个时间步长上推进相同的距离来迭代地址序列。 

## 3.4 Controller Network
The NTM architecture architecture described above has several free parameters, including the size of the memory, the number of read and write heads, and the range of allowed lo- cation shifts. But perhaps the most significant architectural choice is the type of neural network used as the controller. In particular, one has to decide whether to use a recurrent or feedforward network. A recurrent controller such as LSTM has its own internal memory that can complement the larger memory in the matrix. If one compares the controller to the central processing unit in a digital computer (albeit with adaptive rather than predefined instructions) and the memory matrix to RAM, then the hidden activations of the recurrent controller are akin to the registers in the processor. They allow the controller to mix information across multiple time steps of operation. On the other hand a feedforward controller can mimic a recurrent network by reading and writing at the same location in memory at every step. Furthermore, feedforward controllers often confer greater transparency to the network’s operation because the pattern of reading from and writing to the memory matrix is usually easier to interpret than the internal state of an RNN. However, one limitation of a feedforward controller is that the number of concurrent read and write heads imposes a bottleneck on the type of computation the NTM can perform. With a single read head, it can perform only a unary transform on a single memory vector at each time-step, with two read heads it can perform binary vector transforms, and so on. Recurrent controllers can internally store read vectors from previous time-steps, so do not suffer from this limitation.

上面描述的NTM体系结构具有几个空闲参数，包括存储器的大小、读和写头的数目以及允许的位置偏移的范围。 但最重要的架构选择可能是用作控制器的神经网络类型。 尤其是，必须决定是使用递归网络还是前馈网络。 像LSTM这样的递归控制器有它自己的内部存储器，可以补充矩阵中较大的存储器。 如果将控制器与数字计算机中的中央处理单元（尽管具有自适应指令而不是预定义指令）和存储器矩阵与RAM进行比较，则循环控制器的隐藏激活类似于处理器中的寄存器。 它们允许控制器在操作的多个时间步骤中混合信息。 另一方面，前馈控制器可以通过在每个步骤中在存储器中的相同位置读取和写入来模仿递归网络。 此外，前馈控制器通常赋予网络的操作更大的透明性，因为读取和写入存储器矩阵的模式通常比RNN的内部状态更容易解释。 然而，前馈控制器的一个限制是并发读写头的数目对NTM可执行的计算类型施加瓶颈。 使用单个读取头，它在每个时间步只能对单个存储向量执行一元变换，使用两个读取头，它可以执行二进制向量变换，依此类推。 循环控制器可以在内部存储先前时间步长的读取向量，因此不受此限制。

## 4 Experiments
This section presents preliminary experiments on a set of simple algorithmic tasks such as copying and sorting data sequences. The goal was not only to establish that NTM is able to solve the problems, but also that it is able to do so by learning compact internal programs. The hallmark of such solutions is that they generalise well beyond the range of the training data. For example, we were curious to see if a network that had been trained to copy sequences of length up to 20 could copy a sequence of length 100 with no further training. 

本节介绍一组简单算法任务的初步实验，例如复制和排序数据序列。 其目标不仅是确定NTM能够解决这些问题，而且它能够通过学习紧凑的内部程序来解决这些问题。 这种解决方案的特点是它们的泛化能力远远超出了训练数据的范围。 例如，我们很想看看，一个经过训练可以复制长度最多为20的序列的网络，是否可以复制长度为100的序列，而无需进一步的训练。

For all the experiments we compared three architectures: NTM with a feedforward controller, NTM with an LSTM controller, and a standard LSTM network. Because all the tasks were episodic, we reset the dynamic state of the networks at the start of each input sequence. For the LSTM networks, this meant setting the previous hidden state equal to a learned bias vector. For NTM the previous state of the controller, the value of the previous read vectors, and the contents of the memory were all reset to bias values. All the tasks were supervised learning problems with binary targets; all networks had logistic sigmoid output layers and were trained with the cross-entropy objective function. Sequence prediction errors are reported in bits-per-sequence. For more details about the experimental parameters see Section 4.6.

对于所有的实验，我们比较了三种体系结构:NTM与前馈控制器、NTM与LSTM控制器和标准LSTM网络。 因为所有的任务都是偶发的，所以我们在每个输入序列的开头重置网络的动态状态。 对于LSTM网络，这意味着将先前的隐藏状态设置为等于所学习的偏置向量。 对于NTM，控制器的先前状态、先前读取向量的值和存储器的内容都被重置为偏置值。 所有任务都是二值目标的有监督学习问题； 所有网络都具有Logistic Sigmoid输出层，并用交叉熵目标函数进行训练。 序列预测误差以每序列比特为单位报告。 有关实验参数的更多详细信息，请参见第4.6节。

### 4.1 Copy
The copy task tests whether NTM can store and recall a long sequence of arbitrary in- formation. The network is presented with an input sequence of random binary vectors followed by a delimiter flag. Storage and access of information over long time periods has always been problematic for RNNs and other dynamic architectures. We were particularly interested to see if an NTM is able to bridge longer time delays than LSTM. 

复制任务测试NTM是否可以存储和调用任意信息的长序列。 该网络具有随机二进制向量的输入序列，随后是定界符标志。 对于RNN和其他动态体系结构来说，长时间的信息存储和访问一直是个问题。 我们特别感兴趣的是，NTM是否能够桥接比LSTM更长的时间延迟。

The networks were trained to copy sequences of eight bit random vectors, where the sequence lengths were randomised between 1 and 20. The target sequence was simply a copy of the input sequence (without the delimiter flag). Note that no inputs were presented to the network while it receives the targets, to ensure that it recalls the entire sequence with no intermediate assistance. 

训练网络复制8位随机向量的序列，其中序列长度在1和20之间随机化。 目标序列只是输入序列的副本（没有分隔符标志）。 请注意，在网络接收目标时，没有向其提供任何输入，以确保它在没有中间协助的情况下调用整个序列。

As can be seen from Figure 3, NTM (with either a feedforward or LSTM controller) learned much faster than LSTM alone, and converged to a lower cost. The disparity be- tween the NTM and LSTM learning curves is dramatic enough to suggest a qualitative, rather than quantitative, difference in the way the two models solve the problem.

如图3所示，NTM（使用前馈或LSTM控制器）的学习速度比单独使用LSTM快得多，并且收敛到更低的成本。 NTM和LSTM学习曲线之间的差异足以说明两种模型在解决问题的方式上的定性差异，而不是定量差异。

![190902-fig3_Copy_Learning_Curves.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig3_Copy_Learning_Curves.jpg)

We also studied the ability of the networks to generalise to longer sequences than seen during training (that they can generalise to novel vectors is clear from the training error). Figures 4 and 5 demonstrate that the behaviour of LSTM and NTM in this regime is rad- ically different. NTM continues to copy as the length increases2, while LSTM rapidly degrades beyond length 20.

我们还研究了网络泛化到比训练期间更长的序列的能力（从训练误差中可以清楚地看出它们可以泛化到新的向量）。 图4和图5表明，LSTM和NTM在该机制中的行为截然不同。 当长度增加2时，NTM继续复制，而LSTM在长度20之后迅速退化。 

![190902-fig4_NTM_Generalisation_on_the_Copy_task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig4_NTM_Generalisation_on_the_Copy_task.jpg)

![190902-fig5_LSTM_Generalisation_on_the_Copy_task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig5_LSTM_Generalisation_on_the_Copy_task.jpg)

The preceding analysis suggests that NTM, unlike LSTM, has learned some form of copy algorithm. To determine what this algorithm is, we examined the interaction between the controller and the memory (Figure 6). We believe that the sequence of operations per- formed by the network can be summarised by the following pseudocode:

前面的分析表明，NTM不同于LSTM，它学习了某种形式的复制算法。 为了确定这个算法是什么，我们检查了控制器和内存之间的交互（图6）。 我们认为，网络执行的操作顺序可以用以下伪代码概括:

```
initialise: move head to start location 
while input delimiter not seen do 
    receive input vector 
    write input to head location 
    increment head location by 1
end while 
return head to start location 
while true do 
    read output vector from head location 
    emit output 
    increment head location by 1
end while This
```

![190902-fig6_NTM_Memory_Use_During_the_Copy_task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig6_NTM_Memory_Use_During_the_Copy_task.jpg)

This is essentially how a human programmer would perform the same task in a low-level programming language. In terms of data structures, we could say that NTM has learned how to create and iterate through arrays. Note that the algorithm combines both content-based addressing (to jump to start of the sequence) and location-based address- ing (to move along the sequence). Also note that the iteration would not generalise to long sequences without the ability to use relative shifts from the previous read and write weightings (Equation 7), and that without the focus-sharpening mechanism (Equation 9) the weightings would probably lose precision over time.
这是人类程序员在低级编程语言中执行相同任务的基本方式。 在数据结构方面，我们可以说NTM已经学会了如何创建和迭代数组。 注意，该算法结合了基于内容的寻址（跳转到序列的开始）和基于位置的寻址（沿着序列移动）。 还请注意，如果不能使用先前读写权重的相对移位，迭代将不会推广到长序列（等式7），并且如果没有焦点锐化机制（等式9），权重可能会随着时间而失去精度。 

### 4.2 Repeat Copy
The repeat copy task extends copy by requiring the network to output the copied sequence a specified number of times and then emit an end-of-sequence marker. The main motivation was to see if the NTM could learn a simple nested function. Ideally, we would like it to be able to execute a “for loop” containing any subroutine it has already learned. 

重复复制任务通过要求网络输出指定次数的复制序列，然后发出序列结束标记来扩展复制。 主要的动机是看看NTM是否可以学习一个简单的嵌套函数。 理想情况下，我们希望它能够执行一个“for循环”，其中包含它已经学习到的任何子程序。

The network receives random-length sequences of random binary vectors, followed by a scalar value indicating the desired number of copies, which appears on a separate input channel. To emit the end marker at the correct time the network must be both able to interpret the extra input and keep count of the number of copies it has performed so far. As with the copy task, no inputs are provided to the network after the initial sequence and repeat number. The networks were trained to reproduce sequences of size eight random binary vectors, where both the sequence length and the number of repetitions were chosen randomly from one to ten. The input representing the repeat number was normalised to have mean zero and variance one.

网络接收随机二进制向量的随机长度序列，然后是标量值，标量值指示期望的拷贝数，标量值出现在单独的输入通道上。 为了在正确的时间发出结束标记，网络必须能够解释额外的输入，并记录到目前为止执行的副本数。 与复制任务一样，在初始序列和重复号之后，不向网络提供输入。 训练网络以再现大小为8的随机二进制向量的序列，其中序列长度和重复次数都是从1到10随机选择的。 表示重复数的输入被归一化为平均0和方差1。 

Figure 7 shows that NTM learns the task much faster than LSTM, but both were able to solve it perfectly.3 The difference between the two architectures only becomes clear when they are asked to generalise beyond the training data. In this case we were interested in generalisation along two dimensions: sequence length and number of repetitions. Figure 8 illustrates the effect of doubling first one, then the other, for both LSTM and NTM. Whereas LSTM fails both tests, NTM succeeds with longer sequences and is able to perform more than ten repetitions; however it is unable to keep count of of how many repeats it has completed, and does not predict the end marker correctly. This is probably a consequence of representing the number of repetitions numerically, which does not easily generalise beyond a fixed range. 

图7显示，NTM学习任务的速度比LSTM快得多，但两者都能够很好地解决它。3只有当两种体系结构被要求在训练数据之外进行概括时，它们之间的区别才会变得明显。 在这种情况下，我们感兴趣的是沿着两个维度的概括:序列长度和重复次数。 图8说明了LSTM和NTM先加倍，然后再加倍的效果。 虽然LSTM两次测试都失败了，NTM成功的序列更长，并且能够执行十次以上的重复； 但是，它无法计数已完成的重复次数，并且不能正确预测结束标记。 这可能是重复次数用数字表示的结果，而重复次数不容易泛化到固定范围之外。

![190902-fig7_Repeat_Copy_Learning_Curves.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig7_Repeat_Copy_Learning_Curves.jpg)

![190902-fig8_NTM_and_LSTM_Generalisation_for_the_Repeat_Copy_Task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig8_NTM_and_LSTM_Generalisation_for_the_Repeat_Copy_Task.jpg)

Figure 9 suggests that NTM learns a simple extension of the copy algorithm in the previous section, where the sequential read is repeated as many times as necessary.

图9表明NTM在上一节中学习了复制算法的一个简单扩展，其中顺序读取可以根据需要重复多次。

![190902-fig9_NTM_Memory_Use_During_the_Repeat_Copy_task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig9_NTM_Memory_Use_During_the_Repeat_Copy_task.jpg)

### 4.3 Associative Recall

The previous tasks show that the NTM can apply algorithms to relatively simple, linear data structures. The next order of complexity in organising data arises from “indirection”—that is, when one data item points to another. We test the NTM’s capability for learning an instance of this more interesting class by constructing a list of items so that querying with one of the items demands that the network return the subsequent item. More specifically, we define an item as a sequence of binary vectors that is bounded on the left and right by delimiter symbols. After several items have been propagated to the network, we query by showing a random item, and we ask the network to produce the next item. In our experiments, each item consisted of three six-bit binary vectors (giving a total of 18 bits per item). During training, we used a minimum of 2 items and a maximum of 6 items in a single episode.

前面的任务表明NTM可以将算法应用于相对简单的线性数据结构。 组织数据的下一个复杂顺序来自“间接”，即当一个数据项指向另一个数据项时。 我们通过构造一个项列表来测试NTM学习这个更有趣的类的实例的能力，以便查询其中一个项要求网络返回后续项。 更具体地说，我们将项定义为由分隔符符号在左侧和右侧界定的二进制向量序列。 在几个项被传播到网络之后，我们通过显示一个随机项进行查询，并要求网络生成下一个项。 在我们的实验中，每个项目由三个六位二进制向量组成（给出每个项目总共18位）。 在训练中，我们在一集里最少使用了2个项目，最多使用了6个项目。

Figure 10 shows that NTM learns this task significantly faster than LSTM, terminating at near zero cost within approximately 30, 000 episodes, whereas LSTM does not reach zero cost after a million episodes. Additionally, NTM with a feedforward controller learns faster than NTM with an LSTM controller. These two results suggest that NTM’s external memory is a more effective way of maintaining the data structure than LSTM’s internal state. NTM also generalises much better to longer sequences than LSTM, as can be seen in Figure 11. NTM with a feedforward controller is nearly perfect for sequences of up to 12 items (twice the maximum length used in training), and still has an average cost below 1 bit per sequence for sequences of 15 items. 

图10显示NTM学习该任务的速度比LSTM快得多，在大约30000集内以接近零的成本结束，而LSTM在一百万集后没有达到零成本。 另外，具有前馈控制器的NTM比具有LSTM控制器的NTM学习更快。 这两个结果表明NTM的外部存储器比LSTM的内部状态更有效地维护数据结构。 NTM比LSTM更好地将序列推广到更长的序列，如图11所示。 具有前馈控制器的NTM对于多达12个项目的序列（训练中使用的最大长度的两倍）几乎是完美的，并且对于15个项目的序列，每个序列的平均成本仍然低于1位。

![190902-fig10_Associative_Recall_Learning_Curves_for_NTM_and_LSTM.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig10_Associative_Recall_Learning_Curves_for_NTM_and_LSTM.jpg)

![190902-fig11_Generalisation_Performance_on_Associative_Recall_for_Longer_Item_Sequences.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig11_Generalisation_Performance_on_Associative_Recall_for_Longer_Item_Sequences.jpg)

In Figure 12, we show the operation of the NTM memory, controlled by an LSTM with one head, on a single test episode. In “Inputs,” we see that the input denotes item delimiters as single bits in row 7. After the sequence of items has been propagated, a delimiter in row 8 prepares the network to receive a query item. In this case, the query item corresponds to the second item in the sequence (contained in the green box). In “Outputs,” we see that the network crisply outputs item 3 in the sequence (from the red box). In “Read Weightings,” on the last three time steps, we see that the controller reads from contiguous locations that each store the time slices of item 3. This is curious because it appears that the network has jumped directly to the correct location storing item 3. However we can explain this behaviour by looking at “Write Weightings.” Here we see that the memory is written to even when the input presents a delimiter symbol between items. One can confirm in “Adds” that data are indeed written to memory when the delimiters are presented (e.g., the data within the black box); furthermore, each time a delimiter is presented, the vector added to memory is different. Further analysis of the memory reveals that the network accesses the location it reads after the query by using a content-based lookup that produces a weighting that is shifted by one. Additionally, the key used for content-lookup corresponds to the vector that was added in the black box. This implies the following memory-access algorithm: when each item delimiter is presented, the controller writes a compressed representation of the previous three time slices of the item. After the query arrives, the controller recomputes the same compressed representation of the query item, uses a content-based lookup to find the location where it wrote the first representation, and then shifts by one to produce the subsequent item in the sequence (thereby combining content-based lookup with location-based offsetting).

In Figure 12, 我们展示了NTM存储器的操作，由一个单头LSTM控制，在单个测试集上。 在“inputs”中，我们看到输入将项分隔符表示为第7行中的单个位。 在传播项序列之后，第8行中的分隔符准备网络接收查询项。 在这种情况下，查询项对应于序列中的第二项（包含在绿色框中）。 在“outputs”中，我们看到网络按顺序（从红色框中）清晰地输出项3。 在“读取权重”中，在最后三个时间步骤中，我们看到控制器从每个存储项3的时间片的相邻位置读取。 这很奇怪，因为网络似乎已直接跳转到存储项3的正确位置。 然而，我们可以通过查看“写入权重”来解释这种行为。在这里，我们可以看到，即使当输入在项之间呈现分隔符符号时，也会写入内存。 可以在“添加”中确认，当出现分隔符时，数据确实被写入存储器（例如，黑盒中的数据）； 此外，每次出现分隔符时，添加到内存中的向量都是不同的。 对存储器的进一步分析揭示，网络通过使用基于内容的查找来访问它在查询之后读取的位置，该查找产生被移位1的权重。 此外，用于内容查找的键对应于添加到黑框中的向量。 这意味着以下内存访问算法:当每个项分隔符出现时，控制器将写入项的前三个时间片的压缩表示。 在查询到达之后，控制器重新计算查询项的相同压缩表示，使用基于内容的查找来查找它写入第一表示的位置，然后移位1以产生序列中的后续项（由此组合基于内容的查找和基于位置的偏移）。

![190902-fig12_NTM_Memory_Use_During_the_Associative_Recall_Task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig12_NTM_Memory_Use_During_the_Associative_Recall_Task.jpg)

### 4.4 Dynamic N-Grams

The goal of the dynamic N-Grams task was to test whether NTM could rapidly adapt to new predictive distributions. In particular we were interested to see if it were able to use its memory as a re-writable table that it could use to keep count of transition statistics, thereby emulating a conventional N-Gram model.

动态n-grams任务的目标是测试NTM能否快速适应新的预测分布。 特别是，我们有兴趣了解它是否能够将内存用作可重写的表，用于保持转换统计信息的计数，从而模拟传统的n元模型。

We considered the set of all possible 6-Gram distributions over binary sequences. Each 6-Gram distribution can be expressed as a table of 25 = 32 numbers, specifying the prob- ability that the next bit will be one, given all possible length five binary histories. For each training example, we first generated random 6-Gram probabilities by independently drawing all 32 probabilities from the Beta(1 2 , 1 2) distribution.

我们考虑了二进制序列上所有可能的6-Gram分布的集合。 每个6Gram分布可以表示为一个2^5=32的表，在给定所有可能长度的5个二进制历史的情况下，指定下一位为1的可能性。 对于每个训练示例，我们首先通过从beta（1/2，1/2）分布独立地绘制所有32个概率来生成随机6克概率。

We then generated a particular training sequence by drawing 200 successive bits using the current lookup table.4 The network observes the sequence one bit at a time and is then asked to predict the next bit. The optimal estimator for the problem can be determined by where c is the five bit previous context, B is the value of the next bit and N0 and N1 are respectively the number of zeros and ones observed after c so far in the sequence. We can therefore compare NTM to the optimal predictor as well as LSTM. To assess performance we used a validation set of 1000 length 200 sequences sampled from the same distribu- tion as the training data. As shown in Figure 13, NTM achieves a small, but significant performance advantage over LSTM, but never quite reaches the optimum cost. 

然后，我们通过使用当前查找表绘制200个连续比特来生成特定的训练序列。网络一次观察一个比特，然后被要求预测下一个比特。 该问题的最优估计量可由以下公式确定:

![190902-equation10.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-equation10.png)

其中C是前一个5位上下文，B是下一位的值，N0和N1分别是序列中到目前为止在C之后观察到的0和1的数目。 因此，我们可以将NTM与最优预测器以及LSTM进行比较。 为了评估性能，我们使用了1000个长度为200个序列的验证集，这些序列是从与训练数据相同的分布中采样的。 如图13所示，与LSTM相比，NTM实现了较小但显著的性能优势，但从未达到最佳成本。

![190902-fig13_Dynamic_N-Gram_Learning_Curves.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig13_Dynamic_N-Gram_Learning_Curves.jpg)

The evolution of the two architecture’s predictions as they observe new inputs is shown in Figure 14, along with the optimal predictions. Close analysis of NTM’s memory usage (Figure 15) suggests that the controller uses the memory to count how many ones and zeros it has observed in different contexts, allowing it to implement an algorithm similar to the optimal estimator.

图14显示了两种体系结构在观察新输入时的预测演变，以及最佳预测。 对NTM内存使用情况的详细分析（图15）表明，控制器使用内存来计数它在不同上下文中观察到的1和0的数量，从而实现类似于最优估计器的算法。

![190902-fig14_Dynamic_N-Gram_Inference.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig14_Dynamic_N-Gram_Inference.jpg)

![190902-fig15_NTM_Memory_Use_During_the_Dynamic_N-Gram_Task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig15_NTM_Memory_Use_During_the_Dynamic_N-Gram_Task.jpg)

### 4.5 Priority Sort
This task tests whether the NTM can sort data—an important elementary algorithm. A sequence of random binary vectors is input to the network along with a scalar priority rating for each vector. The priority is drawn uniformly from the range [-1, 1]. The target sequence contains the binary vectors sorted according to their priorities, as depicted in Figure 16. 

本任务测试NTM是否能够对数据进行排序——这是一个重要的基本算法。 随机二进制向量序列与每个向量的标量优先级一起被输入到网络。 优先级统一从范围[-1，1]中提取。 目标序列包含根据优先级排序的二进制向量，如图16所示。

![190902-fig16_Example_Input_and_Target_Sequence_for_the_Priority_Sort_Task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig16_Example_Input_and_Target_Sequence_for_the_Priority_Sort_Task.jpg)

Each input sequence contained 20 binary vectors with corresponding priorities, and each target sequence was the 16 highest-priority vectors in the input.5 Inspection of NTM’s memory use led us to hypothesise that it uses the priorities to determine the relative location of each write. To test this hypothesis we fitted a linear function of the priority to the observed write locations. Figure 17 shows that the locations returned by the linear function closely match the observed write locations. It also shows that the network reads from the memory locations in increasing order, thereby traversing the sorted sequence.

每个输入序列包含20个具有相应优先级的二进制向量，并且每个目标序列是输入中的16个最高优先级向量。对NTM的存储器使用的检查导致我们假设它使用优先级来确定每个写入的相对位置。 为了检验这个假设，我们将优先级的线性函数拟合到观察到的写入位置。 图17显示了线性函数返回的位置与观察到的写入位置非常匹配。 它还显示网络以递增的顺序从存储器位置读取，从而遍历排序的序列。

![190902-fig17_NTM_Memory_Use_During_the_Priority_Sort_Task.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig17_NTM_Memory_Use_During_the_Priority_Sort_Task.jpg)

The learning curves in Figure 18 demonstrate that NTM with both feedforward and LSTM controllers substantially outperform LSTM on this task. Note that eight parallel read and write heads were needed for best performance with a feedforward controller on this task; this may reflect the difficulty of sorting vectors using only unary vector operations (see Section 3.4).

图18中的学习曲线表明，具有前馈和LSTM控制器的NTM在此任务上的性能显著优于LSTM。 请注意，使用前馈控制器执行此任务时，需要8个并行读写头才能获得最佳性能； 这可能反映了仅使用一元向量运算对向量排序的困难（见第3.4节）。

![190902-fig18_Priority_Sort_Learning_Curves.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-fig18_Priority_Sort_Learning_Curves.jpg)

### 4.6 Experimental Details

For all experiments, the RMSProp algorithm was used for training in the form described in (Graves, 2013) with momentum of 0.9. Tables 1 to 3 give details about the network configurations and learning rates used in the experiments. All LSTM networks had three stacked hidden layers. Note that the number of LSTM parameters grows quadratically with the number of hidden units (due to the recurrent connections in the hidden layers). This contrasts with NTM, where the number of parameters does not increase with the number of memory locations. During the training backward pass, all gradient components are clipped elementwise to the range (-10, 10).

对于所有的实验，RMSprop算法被用于以Graves，2013中描述的形式（动量为0.9）进行训练。 表1至表3给出了实验中使用的网络配置和学习速率的详细信息。 所有LSTM网络都有三层堆叠的隐藏层。 请注意，LSTM参数的数量随隐藏单元的数量（由于隐藏层中的重复连接）呈二次增长。 这与NTM不同，NTM中参数的数量不随内存位置的数量增加而增加。 在训练向后传球时，所有梯度分量都被基本削波到范围（-10，10）。

![190902-Table1-NTM_with_Feedforward_Controller_Experimental_Settings.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-Table1-NTM_with_Feedforward_Controller_Experimental_Settings.png)

![190902-Table2-NTM_with_LSTM_Controller_Experimental_Settings.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-Table2-NTM_with_LSTM_Controller_Experimental_Settings.png)

![190902-Table3-LSTM_Network_Experimental_Settings.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190902-Table3-LSTM_Network_Experimental_Settings.png)

## 5 Conclusion

We have introduced the Neural Turing Machine, a neural network architecture that takes inspiration from both models of biological working memory and the design of digital computers. Like conventional neural networks, the architecture is differentiable end-to-end and can be trained with gradient descent. Our experiments demonstrate that it is capable of learning simple algorithms from example data and of using these algorithms to generalise well outside its training regime.

我们介绍了神经图灵机，这是一种从生物工作记忆模型和数字计算机设计中得到启发的神经网络结构。 与传统的神经网络一样，该结构是端到端可微分的，并且可以通过梯度下降进行训练。 我们的实验表明，它能够从示例数据中学习简单的算法，并且能够使用这些算法在其训练范围之外很好地进行推广。