#Sentence Embedding
1、主要实现方法。
词嵌入主要是通过C-Bow或者是skip-gram实现的，基本的句嵌入就是通过在实现词向量的时候加入一个表示句子的向量来实现句嵌入
 
现在句子编码主要研究如何有效地从词嵌入通过不同方式的组合得到句子表示。其中，比较有代表性的方法有四种。
（1）神经词袋模型
	简单对文本序列中每个词嵌入进行平均/加总，作为整个序列的表示。
	这种方法的缺点是丢失了词序信息。对于长文本，神经词袋模型比较有效。但是对于短文本，神经词袋模型很难捕获语义组合信息。
（2）递归神经网络（Recursive Neural Network）
	按照一个给定的外部拓扑结构（比如成分句法树），不断递归得到整个序列的表示。递归神经网络的一个缺点是需要给定一个拓扑结构来确定词和词之间的依赖      关系，因此限制其使用范围。
（3）循环神经网络（Recurrent Neural Network）
	将文本序列看作时间序列，不断更新，最后得到整个序列的表示。
（4）卷积神经网络（Convolutional Neural Network）
	通过多个卷积层和子采样层，最终得到一个固定长度的向量。

主要问题：
	句向量主要应用在情感分析，分类、相似性比较等标注工作上的效果比较好。
	句向量设计到词性、时态等，现在中文方面几乎没有有效工作
	句向量现在没有一个公认的好的完善的编码方法。都还在探索中。

2、《Supervised Learning of Universal Sentence Representations from NLI data》
   长期以来，语句嵌入的监督训练被认为是比无监督的方法提供更低质量的嵌入，但是这个假设因为这篇文章被推翻了。
   InferSent是一种结构简单而有趣的方法。它使用语句自然语言推理（Sentence Natural Language Inference）数据集，在语句编码器的顶部训练分类器。
这两个语句使用相同的编码器进行编码，而分类器在从两个语句嵌入构建的一对表示上被训练。Conneauet等人用完成的一个最大池化运算符，
采取一个双向LSTM作为语句编码器。
	
  数据集
   本文是基于NLI数据集进行的：The Natural Language Inference task
SNLI dataset是由570,000人工标注的英语句子对组成，每个句子对都有对应的标签。标签一共有三种，分别是entailment，contradiction和neutral。

  训练模型: 
    两个句子使用同一个编码器进行编码，而分类器则是使用通过两个句子嵌入构建的一对句子表征训练的。Conneau 等人采用了一个通过最大池化操作实现的双向LSTM 作为编码器。

  编码方式：
   目前，有多种多样的神经网络能将句子编码成固定大小的向量表示，并且也没有明确的研究支出哪一种编码方法最好。因此，作者选择了7种不同的architectures： 
   1. standard recurrent encoders with LSTM 
   2. standard recurrent encoders with GRU 
      上述两种是基础的recurrent encoder，在句子建模中通常将网络中的最后一个隐藏状态作为sentence representation； 
   3. conncatenation of last hidden states of forward and backward GRU 
      这种方法是将单向的网络变成了双向的网络，然后用将前向和后向的最后一个状态进行连接，得到句子向量； 
   4. Bi-directional LSTMs (BiLSTM) with mean pooling 
   5. Bi-directional LSTMs (BiLSTM) with max pooling 
   6. self-attentive network 
      这个网络在双向LSTM的基础上加入了attention机制
   7. hierarchical convolutional networks

  本文主要工作：
    Binary and multi-class classification 
    Entailment and semantic relatedness 
    STS14 - Semantic Textual Similarity 
    Paraphrase detection 
    Caption-Image retrieval

 最后结果：
    本文实现的有监督的句嵌入在该数据集的表现由于其他方式，另外采用BiLSTM with max pooling方式生成词嵌入的效果最好。


3、《What you can cram into a single vector: Probing sentence embeddings for linguistic properties》
    本文是Facebook AI Research发表于ACL 2018的工作，文章构建了一系列的句子级别的任务来检测不同模型获得的句子向量的质量。
任务包含表层的信息如预测句子长度或某个字是否出现在句子中，也包含句法信息如句法树的深度，语义信息如时态、主语个数、宾语个数等。
论文旨在比较不同模型获得的句子向量的质量。
    文章最后得到一些结果，由于自然语言输入的冗余，Bag-of-Vectors在捕获句子级属性方面出奇地擅长。文章展示了使用具有相似性能的
相同目标训练的不同编码器架构可以导致不同的嵌入，指出了句子嵌入的架构的重要性
