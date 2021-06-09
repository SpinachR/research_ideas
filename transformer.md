https://www.bilibili.com/video/BV1Di4y1c7Zm/?spm_id_from=333.788.recommend_more_video.4

# 位置编码

对于每个feature的维度，偶数的位置使用sin，奇数的位置使用cos。 f(1,2,2,1) => (cos(1),sin(2),cos(2),sin(1))  
最后将词向量的embedding和对embedding进行位置编码后的向量相加得到的特征作为输入（一般一个词向量有512维）

这种位置编码sin和cos意味着该编码可以转化为两个相对位置的线性组合（也就是有了相对位置信息）f(pos1+pose2, 2i) = f(pos1, 2i)f(pos2, 2i+1) + f(pos1, 2i+1)f(pos2, 2i)

# 注意力机制

Attention(Q, K, V) = softmax(Q\*K^T/sqrt(d_k))\*V  

Q: query |  K: key  | V: value    

Q\*K^T: Q和K的相似程度，softmax将相似向量变成概率。最后和V点乘。

在只有单词向量的情况下，如何获取Q，K，V。用三个可以学习的矩阵W_Q, W_K, W_V分别和词向量相乘即可得到。最后encoder生成的是K, V矩阵，之后和decoder的Q矩阵做self-attention

**多头注意力机制就是 同时学习多个W_Q, W_K, W_V矩阵**

# 解码器

解码器的多头注意力机制中有个masked multi-head attention（意思是需要对当前单词和之后的单词做mask）。因为预测的时候，没有后面的单词信息，所以要在训练的时候把之后的单词遮住。
