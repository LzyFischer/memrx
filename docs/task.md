## 8.24 
1. memory is not only based on utility, but also how good it looks. 



## 8.25
1. 现在问题是什么？
    1. 我们应该用什么做实验。我要确保每一个基础的效果都不会太差，有代表性。-》然后我们要继续设计实验
    2. 我们要设计什么实验？
        1. 问题：
            1. processing并不是consistently有效的：win,tie,lose ratio 三个图，每个图是一个dimension
            2. 不同种类的query适用什么样的processing：热力图，每个图纵轴是query，横轴是processing
            3. 选择processing在不同的阶段有不同的效果：retrieval和answer phase
               1. retrieval phase我们单独算recall
               2. answer phase我们单独算retrieval没问题的accuracy
            4. 我们发现prompt LLM去选择，就已经可以在某种程度上增多win的，减少lose的，提升总的performance。
         2. 
        2. 实验目的：引入challenge，引入我们的方法。
