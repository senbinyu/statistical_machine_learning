马尔可夫模型（Markov Model）是一种统计模型，广泛应用在语音识别，词性自动标注，音字转换，概率文法等各个自然语言处理等应用领域。经过长期发展，尤其是在语音识别中的成功应用，使它成为一种通用的统计工具。

- 生成式模型和判别式模型？  
可观测到的是X，需要预测的是Y,其他变量为Z。生成式模型是先对联合概率密度建模，再通过计算边缘概率密度分布来得到对Y的推断。如下图   
![generate](https://user-images.githubusercontent.com/42667259/91327461-d5d06780-e7c5-11ea-82b6-14b6a17c8b34.png)  
而判别式模型是直接对条件概率密度P(Y,Z|X)建模，然后消掉无关变量Z，如下图  
![discriminant](https://user-images.githubusercontent.com/42667259/91327459-d537d100-e7c5-11ea-89b8-bff43a2ae35b.png)   
朴素贝叶斯，贝叶斯网络，隐马尔科夫都是先通过求联合概率密度，然后计算边缘分布得到的，所以是生成式模型；最大熵模型和条件随机场则是直接对条件概率密度分布建模，属于判别式。

- 马尔科夫过程  
假设在一个随机过程中，$t_n$时刻的状态$x_n$仅仅取决于其前一个状态$x_{n-1}$，则称其为马尔科夫过程，时间和状态都是离散取值的马尔科夫过程也称为马尔科夫链。这里所有的状态都是可见的，因此可以仅包含状态间的转移概率。   
![markov_chain](https://user-images.githubusercontent.com/42667259/91321115-93eff300-e7be-11ea-9d21-abb03c5d2a9e.png)

- 隐马尔科夫模型  
隐马尔科夫模型是对含有未知参数（隐参数，下图中的x）的马尔科夫链进行建模的生成模型。隐状态x对观测者是不可见的，仅有y可见，而观测状态的$y_i$仅取决于$x_i$。在隐马尔科夫模型中，参数包含了隐状态的转移概率，$x_i$到$y_i$的输出概率，$x_i$, $y_i$的取值空间及初始状态的概率分布。    
![hidden_markov](https://user-images.githubusercontent.com/42667259/91321114-93575c80-e7be-11ea-948d-78c131613b9f.png)

- 最大熵马尔科夫模型  
实际中，隐状态不仅和单个之前的状态有关，还和其他上下文的状态有关。最大熵马尔科夫模型去除了隐马尔科夫模型中观测状态$y_i$之间相互独立的假设，考虑了整个观测序列，由此也获得了强表达能力。同时，隐马尔科夫模型是一种对隐状态序列和观测状态序列的联合概率p(x,y)进行建模的生成式模型，而最大熵马尔科夫模型是直接对标注后的p(y|x)进行建模的判别式模型。    
![maxEnt_markov](https://user-images.githubusercontent.com/42667259/91324349-42496780-e7c2-11ea-9069-189c715193d6.png)   
其建模如下图：  
![maxEnt_markov_2](https://user-images.githubusercontent.com/42667259/91326050-2cd53d00-e7c4-11ea-8c6b-1587d961d4a8.png)   
右式中的$p(x_i|x_{i-1},y_{1...n})$会在局部进行归一化，枚举全部的$x_i$后进行求和后计算概率。其中Z是局部归一因子。正是由于这个局部归一化，可能会导致标注的时候产生偏置。    
![maxEnt_markov_3](https://user-images.githubusercontent.com/42667259/91326051-2d6dd380-e7c4-11ea-9538-2bc22aa56e49.png)
![maxEnt_markov_4](https://user-images.githubusercontent.com/42667259/91326052-2d6dd380-e7c4-11ea-9c77-a5486d6fb3f9.png)


- 条件随机场（conditional random field， CRF）  
CRF在最大熵马尔科夫模型基础上进行了全局归一化。整体形式酷似上述的最大熵马尔科夫模型，就是Z是在全局上进行的归一化，枚举了所有隐状态带来的全部可能，因此可以解决局部归一化带来的偏置问题。   
![crf](https://user-images.githubusercontent.com/42667259/91326475-ad943900-e7c4-11ea-9d74-f96c66b97681.png)


