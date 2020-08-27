综合了常用于机器学习的各类损失函数，当然，这些基础函数很多还用于深度学习中。下图大致给出了某些机器学习模型常用的损失函数，但具体还要视任务而定。  
![机器学习](https://user-images.githubusercontent.com/42667259/91482447-5feffd00-e8a6-11ea-9097-7b0a497eb59a.png)

#### 回归损失函数
- MSE, mean square error,均方误差  
https://www.cnblogs.com/hansjorn/p/11458031.html   
常用在回归任务中，MSE的特点是光滑连续，可导，方便用于梯度下降。因为MSE是模型预测值 f(x) 与样本真实值 y 之间距离平方的平均值，故离得越远，误差越大，即受离群点的影响较大  
![MSE](https://user-images.githubusercontent.com/42667259/91484410-7481c480-e8a9-11ea-851d-a3e69408d395.png)

- RMSE(root mean squared error),均方根误差  
是在上述MSE的基础上，取平方根。  
![RMSE](https://user-images.githubusercontent.com/42667259/91490109-34730f80-e8b2-11ea-9a97-726b2a25208f.png)

- RMSLE (Root Mean Squared Logarithmic Error)，均方根对数误差  
https://blog.csdn.net/qq_24671941/article/details/95868747  
RMSLE 惩罚欠预测大于过预测，适用于某些需要欠预测损失更大的场景，如预测共享单车需求。  
如果预测的值的范围很大，RMSE 会被一些大的值主导。如果有一个很大的值偏离，RMSE会使得误差很大。而先取log，再求其RMSE，即使用RMSLE可以减轻这个问题。   
![RMSLE](https://user-images.githubusercontent.com/42667259/91490110-350ba600-e8b2-11ea-8f90-7a50cd3d828b.png)

- MAE，mean absolute error,平均绝对值误差   
指的是模型预测值 f(x) 与样本真实值 y 之间距离的平均值，其在距离为0时，不可导。且梯度处处相等，即当误差较小时，梯度也和其他时候一样大，不利于学习。但相比于MSE，受离群点影响小。  
![MAE](https://user-images.githubusercontent.com/42667259/91484407-73e92e00-e8a9-11ea-986c-389d2c53f692.png)

- Huber loss  
为了结合上述两者的优势，在离群点处用delta控制斜率的MAE，而在delta范围内用MSE   
![huber](https://user-images.githubusercontent.com/42667259/91485171-b52e0d80-e8aa-11ea-93e4-bf40d9c7fb70.png)

- smoothL1Loss  
特殊的，smoothL1Loss是huber loss中的delta=1时的情况。这个损失函数也用在了faster RCNN中，用于定位框的回归损失。    
![smoothL1Loss](https://user-images.githubusercontent.com/42667259/91488847-36d46a00-e8b0-11ea-8197-dfbf551309d5.png)

#### 分类损失函数
- 0-1损失函数  
https://zhuanlan.zhihu.com/p/47202768  
最简单的分类函数，分类正确即为
![0-1loss](https://user-images.githubusercontent.com/42667259/91485863-c592b800-e8ab-11ea-9235-93fc36e5a298.png)  
当然，上述即使预测值y^无限接近y，但不等于y时，都仍存在误差，这在实际情况中不适用。因此，引入一个t作为阈值，相当于适当放宽了条件，“软”soft  
![0-1loss_2](https://user-images.githubusercontent.com/42667259/91485865-c62b4e80-e8ab-11ea-86e1-68a120e33024.png)  

- Cross entropy, CE Loss,交叉熵损失   
https://zhuanlan.zhihu.com/p/58883095  
如下图所示,y是真实标签，a是预测标签，一般可通过sigmoid，softmax得到，x是样本，n是样本数目  
![ce_loss](https://user-images.githubusercontent.com/42667259/91491995-2c689f00-e8b5-11ea-8294-e6c122da3476.png)  

- hinge loss，合页损失  
用在SVM中的损失函数。当y落在满足条件的一侧时，不管是多少，损失函数为0；而在不满足条件的一侧，会逐渐增大。这就保证了普通向量损失为0，不参与超平面的最终决定，这才是SVM的核心所在，保证了SVM解的稀疏性  
![hinge_loss](https://user-images.githubusercontent.com/42667259/91485866-c62b4e80-e8ab-11ea-82d8-38ee25377606.png)  
![hinge_loss_2](https://user-images.githubusercontent.com/42667259/91487849-b103ef00-e8ae-11ea-9697-04e2103f0ce1.jpg)

- Exponential Loss  
Exponential Loss为指数误差，常用于boosting算法中，如adaboost中。当预测分类和之前分类一致时，损失为1/e，不相同时，为e，由此可见，对离群点较为敏感。  
![exponential_loss](https://user-images.githubusercontent.com/42667259/91489024-726f3400-e8b0-11ea-9e09-123086aeaaf4.png)   

