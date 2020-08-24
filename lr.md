此处将linear regression和logistic regression放在一块。

## linear regression，线性回归
1. 吴恩达课程：损失函数，最小均方误差， MSE： J(w) = 1/2m sum_{i=1}^{m} (h_w(x_i) - y_i)^2  
<!--![lr_loss](https://user-images.githubusercontent.com/42667259/91022105-fc8b7400-e5f4-11ea-83c1-287feb35bbd3.png)-->

2. 梯度下降，因为h_w(x) = wx + b, w即为图中的\theta, 如下的\alpha即为学习率  
![lr_grad](https://user-images.githubusercontent.com/42667259/91022102-fbf2dd80-e5f4-11ea-88dc-c925ee985898.jpg)

3. 岭回归，意指在对角线上加正则项，状似岭，故名岭回归，其实就是加上正则化的线性回归。h_w(x) = h_w(x) + \lambda w_i^2

4. 什么时候使用岭回归，即什么时候使用正则化合适？  
当样本数少的时候，或者样本重复程度高。

5. 什么时候使用Lasso回归？  
特征过多，稀疏线性关系，目的为了在一堆特征里面找出主要的特征

## logistic regression，逻辑斯蒂回归
1. 
