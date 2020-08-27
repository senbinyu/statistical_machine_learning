综合了常用于机器学习的各类损失函数，当然，这些基础函数很多还用于深度学习中。  
![机器学习](https://user-images.githubusercontent.com/42667259/91482447-5feffd00-e8a6-11ea-9097-7b0a497eb59a.png)

- MSE, mean square error,均方误差  
https://www.cnblogs.com/hansjorn/p/11458031.html   
常用在回归任务中，MSE的特点是光滑连续，可导，方便用于梯度下降。因为MSE是模型预测值 f(x) 与样本真实值 y 之间距离平方的平均值，故离得越远，误差越大，即受离群点的影响较大  
![MSE](https://user-images.githubusercontent.com/42667259/91484410-7481c480-e8a9-11ea-851d-a3e69408d395.png)

- MAE，mean absolute error,平均绝对值误差   
指的是模型预测值 f(x) 与样本真实值 y 之间距离的平均值，其在距离为0时，不可导。且梯度处处相等，即当误差较小时，梯度也和其他时候一样大，不利于学习。但相比于MSE，受离群点影响小。  
![MAE](https://user-images.githubusercontent.com/42667259/91484407-73e92e00-e8a9-11ea-986c-389d2c53f692.png)

- Huber loss  
为了结合上述两者的优势，在离群点处用delta控制斜率的MAE，而在delta范围内用MSE   
![huber](https://user-images.githubusercontent.com/42667259/91485171-b52e0d80-e8aa-11ea-93e4-bf40d9c7fb70.png)

- 0-1损失函数  
最简单的分类函数，分类正确即为
![0-1loss](https://user-images.githubusercontent.com/42667259/91485863-c592b800-e8ab-11ea-9235-93fc36e5a298.png)  
当然，上述即使预测值y^无限接近y，但不等于y时，都仍存在误差，这在实际情况中不适用。因此，引入一个t作为阈值，相当于适当放宽了条件，“软”soft  
![0-1loss_2](https://user-images.githubusercontent.com/42667259/91485865-c62b4e80-e8ab-11ea-86e1-68a120e33024.png)  

- hinge loss，合页损失  
![hinge_loss](https://user-images.githubusercontent.com/42667259/91485866-c62b4e80-e8ab-11ea-82d8-38ee25377606.png)

