**kaggle_taxi**  
# **纽约市出租车乘车时间(New York City Taxi Trip Duration)**

 - ## **任务描述**
 - ## **任务目标**
 - ## **数据统计与可视化:**
   - **数据统计**
   - **乘车时间统计分布**
   - **日期乘车数量统计**
   - **司机平均时间**
   - **乘客数量与时间的关系**
   - **乘车地点**
 - ## **特征工程**
   - 时间处理
   - 距离处理
   - 位置信息处理
 - ## **模型预测**
 
## 任务描述：
  本次数据主要是纽约市的乘车数据，分为训练集和预测集两个部分。训练集特征包括:  
- id - a unique identifier for each trip
- vendor_id - a code indicating the provider associated with the trip record
- pickup_datetime - date and time when the meter was engaged
- dropoff_datetime - date and time when the meter was disengaged
- passenger_count - the number of passengers in the vehicle (driver entered value)
- pickup_longitude - the longitude where the meter was engaged
- pickup_latitude - the latitude where the meter was engaged
- dropoff_longitude - the longitude where the meter was disengaged
- dropoff_latitude - the latitude where the meter was disengaged
- store\_and\_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
- trip_duration - duration of the trip in seconds  
然后测试集没有了'trip_duration'。
## 任务描述：
  预测测试集的乘车时间。   
  **评价标准(RMSLE)：**`$\epsilon = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$`   
`$\epsilon$` is the RMSLE value (score)  
n is the total number of observations in the (public/private) data set,
`$p_i$` is your prediction of trip duration, and
`$a_i$` is the actual trip duration for `$i$`.    
`$\log(x)$` is the natural logarithm of `$x$`
## 数据统计与可视化：
   - ### 数据统计：
这里主要是看最小值和最大值，因为这样可以看出是否存在有异常点，可以为后面消除异常点提供帮助。    
     ***训练集：***
          ![image](http://ww4.sinaimg.cn/large/0060lm7Tly1fjbdf9ibigj319s0j012t.jpg)
     ***测试集：***
          ![image](http://ww2.sinaimg.cn/large/0060lm7Tly1fjbdhhpliij319o0j8jz4.jpg)
    在训练集中，有乘客数竟然大于9，最小的是0，而乘车的时间也存在问题，有的时间只有1秒，长的竟然有`$3.52 * 10^6$`，这些明显不正常的值到时候肯定是要剔除的。
   - ### 乘车时间统计分布：
        ![image](http://ww3.sinaimg.cn/large/0060lm7Tly1fjbdw5k8mjj30e509jt8p.jpg)
        ![image](http://ww4.sinaimg.cn/large/0060lm7Tly1fjbdwn2wxfj30e509j0sq.jpg)   
        可以看出一开始的数据是服从幂律分布的，主要集中在0〜2000。然后我们就可以取对数，服从一个高斯分布。**后面训练的时候我们也用取对数的情况。因为很多时候我们都是假定数据是服从正态分布的**。
    - ### 日期乘车数量统计：
      这里统计的是训练集和测试集在时间在的车量变化。
      ![image](http://ww1.sinaimg.cn/large/0060lm7Tly1fjbe6q3s88j30dm095t9e.jpg)
      ***这里我们可以看出来在这个指标上面，集练集和测试集是趋势一致的。这可以假设是说明其实测试数据和训练数据是分布一致的，这也为后面抽cv提供了依据。***
    - ### 司机平均时间：
      这里统计的是不同司机所用的时间。   
      ![image](http://ww3.sinaimg.cn/large/0060lm7Tly1fjbf9gm5rpj30du09xjrd.jpg)   
   这里可以看到，司机不同，花费的时间还是有一定的差距的。一般来说，2比1花的时间更多。
    - ### 乘客数量与时间的关系：
      这里我们考查的是看乘客数量与所用时间的关系。**(特别说明一下，这里我已经把异常值舍弃了。）**
      ![image](http://ww1.sinaimg.cn/large/0060lm7Tly1fjbfczxk4tj30ry0gqwej.jpg)   
      其实在这里就可以发现,乘客的数量与用时其实并没有多大的关系。
    - ### **乘车地点：**
      这里我主要是想看一下训练集与测试集在地点上（经纬度）的关系。
      ![image](http://ww2.sinaimg.cn/large/0060lm7Tly1fjbfmfn26vj30eo0angn2.jpg)   
      这里可以进一步证明我们上次的观点就是训练集和测试集是同分布。为我们后面直接从训练集里面分数据提供了证据支撑。
## **特征工程**
  - 时间处理     
 这里我将时间这个特征分为月，天，小时，还有星期几这个几个特征，并且后面使用one-hot编码。这样做的原因是因为时间这个数字是一个周期性的数值，其本身是没有数值大小概念的，所以为了防止模型错误理解这个特征意义，我将其进行one-hot编码。当然，这也可能会维度的增加，但是因为它数据本身特征并不多，所以这个问题暂时不用考虑。
  - 距离处理  
  乘车时间很容易想到与距离有关系，因为按照经验来说，我们会很自然想到时间与距离是成正比的。但是我们现有的数据并没有直接给出距离的特征，这里我们可以利用起点经纬度和终点经纬度，进行计算。算距离有两个指标，一个是欧氏距离，一个是曼哈顿距离，还增加了一个方向。
 - 位置信息处理：   
 这里我用了kmeans聚类，将这些距离进行了一个聚类。我开始不是很明白为什么会这样做，但是后面我的理解是在同一个区域，那么它们的路况信息什么的都是比较接近的，二来也是增加了数据的特征，减少数据稀疏性（在经纬度上）。
## 模型预测：
这里选用的是xgboost。
xgboost的相关知识介绍可以参考：
>http://blog.csdn.net/sb19931201/article/details/52557382   
- 一般调参数的方法：
在过拟合情况：
  - 可以控制模型复杂度：
  调节max_depth,min_child_weight等
  - 可以增加训练集的随机性：
  调节subsample,colsample\_bytree，也可以调节eta（学习速率），但是也要增加num\_round
具体调参过程可以参考： http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
- 训练过程
![image](http://ww2.sinaimg.cn/large/0060lm7Tly1fjnqhl73u9j30n40vmjz1.jpg)
    