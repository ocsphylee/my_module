
**Notes on machine learning**
**Author: Ocsphy**
[TOC]

* 这个notes默认读者已经熟练掌握高等数学、线性代数以及概率统计的内容，大部分推导过程会被省略掉。
* 这个notes的内容仅表示作者写作时的知识水平，如果有傻逼错误，请联系今天更加聪明的作者。

# 分类算法

## Logistic Regression
* Logistic Regression 事实上是广义线性回归，将普通线性回归（OLS）的结果，对应到二元分类里面

### 模型
* 考虑到函数性质（可微），我们选取sigmoid函数将OLS结果转化成｛0，1｝变量。sigmoid函数图像（[代码见附录](#附录1)）如下所示:
$$y=\frac{1}{1+e^{-x}}$$
<div align="center"><img src="ML_pic\sigmoid.png"></div>

* 从而，我们令  

$$样本方程： h_i=\frac{1}{1+e^{X_{i}^{T}\beta}} \Rightarrow \ln\frac{h_i}{1-h_i} = X_{i}^{T}\beta$$

$$总体方程： H=\frac{1}{1+e^{X\beta}}$$

其中：  

$$X = \begin{bmatrix}
      1      & x_{1}^{1} & x_{1}^{2} & \cdots & x_{1}^{d} \\
      \vdots & \vdots    & \vdots    & \ddots & \vdots    \\
      1      & x_{n}^{1} & x_{n}^{2} & \cdots & x_{n}^{d} 
      \end{bmatrix} \qquad
  X_i =  \begin{bmatrix}
      1      & x_{i}^{1} & x_{i}^{2} & \cdots & x_{i}^{d} 
      \end{bmatrix}^T$$  

$$
  Y =  \begin{bmatrix}
      y_{1}     & y_{2} & y_{3} & \cdots & y_{n}
      \end{bmatrix}^T \qquad
  \beta =  \begin{bmatrix}
      \beta_{1}     & \beta_{2} & \beta_{3} & \cdots & \beta_{n}
      \end{bmatrix}^T
$$


**如果我们把$h_i$看成是第$i$个样本分类为1的概率的话，这个模型会得到很好的理论解释。**


### 梯度下降法
* 损失函数（Loss Function）：  
    应用MLE，我们有似然函数：
    $$
    L(\beta) = \prod_{i=1}^{n}h_{i}^{y_i}(1-h_i)^{1-y_i}
    $$
    因此，有损失函数：
    $$
    l(\beta) = -\frac{1}{n}\sum_{i=1}^{n}[{y_i}\ln(h_i) + {(1-y_i)}\ln(1-h_i)]
    $$
    从而，我们的问题变成了：
    $$
    \hat{\beta} = arg\min_{\beta}l(\beta)
    $$

* 梯度下降法（Gradient Descend）:  
    1. 梯度（Gradient）  
       简单来说，梯度是函数导数的**反方向**。也就是说，梯度是一个**向量**。
    2. Idea  
       GD是求解 ***凸优化问题*** 的一种方法。比如函数$y=x^2$,给定任一点$x_0$,从这点出发，沿着梯度方向走，随着走的步数越来越多，其对应的函数值就越接近其最值。如图所示（[代码见附录](#附录1)）：  <div align="center"><img src="ML_pic\GD_pic.png"></div>
    3. 算法实现  
       ① 任意$x_0$，计算梯度$d_0 = -\frac{\partial f}{\partial x}|_{x=x_0}$  
       ② 选择步长（学习率）$\alpha$，更新公式为$x_1 = x_0 + \alpha d_0$  
       ③ 以此类推，可以通过判定阀值$\epsilon$，要求$\mid f(x_{k+1})-f(x_k) \mid \le \epsilon$；
          或者，选择设定最大迭代次数$k$，来停止迭代。
    4. 用GD求解Logistic Regression  
       ① diff $l(\beta)$ w.r.t. $\beta$ :
       $$
       \frac{\partial l}{\partial \beta} = -\frac{1}{n}\sum_{i=1}^{n}X_{i}^{T}(y_i-h_i)
       = -\frac{1}{n}X^T(y-h)
       $$  
       ② update:$\beta = \beta + \alpha \frac{\partial l}{\partial \beta}$
       

### 算法实现
* 导入数据集  
   #这个note的所有数据集都可以在我的[GitHub](https://github.com/ocsphylee/Training_dataSet.git)主页找到


```python
import numpy as np


def load_data(path):
    """ load data from txt file
    this function requires data structure to be [features, label]

    input:  path(str): the path of data file
    output: label(mat): an n*1 matrix of label
            features(mat): a n*(d+1) matrix of features
    """
    lines = []
    with open(path) as f:
        for line in f.readlines():
            lines.append(line.split())
    raw_data = np.array(lines, dtype=float)

    n = raw_data.shape[0]
    label = raw_data[:, -1].reshape((n, 1))
    features = raw_data.copy()
    features[:, -1] = 1

    return np.mat(label), np.mat(features)
```

* 定义sigmoid函数，并实现梯度下降


```python
def sig(x):
    return 1 / (1 + np.exp(-x))


def logit_gd(features, label, max_cycle, step):
    """train Logistic model with Gradient Descend
    
    input:  features(mat): a n*(d+1) matrix of features
            label(mat): an n*1 matrix of label
            max_cycle(int): maximum iteration times
            step(float): learning ratio
    output: beta(mat): a (d+1)*1 matrix of parameters
    """
    n, d = features.shape
    # initialize beta
    beta = np.ones((d,1))

    while max_cycle:
        max_cycle -= 1
        # calculate the gradient
        err = label - sig(features * beta)
        gd = features.T * err
        
        # track the approximate error (not necessary)
        if max_cycle % 100 == 0:
            error = np.sum(err) /n
            print("error: {} {}".format(max_cycle, error))
            
        # update beta
        beta += step * gd

    return beta
```

* 计算预测值和准确率


```python
def get_predict(features, beta):
    """predict label uses trained model
    input:  features(mat): a n*(d+1) matrix of features
            beta(mat): a (d+1)*1 matrix of parameters
    output: prediction(mat): a n * 1 predicted value of label
    """
    h = sig(features * beta)
    n = features.shape[0]
    predict_label = []
    for i in range(n):
        if h[i,0]>0.5:
            predict_label.append(1)
            continue
        predict_label.append(0)
    prediction = np.array(predict_label).reshape((n,1))

    return prediction

def get_accuracy(label, prediction):
    """calculate the accuracy of the model
    input:  label(mat): an n*1 matrix of label
            prediction(mat): a n * 1 predicted value of label
    output: acc(float): accuracy of the model
    """
    n = label.shape[0]
    result = 0
    for i in range(n):
        if label[i,0] == prediction[i,0]:
            result += 1
    acc = result / n
    return acc
```

* 运行


```python
if __name__ == "__main__":
    label, features = load_data("data/1.logit_data.txt")
    beta = logit_gd(features, label, 1000, 0.01)
    prediction = get_predict(features,beta)
    accuracy = get_accuracy(label,prediction)
    print(beta)
```

### 附录1
* sigmoid 图像绘制


```python
import numpy as np
import matplotlib.pyplot as plt


def sig(x):
    return 1/(1 + np.exp(-x))


x = np.linspace(-10, 10, 100)
y = np.ones((100,))
plt.plot(x, sig(x))
plt.plot(x, y, c='grey', linestyle='--')
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
ax.spines['left'].set_position(('data', 0))  # 移动左坐标轴到数据为0的位置
```

* GD图示


```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2
x = np.linspace(-10, 10, 100)

step = 10
x0 = 8
while step>0:
    step -= 1 
    x1 = x0 - 2 * x0*0.245
    plt.scatter(x0,f(x0),c="r",s=10)
    plt.plot([x0,x1],[f(x0),f(x1)],"r")
    x0 = x1
plt.plot(x,f(x))

ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
ax.spines['left'].set_position(('data', 0))  # 移动左坐标轴到数据为0的位置
```
---

## Factorization Machine
### 线性可分和非线性可分
* 数据线性可分是指能够利用一个（或多个）超平面来将数据区分，从而实现分类的目的；如果数据不能简单用超平面区分，那么就是非线性可分数据。
* Logistic Regression 是利用特征的线性组合来实现分类的线性模型，但是对于很多问题，并不一定存在线性可分，那么logistic Regression就存在局限性。对于非线性可分的模型，我们可以采取两种做法：  
  ① 核函数：即将非线性可分问题利用核函数转成线性可分，如将$y=\beta x$ 变成$y=\beta x^2$来估计。显然，对于复杂的数据，这个方法存在很多技术性要解决。  
  ② FM 模型：FM模型是对logistic Regression 的推广，是基于矩阵分解的一种机器学习方法。

### 模型
* 考虑含交叉项的普通线性回归($m,n$分别为样本数和特征数) $$
 \hat{y} =\beta_0 + \sum_{i=1}^{n} \beta_i x_i + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} \omega_{ij}x_ix_j
$$

  这种交叉项的引入在**数据稀松**的时候，即面临的数据没有明显的相互关系的时候，参数估计出现很大的问题。

* 辅助向量  
  我们令：$$
  V = \begin{bmatrix}
      v_{11}      & v_{12} & \cdots & v_{1k} \\
      \vdots & \vdots    & \ddots & \vdots    \\
      v_{n1}      & v_{n1} & \cdots & v_{nk} 
      \end{bmatrix} 
    = \begin{bmatrix} 
      v_{1}\\
      \vdots\\
      v_{n} 
    \end{bmatrix} \qquad
  \hat{W} = VV^T \qquad
  \hat{w}_{ij} = v_iv_j^T
    $$  
  因此，我们有FM模型：$$
 \hat{y} =\beta_0 + \sum_{i=1}^{n} \beta_i x_i + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} <v_i,v_j>x_ix_j
 $$
  * *从上面的形式可以看出FM也可以用来进行回归分析*
* 将FM模型应用于分类任务，同样，取sigmoid函数 $h=\frac{1}{1+e^{\hat{y}}}$,我们有logit loss损失函数：$$l(y,\hat{y}) = -\sum_{i=1}^{m}\ln h(y_i\hat{y}_i)
$$
### 随机梯度下降法
* 交叉处理  
  $$
  \begin{aligned}
  \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} <v_i,v_j>x_ix_j 
  &= \frac{1}{2} \sum_{i=1}^{n}\sum_{j=i+1}^{n} <v_i,v_j>x_ix_j 
    -\ \frac{1}{2} \sum_{i=1}^{n} <v_i,v_j>x_ix_i\\
  &= \frac{1}{2}( \sum_{i=1}^{n}\sum_{j=i+1}^{n} \sum_{f=1}^{k} v_{if}v_{jf}x_ix_j   - \sum_{i=1}^{n} \sum_{f=1}^{k} v_{if}^2x_i^2)\\
  &= \frac{1}{2} \sum_{f=1}^{k} [ (\sum_{i=1}^{n}v_{if}x_i ) (\sum_{j=1}^{n}v_{jf}x_j ) - \sum_{f=1}^{k} v_{if}^2x_i^2]\\
  &= \frac{1}{2}  \sum_{f=1}^{k} [ (\sum_{i=1}^{n}v_{if}x_i )^2 - \sum_{f=1}^{k} v_{if}^2x_i^2]
  \end{aligned}$$

* 随机梯度下降（Stochastic Graditent Descent）
  由于梯度下降法在每次迭代过程中都需要把所有的样本都运算一遍，从而导致随着样本的增加训练的时间变长。用随机梯度（即随机从样本中抽取出来一组小样本，然后计算平均梯度）替代梯度，可以有效减少训练时间，提高收敛速度。
  *对于样本量比较小的情况，我们可以直接选取单个样本来计算随机梯度*
  1. （单个样本）随机梯度：
    $$
    \frac{\partial l_i(y,\hat{y})}{\partial \theta} = [1-h(y_i\hat{y}_i]y_i \frac{\partial \hat{y}_i}{\partial \theta}
    $$

    $$
    \frac{\partial \hat{y}_{i}}{\partial \theta} =
    \begin{cases}
    1 \qquad & \theta = \beta_0 \\
    x_{ij} \qquad & \theta = \beta_j \\
    x_{ij} \sum_{s=1}^{n} v_{sf}x_{(is)} - v_{jf}x_{ij}^2 \qquad & \theta = v_{j,f}
    \end{cases}
    $$
    其中，$x_{ij}$表示第$i$个样本的第$j$个特征。

  2. SGD训练FM
    ① 初始化参数 $ \beta_i,\ i=0, \cdots ,n $ ，以及$V$
    ② 对于每一样本：更新 $\theta = \theta - \alpha \frac{\partial l_i(y,\hat{y})}{\partial \theta}  $ ，其中 $ \theta = \beta_i, v_{i,f},f=1, \cdots ,k$

### 算法实现
* 导入数据
```python
import numpy as np


def load_data(path):
    '''load data

    input:  path(str): file path
    output: features(mat): a m*n matrix of features
            label(mat): an m*1 matrix of label
    '''

    data = np.loadtxt(path)
    n = data.shape[1]-1  # number of coefficient
    features = np.mat(data[:,:n])
    label = np.mat(data[:, -1]*2-1).T
    return features, label
```

* 用SGD训练FM模型
```python
def sigmoid(x):
    return 1/(1+np.exp(-x))

def FM_train_SGD(features, label, k, max_iter, step):
    ''' train FM model with SGD

    input:  features(mat): a n*(d+1) matrix of features
            label(mat): an n*1 matrix of label
            k(int): number of auxilliary vector
            max_iter(int): maximum iteration times
            step(float): learning ratio
    output: beta_0(float): intercept
            beta(mat): an n*1 matrix of parameters
            v(mat): an n*k auxilliary matrix
    '''

    # initialize
    m, n = features.shape
    beta_0 = 0
    beta = np.zeros((n, 1))
    v = np.random.normal(0, 0.2, n*k).reshape((n, k))

    # train
    for it in range(max_iter):
        for x in range(m):

            # calculate the interaction part
            inter1 = np.array(features[x]*v)
            x_ary = np.array(features[x])
            inter2 = np.dot((x_ary**2), (v**2))
            interaction = np.sum(inter1**2 - inter2)/2

            # predition value
            p = beta_0 + features[x] * beta + interaction
            loss = sigmoid(label[x] * p[0, 0])-1

            # update beta0
            beta_0 -= step * loss * label[x]
            for i in range(n):
                if features[x, i] != 0:
                    # update beta
                    beta[i, 0] -= step * loss * label[x] * features[x, i]
                    # update v
                    for j in range(k):
                        partial_v = features[x, i] * inter1[0, j] -\
                            v[i, j] * (features[x, i]**2)
                        v[i, j] -= step * loss * label[x] * partial_v

    return beta_0, beta, v

```

* 开始训练并计时
```python
if __name__ == "__main__":
     #clocking start
    start =time.process_time()
    
    feature ,label = load_data("data/2.data_FM.txt")
    beta_0,beta,v = FM_train_SGD(feature,label,3,5000,0.01)
    
    end =time.process_time()
    print('Running time: %s Seconds'%(end-start))

```
* 同样的，也可以利用类似Logistic Regession中的方法，预测并计算准确率。


----

## 支持向量机（SVM）
* 前方高能提醒：前方理论众多，已经尽量简化，有些推导如果不想看可以直接跳过，看结论就好了，会用 ***黑斜体*** 提醒结论。

### 超平面和间隔
* 超平面  
在线性可分的数据中，我们的目的是寻找一个超平面来分隔数据，从而实现分类。但是一般情况下，都存在无穷个超平面满足我们的要求：
<div align="center"><img src="ML_pic\hyperplane.png"></div>

* 间隔  
  理论上，一个点如果离超平面越远，我们越能相信这个点的分类准确性。从而，直觉上，我们希望离**分隔超平面最近的点**与**超平面**的**距离**越大越好。  
  我们定义**间隔**为两个刚好经过数据集的**分割超平面**的距离。一般来说，间隔越大，这个分割超平面的容错率就越大，直觉上，它的**泛化误差**也就越小。
<div align="center"><img src="ML_pic\distance.png"></div>

* 函数间隔和几何间隔  
  1. 在给定分割超平面$\beta_0 + X \beta$的时候，$| \beta_0 + x_i^T \beta|$可以相对表示样本点$x_i$到超平面的距离；而当$( \beta_0 + x_i^T \beta)$与$y_i$同号的时候，表明该样本的分类是正确的，因此，我们可以用$y_i( \beta_0 + x_i^T \beta）$可以表示分类的正确性和可信度。因此，我们定义**函数间隔**为：

$$\hat{\gamma}_i = y_i( \beta_0 + x_i^T \beta）$$

$$\hat{\gamma} = \min_{i=1,...,m} \hat{\gamma}_i$$

  2. 几何间隔
    几何间隔比较好理解，就是样本点到分割超平面的距离：

$$\gamma_i = \frac{y_i( \beta_0 + x_i^T \beta）}{||\beta||} = \frac{\hat{\gamma}_i}{||\beta||}$$

$$\gamma = \frac{y( \beta_0 + x^T \beta）}{||\beta||} =  \frac{\hat{\gamma}}{||\beta||}$$

### 模型

* 目标函数  
  给定数据集$S = \{ (x_i,y_i) \}_{i=1}^{m} $，$x_i = (x_{i1},...,x_{in})^T$，其中，$y_i \in \{ -1,1 \} $。我们有分割超平面为：$\beta_0 + x^T \beta = 0$，其中，$\beta$为法向量。那么，我们的目标是：
  对于每一个样本：

  $$\max_{\beta_0,\beta} \frac{\hat{\gamma}}{||\beta||}$$ 

  $$s.t. \quad y_i( \beta_0 + x_i^T \beta） \ge \hat{\gamma} $$

  <div align="center"><img src="ML_pic\support_vector.png" width=40%></div>

  由于：
  $$(x_1^T - x_2^T)\beta = 2 = (\overrightarrow{x_1^Tx_2^T})\beta $$

  从而，将$\hat{\gamma}$看成是$\overrightarrow{x_1^Tx_2^T}$投影到$\beta$上的模，我们有：
  $$\hat{\gamma} = \frac{2}{||\beta||^2}  $$  

  代入原问题后，我们有：
  $$\begin{cases}
  \max\limits_{\beta_0,\beta} \frac{2}{||\beta||^2} \\
  s.t. \quad y_i( \beta_0 + x_i^T \beta） \ge 1
  \end{cases}
  \Longrightarrow
  \begin{cases}
  \min\limits_{\beta_0,\beta} \frac{||\beta||^2}{2} \\
  s.t. \quad y_i( \beta_0 + x_i^T \beta） \ge 1
  \end{cases}
  $$

* 支持向量的理解  
  在二分类问题中，**支持向量**指的是离分割超平面最近的样本点，即满足：
  $$ y_i( \beta_0 + x_i^T \beta） = 1 $$

  *也可以认为是那些使得约束条件binding的那些点。*
  由于在最后确定超平面的时候，仅有这些点（支持向量）在起作用，因此这个模型叫做**支持向量机**。

* 线性SVM  
  SVM存在一个潜在假设是要求数据线性可分，这个假设过于苛刻。我们可以放松到**提出某些特异点**后，数据线性可分。为了解决这个问题，我们引入**松弛变量**$\xi$，使得这些点在加上这个$\xi$之后，满足我们的约束条件：$$y_i( \beta_0 + x_i^T \beta） = 1-\xi_i$$
  对每一个$\xi_i$我们支付一个代价$C$，因此 ***新的问题变成了*** ：
  $$\min_{\beta_0,\beta} \frac{||\beta||^2}{2} + C\sum_{i=1}^{m}\xi_i$$

  $$s.t. \quad y_i( \beta_0 + x_i^T \beta） \ge 1-\xi_i \\
  \xi_i \ge 0 \\
  i = 1,2,...,m
  $$


### SVM的解  
* Lagrangian Function  

  $$L(\beta, \beta_0,\xi, \lambda, \mu) = \frac{||\beta||^2}{2} + C\sum_{i=1}^{m}\xi_i -
  \sum_{i=1}^{m} \lambda_i[y_i( \beta_0 + x_i^T \beta） - 1+ \xi_i] - 
  \sum_{i=1}^{m} \mu_i\xi_i
  $$

  那么我们的问题变成了：

  $$\min_{\beta,\beta_0, \xi} \max_{\lambda, \mu}L(\beta, \beta_0,\xi, \lambda, \mu) \quad
  \underrightarrow{Dual \ Problem} \quad
  \max_{\lambda, \mu} \min_{\beta,\beta_0, \xi} L(\beta, \beta_0,\xi, \lambda, \mu)
  $$

  因此，我们可以将这个问题分成两步求解：（1）先求$\tilde{L}(\lambda, \mu) = \min L $，（2）在求$\max \tilde{L}$，从而回代第一步得到$\beta,\beta_0$
$ \quad $
* $\min L$
  F.O.C on $\beta,\beta_0, \xi$:

  $$\begin{cases}
  \frac{\partial L}{\partial \beta} = \beta - \sum\limits_{i=1}^{m} \lambda_i y_i x_i = 0 \\
  \frac{\partial L}{\partial \beta_0} = - \sum\limits_{i=1}^{m} \lambda_i y_i= 0 \\
  \frac{\partial L}{\partial \xi_i} = C - \lambda_i - \mu_i = 0
  \end{cases}
  \Longrightarrow
  \begin{cases}
  \beta = \sum\limits_{i=1}^{m} \lambda_i y_i x_i \\
  \sum\limits_{i=1}^{m} \lambda_i y_i= 0 \\
  C - \lambda_i - \mu_i = 0
  \end{cases}
  $$

$ \quad $  
* $\max \tilde{L}$
  把上述结果代入到$L$中，我们有：

  $$\begin{aligned}
  \tilde{L} (\lambda) &= - \frac{||\beta||^2}{2} + \sum\limits_{i=1}^{m} \lambda_i \\
  &= - \frac{1}{2} \sum\limits_{i=1}^{m} \sum\limits_{j=1}^{m} \lambda_i \lambda_j y_i y_j x_i^T x_j +
  \sum\limits_{i=1}^{m} \lambda_i 
  \end{aligned}
  $$

  因此，我们的问题变成了：
  $$\begin{aligned}
  & \min_{\lambda} \frac{1}{2} \sum\limits_{i=1}^{m} \sum\limits_{j=1}^{m} \lambda_i \lambda_j y_i y_j x_i^T x_j - \sum\limits_{i=1}^{m} \lambda_i \\
   s.t. \quad   & \sum\limits_{i=1}^{m} \lambda_i y_i= 0 \\
  & 0 \le \lambda_i \le C
  \end{aligned}
  $$
  
  由Kuhn-Tucker条件可以解出$\lambda ^*$ ， ***从而有*** ：
  $$\begin{cases}
  \beta^* = \sum\limits_{i=1}^{m} \lambda_i^* y_i x_i \\
  \beta_0 = y_j - \lambda_j^* \sum\limits_{i=1}^{m}  y_i (x_i^T x_j) \ ,\ for\ any\ 0 \le \lambda_j^* \le C
  \end{cases}
  $$

### SMO解受约束优化问题
* Sequential Minimal Optimization （SMO）
  对于受约束的优化问题，我们可以采用序列最小优化算法（SMO）来实现，其基本思想是把一个大的优化问题转化成一系列的小问题，通过求解这些小问题来得到最后的优化结果。在SVM中，其主要思想可以分成两步：
  ① 每次选取两个变量$\lambda_i,\lambda_j$，控制其他变量不变后，我们有$\lambda_i = -\sum\limits_{k \ne i} \lambda_k y_k / y_i$,一旦$\lambda_j$被决定后，$\lambda_i$就可以被解出来了
  ② 不断重复以上过程，更新$\lambda_i,\lambda_j$，直到满足某些条件（收敛）。

  那么剩下的问题就是**如何选择变量**和**如何更新变量了**

* **更新公式**
  1. 假定我们选择了$\lambda_1,\lambda_2$，那么我们的优化问题变成了：
  $$\begin{aligned}
   \min_{\lambda}  \frac{1}{2} K_{11} &\lambda_1^2 +  \frac{1}{2} K_{22} \lambda_2^2 + y_1 y_2 K_{12} \lambda_1 \lambda_2 - (\lambda_1+\lambda_2) \\
   \qquad +& y_1 \lambda_1 \sum\limits_{i=3}^{m} \lambda_i y_iK_{i1} + y_2 \lambda_2 \sum\limits_{i=3}^{m} \lambda_i y_iK_{i2} + M_1 \\
   s.t. \quad   & \lambda_1 y_1 + \lambda_2 y_2 = - \sum\limits_{i=3}^{m} \lambda_i y_i= M_2 \\
  & 0 \le \lambda_1, \lambda_2 \le C
  \end{aligned}
  $$

  其中：$K_{ij} = x_i^T x_j$，为核函数。且如果我们定义$K_{ij} = f(x_i^T, x_j)$,我们便可以应用到非线性的模型当中去。
  2. 将$\lambda_1 y_1 + \lambda_2 y_2 = M_2$代入目标函数，我们有：
  $$\begin{aligned}
  W(\lambda_2) = & \frac{1}{2} K_{11} \ [(\lambda_2 y_2 - M_2) y_1]^2 + \frac{1}{2} K_{22} \lambda_2^2 + y_1 y_2 K_{12} [(\lambda_2 y_2 - M_2) y_1] \lambda_2 \\
  & - ([(\lambda_2 y_2 - M_2) y_1]+\lambda_2) + (\lambda_2 y_2 - M_2)\sum\limits_{i=3}^{m} \lambda_i y_iK_{i1} \\
  & + y_2 \lambda_2 \sum\limits_{i=3}^{m} \lambda_i y_iK_{i2} + M_1 
  \end{aligned}
  $$
 
   定义：

   $$g_j = \sum\limits_{i=1}^{m} \lambda_i y_iK_{ij} + \beta_0 \qquad
   E_j = g_j - y_j
   $$
   
   F.O.C:

   $$
   \frac{\partial W(\lambda_2)}{\partial \lambda_2} = (K_{11}+ K_{22} - 2K_{12} )\lambda_2 - 
   \begin{bmatrix}
    K_{11}M_2 - K_{12}M_2 - y_1 +y_2 + \\ 
    (g_1 - \sum\limits_{i=1}^{2} \lambda_i y_iK_{i1}- \beta_0) - (g_2 - \sum\limits_{i=1}^{2} \lambda_i y_iK_{i2}- \beta_0) 
    \end{bmatrix}
    y_2 = 0
   $$

   令$ \eta = K_{11}+ K_{22} - 2K_{12} $,在第$t+1$次迭代时：

   $$
   \eta \lambda_2^{t+1} = \eta \lambda_2^{t} + (E_1-E_2)y_2 \qquad \Rightarrow \qquad 
   \lambda_2^{t+1} =\lambda_2^{t} + \frac{(E_1-E_2)y_2 }{\eta}
   $$

   3. 同时，对于$\lambda_2^{t+1}$ ，我们有**约束条件**：$L \le \lambda_2^{t+1} \le H $，其中：

   $$
   L = \begin{cases}
   \max \{0,\lambda_2^{t} - \lambda_1^{t} \} \qquad \quad ,y_1 \ne y_2 \\
   \max \{0,\lambda_2^{t} + \lambda_1^{t} -C \} \quad ,y_1 = y_2
   \end{cases}
   \qquad
   H = \begin{cases}
   \min \{C,C + \lambda_2^{t} - \lambda_1^{t} \} \quad ,y_1 \ne y_2 \\
   \min \{C,\lambda_2^{t} + \lambda_1^{t}\} \quad \qquad  ,y_1 = y_2
   \end{cases}
   $$
  ***综上，我们有更新公式：***
  $$
   \lambda_2^{t+1} = \begin{cases}
   H \qquad \qquad \qquad ,\lambda_2^{t+1} \ge H \\
   \lambda_2^{t} + \frac{(E_1-E_2)y_2 }{\eta} \quad , L \le \lambda_2^{t+1} \le H \\ 
   L \qquad \qquad \qquad \ \  ,\lambda_2^{t+1} \le L 
   \end{cases}
   \qquad , \qquad
   \lambda_1^{t+1} = \lambda_1^{t} + y_1y_2(\lambda_2^{t} - \lambda_2^{t+1})
  $$

----

