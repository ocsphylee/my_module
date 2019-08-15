[TOC]
### Learning Time Series With Python

#### Chapter 1 差分方程和随机差分方程
##### 1.1 （确定性）差分方程

* 形如 $f(y_t,y_{t-1},...,y_{t-n})=x_t $ 的方程称为差分方程。

* 与微分方程一样，差分方程的结构也是由 **通解(complementary solution)** 和 **特解(particular sulution)** 组成的，即$y_t = y_p + y_c$.

* 通解
    - 一阶差分方程的通解：$y_t=ay_{t-1} \Rightarrow y_t = Aa^t$, A是任意常数。

***未完待续***

---

#### Chapter 2 ARMA 模型

##### 2.1 white noise, MA & ARMA process

* White noise: $ \{\varepsilon_t\} $ is called white noise iff it satisfies:
    1. Zero mean: $E(\varepsilon_t) = E(\varepsilon_{t-1}) = ... = 0 $
    2. Homogeneous: $E(\varepsilon_t^2) = E(\varepsilon_{t-1}^2) = ... = \sigma^2 $
    3. Non-autocorrelated: $E(\varepsilon_t \varepsilon_{t-s}) = E(\varepsilon_{t-j} \varepsilon_{t-j-s}) =0$
&nbsp;

* MA process (Moving Average):$ \{\varepsilon_t\} $ is white noise, $x_t = \sum_{i=0}^q \beta_i \varepsilon_{t-i} $ is called the q-order moving average process.
  
    Note that MA(q) satisfies Zero mean and homogeneous, but it is autocorrelated, hence it's not white noise.|
    --|
&nbsp;

* ARMA (Auto-Regressive Moving Average):ARMA is the combination of MA and linear difference equation, $y_t = a_0 + \sum_{i=1}^{p} a_i y_{t-i} + x_t$, where $x_t = \sum_{i=0}^q \beta_i \varepsilon_{t-i} $ is MA(q) process, denoted as ARMA(p,q).

    Note that the charateristic roots of the AR(difference equation) must lies in the unit circle, or else we call it ARIMA process, auto-regressive integrated moving average process.|
    --|
&nbsp;

##### 2.2 stationary 
  * Covariance Stationary
    1. Stationary often refers to covariance or weakly stationary: **a stochastic process $\{y_t\}$ is stational iff** 
    a) $E(y_t) = E(y_{t-s}) = \mu$
    b) $R(y_t,y_{t-s}) = cov(y_t,y_{t-s}) = R(s)$, which means that correlation only depends on the time difference but not time.
  
          Another stationary is called strongly or strictly stationary, which requires the same finite distribution over time. 
    2. if $\{y_t\}$ is stational, then the autocorrelation $\rho(y_t,y_{t-s}) = \rho_s = \frac{R(s)}{R(0)} $
&nbsp;

  * **[Theorem] The stationary condition for ARMA(p,q) is that: a) the characteristic roots of the AR model must lies in the unit circle; b) the MA process must be stational.**
  &nbsp;
    **[Proof]**
    * First, consider the MA($\infty$) process  $x_t = \sum_{i=0}^\infty \beta_i \varepsilon_{t-i} $
      a) mean: $E(x_t) = 0$, finite and independent of time.
      b) variance: $var(x_t) = E(x_t^2) = E(\beta_0^2 \varepsilon_t^2) + E(\beta_1^2 \varepsilon_{t-1}^2) + ... =  \sigma^2 \sum_{i=0}^{\infty} \beta_i$, independent of time, and finite when $\sum \beta_i$ is bounded.
      c) covariance: for any s>0, $E(x_t x_{t-s}) = E(\beta_0 \varepsilon_t + \beta_1 \varepsilon_{t-1} + ...)(\beta_0 \varepsilon_{t-s} + \beta_1 \varepsilon_{t-s-1} + ...) $. 
      Note that for any $s \ne 0 $, $E(\varepsilon_t \varepsilon_{t-s})=0$, hence, $E(x_t x_{t-s}) =\sigma^2 (\beta_s + \beta_1\beta_{s+1} + \beta_2\beta_{s+2} + ... ) $, independent of time and finite if $(\beta_s + \beta_1\beta_{s+1} + \beta_2\beta_{s+2} + ... ) $ is bounded.
      **hence, MA is stational if and only if a)  $\sum \beta_i$ is bounded; b) $(\beta_s + \beta_1\beta_{s+1} + \beta_2\beta_{s+2} + ... ) $ is bounded.**

    * for AR model,  $y_t = a_0 + \sum_{i=1}^{p} a_i y_{t-i} + \varepsilon_t$, the sufficient and necessary condition for stationary is that all the characteristic roots lies in the unit circle, hence, along with the MA stationary condition, we have that:
    $$
    y_t = (1-\sum_{i=1}^{p}a_i L^i)^{-1} (a_0 + \sum_{i=0}^q \beta_i \varepsilon_{t-i})
    $$
    is converge to a stable value.

    Note that the complementary solution can cause long-term drift from the stable state. Hence, another assumption is that the complementary solution is zero or the data generate a long time ago($t \rightarrow \infty$)|
    --|

&nbsp;

##### 2.3 ACF & PACF
  * ACF(Auto-Correlation Function): $\rho_s = corr(y_t,y_{t-s})$ is the correlation with $y_t$ and $y_{t-1}$, **for any stational series, $\rho_s$ should converges to 0 over time.** 

  * However, in $corr(y_t,y_{t-s})$, the effect of $y_{t-s}$ on $y_t$ is confusing, because it concludes not only the derect effect on $y_t$, but also the inderect efforts. In OLS estimation, we know that linear estimation is equal to calculate the correlation. Hence, **the ACF is equivalent to estimate** $y_t =\beta_0 + \beta_1 y_{t-1} + e$.

  * PACF(Partial Auto-Correlation Function) is to identify the derect effect of $y_{t-s}$. It is equivalent to estimate  $y_t = a_0 + \sum_{i=1}^{p} a_i y_{t-i} + \varepsilon_t$, and **the $a_s$ is the partial effect we needed.**

    Note that in AR(p), when s>p, PACF should be zero. hence, it can help us to identify which AR(p) model we should use in estimating.|
    --|
  &nbsp;

  * **[Application]**: for ARMA(p,q), **ACF would decrease from q period, and PACF would decrease from p period.** The convergence process could be oscillatorily or geometrically.  hence, for a unkown stational process, we can apply this characters to identify some models to be used.

  
  ---
  <center class="half">
    <h5>Experiment 1 Introduction of StatsModels</h5>
  </center>

  * [StatsModels](http://www.statsmodels.org/stable/index.html) is a powerful python module that contains a lots of statistic and econometrics models, including various regression models, statistic inference, time series analyse, multivariate statistics, ploting, etc.
  * we use the [tsa](http://www.statsmodels.org/stable/tsa.html) model to generate ARMA process and plot the ACF and PACF.
  <center class="half">
  <h5>Figure 1 ACF and PACF of ARMA(2,2) </h5>
    <img src="figures\exp1.png" width="800"/>
  </center>

  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import statsmodels.api as sm

  # generate ARMA model
  np.random.seed(12345)
  arparams = np.array([.75, -.25])
  maparams = np.array([.65, .35])
  ar = np.r_[1, -arparams] # add zero-lag and negate
  ma = np.r_[1, maparams] # add zero-lag
  y = sm.tsa.arma_generate_sample(ar, ma, 250)

  #plot ACF and PACF
  fig,axes = plt.subplots(1,2,figsize = (16,6))
  sm.graphics.tsa.plot_acf(y,ax = axes[0])
  sm.graphics.tsa.plot_pacf(y, ax = axes[1])
  ```
  ---
&nbsp;

##### 2.4 Autocorrelation of samples and test
  * Autocorrelation of samples (suppose the series is stational) can be calculated as:
  $$
  r_s = \frac{\sum\limits_{t=s+1}^{T}
               (y_t-\bar{y}) (y_{t-s}-\bar{y})
               }
            {\sum\limits_{t=1}^{T} (y_t-\bar{y})^2
               } 
  $$
  * Under the assumption of *stationary* and *Normality*, we can run some tests about the generation process for MA(s-1).
    - $H_0 : s = i \quad $ v.s. $\quad H_1 : s > i$ 
    1. t- test: since $var(r_s) = \begin{cases}
       T^{-1} \quad & s=1 \\ (1+2\sum\limits_{j=1}^{s-1}r_j^2)T^{-1} \quad & s>1
       \end{cases}$ , and further, when T(# of samples) is large enough, we have that $r_s \sim N(0,var(r_s)) $. Then we can run the t-test on $r_s$.
     
    2. **Ljung-Box test(Q-statistics)**:Box-Jenkins(1976) suggested a $\chi^2$ statistics ($Q = T \sum\limits_{k=1}^{s} r_k^2 \sim \chi^2(s)$) to test the MA(s-1) process. however, it does not perform well even in large samples. Ljung-Box provide a more consistant statistics and performs better even in small data. 
    $$Q = T(T+2) \sum\limits_{k=1}^{s} \frac{r_k^2}{T-k} \sim \chi^2(s) $$ 

      This statistic can also use to test the residuals of ARMA(p,q), to see if it's a white noise. But the degree of freedom needs to be adjusted: $s-p-q$ ($s-p-q-1$ if it contains a constant). |
      --|
&nbsp;


##### 2.5 Model Selection Criteria
  * According to ACF and PACF, we can guess the p and q. However, if AR(p) and MA(q) are possible, ARMA(p,q) is available as well. Hence, we need to choose a better model out of all our guesses.
  * In econometrics, we use **goodness-of-fit** to measure the fitness, and we also know that add a variable can reduce the residuals and enlarge the goodness of fit, yet easily cause overfitting. 
  * the idea of selection criteria is to include a penalty for each parameter. There are two frequently used criteria: **Akaike Information Criterion (AIC,赤池信息准则)** and the **Schwartz Bayesian Criterion (SBC, 施瓦茨准则)**.
  $$\begin{aligned}
   AIC &= T \ln(SSR) +2n \\
   SBC &= T \ln(SSR) + n \ln(T)
  \end{aligned}$$
  where SSR means **residual sum of squares** ,T means # of samples, and n means parameters estimated.

  * **Theoretically, AIC and SBC should be as small as possible (both can be negative), and the best situation would be both of them suggest the same model.** If not, AIC and SBC differs in following aspects:
    - SBC has larger penalty than AIC, hence, SBC would choose the simplier model.
    - SBC is suitable for large sample, and AIC performs better under small sample. 
    - if SBC and AIC suggest different model, we need to be cautious and run more tests.


  ---
  <center>
    <h5>Experiment 2 Estimation of ARMA model</h5>
  </center>

  *the dataset can be found in SIM_2.xls*

  * **AR(1) model**
    - We generate a sereis of $y_t = 0.7 y_{t-1} + \varepsilon_t $，and try to estimate the data.
    - From the ACF and PACF (the blue areas are the confidence interval of t-test), ACF decrease from the first period and PACF reach a peak also in first period.
    - Hence, our best guess would be AR(1) or ARMA(1,1).
    - From the results, we can see that model 1 is better than model 2, and the residuals are both not Zero significantly. We have a good estimation.

  <center>
  <h5>Figure 1 ACF and PACF of ARMA(1,0) </h5>
    <img src="figures\exp2.png" width="800">
  </center>
  <center>
  <img src="figures\exp2-1.png" width="400">
  </center>

  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import statsmodels.api as sm

  #import data and plot the data, ACF and PACF
  data = pd.read_excel('./data/sim_2.xls')
  y1 = np.array(data['Y1'])
  fig,axes = plt.subplots(1,3,figsize = (16,5))
  axes[0].plot(y1)
  axes[0].set_title("Raw Data")
  sm.graphics.tsa.plot_acf(y1,ax = axes[1])
  sm.graphics.tsa.plot_pacf(y1, ax = axes[2])
  plt.show()

  # fit the data
  ar1 = sm.tsa.ARMA(y1,(1,0)).fit(trend='nc')
  arma1 = sm.tsa.ARMA(y1,(1,1)).fit(trend='nc')
  
  # test the residuals
  r1 = ar1.resid
  r2 = arma1.resid
  r1_acf = pd.DataFrame(sm.tsa.acf(r1,qstat=True)).T
  r1_acf.columns=["ACF", "Q", "Prob(>Q)"]
  r2_acf = pd.DataFrame(sm.tsa.acf(r2,qstat=True)).T
  r2_acf.columns=["ACF", "Q", "Prob(>Q)"]
  ```
  

  * **ARMA(1,1) model**
    - This data is generater by ARMA(1,1) :  $y_t = -0.7 y_{t-1} + \varepsilon_t + -0.7 \varepsilon_{t-1}$
    - Observing the ACF and PACF, both of them decrease oscillatorily from period 1.
    - Hence, we can guess three models :

    $$ \begin{aligned}
    Model \ 1:& \ y_t = a_1 y_{t-1} + \varepsilon_t \\
    Model \ 2:& \ y_t = a_1 y_{t-1} + \varepsilon_t + \beta_1 \varepsilon_{t-1} \\
    Model \ 3:& \ y_t = a_1 y_{t-1}  + a_2 y_{t-2} + \varepsilon_t
    \end{aligned}
    $$

    - AIC and BIC both suggest model 3, and the Q-statistics for it's residuals are better than both model 1 and 2.
    <center>
    <h5>Figure 2 ACF and PACF of ARMA(1,1) </h5>
    <img src="figures\exp2-3.png" width="800">
    </center>
    <center>
    <img src="figures\exp2-2.png" width="500">
    </center>

  ---
&nbsp;

##### 2.6 Box-Jenkins Model Selection
  * **Identification Stage** 
    In this stage, researcher needs to observe the data by ploting the scatter diagram, ACF and PACF, to see if the data has a significant trend, and ensure the data is stational.
    - *parsimony pricipal*: use the least variables to fit the data as possible. And note that different process may generate the same process, AR(1) equals to MA($\infty$) for example, if possible, use the more parsimony form.  **To ensure parsimony, we ensure all the parameters' t-statistic larger than 2.**
    - *stationary and invetible*: stationary is the key assumption in time series analysis. And invertible ensure that the process is converge. 

  * **Estimation stage** 
    Model selection criteria is used in this stage.

  * **Diagnostic Checking**
    Ensure the residuals are white niose is the most important part of this stage. 
&nbsp;

##### 2.7 Prediction
  * After Seleting a model, we need to use it to predict the fulture values. 
  * AR(1):
    - Given $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$, denote the predicted value $E(y_{t+j}|y_t,...,y_{t-j-1},\varepsilon_t,...,\varepsilon_{t-j-1}) = E_t y_{t+j}$
    - It is easy to conduct that $E_t y_{t+j} = a_0 \sum\limits_{i=0}^{j-1} a_1^i + a_1^j y_t $, this is called the **Forecast Function**, and with $ |a_1|<1 $, the prediction is converge to $ \frac{a_0}{1-a_1} $.
    
    This result is profounding: for any stational ARMA, conditional expectation is converge to unconditional expectation.|
    --|

    - This prediction is unbiased yet not accurate. Because the prediction errors have strong impact on the prediction. 
    - Define $e_t(j)$ as the prediction error of t+j on t period. then : $$ \begin{aligned}
    e_t(1) & = y_{t+1} - E_ty_{t+1} = \varepsilon_{t+1} \\
    e_t(2) & = y_{t+2} - E_ty_{t+2} = a_1(y_{t+1} - E_ty_{t+1}) + \varepsilon_{t+2} = a_1\varepsilon_{t+1} + \varepsilon_{t+2} \\
    ...\\
    e_t(j) & =  \sum\limits_{i = 0}^{j-1} a_1^i \varepsilon_{t+j-i}
    \end{aligned}
    $$
    - We can see that the variation of $e_t(j)$ is $\sigma^2 \sum_{i=0}^{j-1} a_1^{2i}$, hence, the prediction is worse when in long term. When $j \rightarrow \infty$, $var[e_t(j)] = \frac{\sigma^2}{1-a_1^2}$.
  
  * ARMA(p,q): 
    - the prediction error for $y_t = a_0 + \sum_{i=1}^{p} a_i y_{t-i} + \sum_{i=0}^q \beta_i \varepsilon_{t-i} $ satiesfies :$$
    e_t(j) = \sum_{i=1}^{p} a_i e_t(j-i) + \sum_{i=0}^q \beta_i \varepsilon_{t-i}
    $$

  * Presiction Evaluation
    Since we can't know $y_{t+1}$ at t period, one way to aquired the predicted value is using only part of the data to estimate(train), and the rest of them are used to test. Given two model, there are several ways to compare their prediction power.
    - **F-Test**: assuming the predicted errora to be 1) $N(0,\sigma^2)$, 2) non-autocorrelation, 3) the two groups of errors are not correlated as well. Then, for each model, we get H one step prediction, the MSPE(minimum squared predicted error) of model 1 is $\sum_{i=1}^H e_{1i}^2$. Hence, under the condition above, we can construct a F-statistic (assuming MSPE1 is the larger one): $$ F = \frac{\sum_{i=1}^H e_{1i}^2}{\sum_{i=1}^H e_{2i}^2} \sim F(H,H)$$ However, the assumption above usually dont stands for the errors between groups are usually correlated. Hence, this method may invalid.

    - **Granger-Newbold Test**: to overcome the between group correlation problem, Granger and Newbold(1976) suggest that we can generate two new series: $x_i = e_{1i}+e_{2i}, z_i = e_{1i}-e_{2i}$, and under the assumption 1)and 2), $x_i$ and $z_i$ should be non-correlated if they have the same prediction power($\rho_{xz} = Ex_iz_i = E(e_{1i}^2 - e_{2i}^2)$). Further, denote $r_{xz}$ as the sample correlation, then $$\frac{r_{xz}}{\sqrt{\frac{1-r_{xz}^2}{H-1}}} \sim t(H-1)$$ Hence, statistically, if $r_{xz} \ne 0 $, model 1 has larger MSPE (worse) if positive.

    - **Diebold-Mariano Test**: sometimes, when we focus only on the prediction power not the explaination, the loss may differ in positive and negative error. Generally, we can define loss as $g(e_i)$, the difference of loss is $d_i = g(e_{1i})-g(e_{2i})$ and the mean loss is $$\bar{d} = \frac{1}{H} \sum\limits_{i=1}^{H} [g(e_{1i})-g(e_{2i})]$$ Define $\gamma_i$ as the i-order autocorrelation of $d_t$, then for the first q  $\gamma_i \ne 0$, we have: $$DM =\frac{\bar{d}}{\sqrt{\frac{\gamma_0+2\gamma_1+...+2\gamma_q}{H-1}}} \sim t(H-1)  $$
  
  &nbsp;

  ---
  <center>
    <h5>Experiment 3 Estimation of Interest Spreads</h5>
  </center>
  
  * This Experiment was to demostrate the application of Box-Jenkins method. Here, we use the speads between the interest rates of 5-year U.S. federal bonds and the 3-month U.S. T-Bills to measure the interest spreads ranged from 1960Q1 to 2012Q4.
  * From ACF and PACF, we can observe the following facts:
    - ACF and PACF converge to zero very quickly, hence, the series is basically stational.
    - In AR(p) process, the truncation of PACF after p period becomes zero, hence, this could be AR(6) or AR(7)
    - This Oscillation of PACF also indicates that positive MA coefficient.

  <center>
  <h5>Figure 3 Desription of Interest Spreads </h5>
  <img src="figures\exp3-1.png" width="800">
  </center>

  * Hence, we run the following test on the data.
    - In AR(7), $a_7$ is not significant, hence it is not sure for $y_{t-7}$ to hold. But Q(4), Q(8) and Q(12) both suggest that the residuals are not autocorrelated. 
    - In AR(6), the results are satisfying, yet compare to AR(7), AIC suggest AR(7) while BIC suggest AR(6).
    - Try AR(2) and ARMA(1,1), Q-statistic indicates that the residuals are autocorrelated.
    - Considering that $a_3 + a_4$ and $a_5 + a_6 $ are very close to zero in AR(7), we turn to estimate without them. But $a_7$ tend to be unsignificant and Q-statistic also suggest that this model is not good enough.
    - Finally, PACF suggests that it may contains MA process, and PACF decrease rapidly in period 2, we can estimate ARMA(2,1), and compare to AR(6) and AR(7), AIC and BIC both suggest ARMA(2,1).
    - Notice that there is significant lags in both ACF and PACF, we can estimate ARMA(2,(1,7)). But AIC suggests ARMA(2,1) and BIC suggests ARMA(2,(1,7)) in this case. Even though both models are reasonable, however, a cautious researcher would chooce ARMA(2,1) to aviod overfitting. 

  <center>
  <h5> Results of estimations </h5>
  <img src="figures\exp3.png" width="700">
  </center>

  * Prediction and Evaluation
    - Compare AR(7) and ARMA(2,(1,7)), since the observation of this data was 212, we can left the last 50 ones (range from 2000Q3 to 2012Q4) to test. We gather two one-step prediction errors, then we can run the Granger-Newbold test and or the Diebold-Mariano test. **Suppose the loss of error grows rapidly, we define the loss function as $e_i^4$**. 
     - From variance, we can see that ARMA(2,(1,7)) predicts more stable, yet, the both GN and DM test suggest that they are indifferent. And ACF and PACF of $\{d_i\}$ seem to suggest that $\{d_i\}$ is not autocorrelated(even though the 5-period has out range the condifence interval, we consider it as "mistake" to rule out).

  <center>
  <h5> Test for Prediciton </h5>
  <img src="figures\exp3-2.png" width="300">
  <img src="figures\exp3-3.png" width="700">
  </center>
  &nbsp;

  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import statsmodels.api as sm
  from scipy import stats
  import time
  # load data 
  quarter = pd.read_excel('./data/quarterly.xls')
  dta = pd.DataFrame(quarter['r5']-quarter['Tbill'],columns=['Gap'])
  dta.index =pd.date_range('1960','2013',freq="Q")

  # plot the data and ACF & PACF
  y = np.array(dta['Gap'])
  y_1 = dta.diff()
  fig,axes = plt.subplots(2,2,figsize = (16,9))
  axes[0][0].plot(dta)
  axes[0][0].set_title("Interest Spreads")
  axes[0][1].plot(y_1)
  axes[0][1].set_title("Interest Spreads (1-order difference)")
  sm.graphics.tsa.plot_acf(y,ax = axes[1][0],lags=15)
  sm.graphics.tsa.plot_pacf(y, ax = axes[1][1],lags=15)
  plt.show()

  # train models and present the results
  m1 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(7,0,0)).fit(disp=False)
  m2 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(6,0,0)).fit(disp=False)
  m3 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(2,0,0)).fit(disp=False)
  m4 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(1,0,1)).fit(disp=False)
  m5 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(2,0,1)).fit(disp=False)
  m6 = sm.tsa.statespace.SARIMAX(dta,trend = "c", order=(2,0,(1,0,0,0,0,0,1))).fit(disp=False)
  m7 = sm.tsa.statespace.SARIMAX(dta,trend = "c", order=((1,1,0,0,0,0,1),0,0)).fit(disp=False)

  def L_B_Q(model,n):
    r = model.resid
    r_acf = pd.DataFrame(sm.tsa.acf(r,qstat=True)).T
    r_acf.columns=["ACF", "Q", "Prob(>Q)"]
    print("Q({}): {}, p:{}".format(n,r_acf.iloc[n,1].round(4),r_acf.iloc[n,2].round(4)))
  
  print(m3.summary())
  print("*"*80)
  L_B_Q(m3,4)
  print("*"*80)
  L_B_Q(m3,8)
  print("*"*80)
  L_B_Q(m3,12)

  # GN and DM test (not find direct modules to do that, so I wrote a algrithm myself)
  def one_step_error(data,order,start = None, end = None):
      dates = pd.date_range(start = start, end=end, freq="Q")
      err_lit = []
      predict = []
      for i,date in enumerate(dates):
          date = date.strftime("%Y-%m-%d")
          model =sm.tsa.statespace.SARIMAX(data[:dates[i-1]],trend = "c", order= order).fit(disp=False)
          pred = model.predict(start=date, end=date)[0]
          predict.append(pred)
          err = data.loc[date][0] - pred
          err_lit.append(err)
      err_df = pd.DataFrame(err_lit,index = dates,columns=["err"])
      err_df["real"] = data[start:]
      err_df["predict"] = predict
      return err_df

  def sample_test(err_df1,err_df2):
      err1 = pd.DataFrame(err_df1["err"])
      err2 = pd.DataFrame(err_df2["err"])
      GN = err1 + err2
      GN['z'] = err1 - err2
      GN.columns = ["x","z"]
      corr = GN.corr().iloc[0,1]
      GN_T = corr / np.sqrt((1-corr**2)/49)
      diff = (err1 ** 4).sub((err2 ** 4), axis = 0)
      DM = diff.mean()[0]/ np.sqrt(diff.var()[0]/49)
      return GN_T, DM,diff

  start='2000-09-30'
  end='2012-12-31'
  err1 = one_step_error(dta,(7,0,0),start = start, end = end)
  err2 = one_step_error(dta,(2,0,(1,0,0,0,0,0,1)),start = start, end = end)
  GN,DM,diff= sample_test(err_df1 = err1,err_df2 = err2)
    # present the results
  print("-"*40)
  print("{:^10}{:^15}{:^15} ".format("","AR(7)","ARMA(2,(1,7))"))
  print("-"*40)
  print("{:^10}{:^15}{:^15} ".format("mean",err1['err'].mean().round(4),err2['err'].mean().round(4)))
  print("{:^10}{:^15}{:^15} ".format("variance",err1['err'].var().round(4),err2['err'].var().round(4)))
  print("{:^10}{:^30}".format("GN(p-value)",str(GN.round(4)) + "(" + str(stats.t.sf(GN,49).round(4)) + ")"))
  print("{:^10}{:^30}".format("DM(p_value)",str(DM.round(4)) + "(" + str(stats.t.sf(DM,49).round(4)) + ")"))
  print("-"*40)
    # plot the diff
  fig,axes = plt.subplots(2,2,figsize = (16,8))
  err1[["err","real"]].plot(ax = axes[0][0])
  err2[["err","real"]].plot(ax = axes[0][1])
  axes[0][0].set_title('Prediction of AR(7)')
  axes[0][1].set_title('Prediction of ARMA(2,(1,7))')
  sm.graphics.tsa.plot_acf(diff, ax = axes[1][0],lags=15)
  sm.graphics.tsa.plot_pacf(diff, ax = axes[1][1],lags=15)
  plt.show()

  ```
  ---
  &nbsp;

##### 2.8 Seasonal Model
  * **Many economic datas are seasonal**, effected by wheather, algriculture, tourism, or festivals; hence, seasonality can explain the large variance of the datas. 
  * Datas from government or other orgnization may be deseasonalized or seasonally adjusted, but we still need to concern about the seasonality of the datas. More importantly, most seasonally adjusted datas are conduted (implicitly) by two steps, seasonal moving (logarithmetics or difference) and estimate by Box-Jenkins method. Hence, if we use the adjusted data to estimate, we may gain a larger bias. ***The best way is not to use seasonal adjusted data and estimate the seasonal effect and ARMA coefficient at the same time.***
&nbsp;  
  * Considering the effect of seasonality, there are usually two kinds of seasonal model: **Pure Seasonal model** and **Multiplicative  Seasonal model**.

  * **Pure Seasonal Model** 
    - This model captures only the seasonal effect, and usually has forms like:  $$ \begin{cases}
    y_t & = a_s y_{t-s} + \varepsilon_t  & ARMA((s),0)\\
    y_t & = \varepsilon_t + \beta_s \varepsilon_{t-s} & ARMA(0,(s))
    \end{cases}
    $$

  * **Multiplicative Seasonal Model** (Box-Jenkins,1976)
    - This model consider not only the seasonal effect, but the interaction between the seasonal factors and the ARMA factors. 
    - For starters, we need to take difference or log to stationalize the data.
    - Second, by oberving the ACF and PACF, we conclude some possible models.
    - For example, in quarterly data, we consider the effect of 4-period factor: $$ \begin{cases}
      (1-a_1 L) y_t&= (1+\beta_1 L)(1 + \beta_4 L^4) \varepsilon_t  & (a)\\
      (1-a_1 L)(1-a_4 L^4) y_t&= (1+\beta_1 L)\varepsilon_t &(b)
      \end{cases}
      $$ ($a$) indicates the interaction between moving average term and (b) indicates the interaction between autoregressive term.
    - Generally, multiplicative seasonal model can be written as:$$ ARMA(p,d,q)(P,D,Q,s) \ : \  \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D y_t = A(t) + \theta_q (L) \tilde \theta_Q (L^s) \epsilon_t
    $$

      where:
      - $\phi_p (L)$ is the non-seasonal autoregressive lag polynomial, p is the order of AR
      - $\tilde \phi_P (L^s)$ is the seasonal autoregressive lag polynomial, P is the number of seasonal AR
      - $\Delta^d \Delta_s^D y_t$ is the time series, differenced d times, and seasonally differenced D times
      - $A(t)$ is the trend polynomial (including the intercept) 
      - $\theta_q (L)$ is the non-seasonal moving average lag polynomial, q is the order of MA
      - $ \tilde \theta_Q (L^s) \epsilon_t $  is the seasonal moving average lag polynomial, Q is the number of seasonal MA
      - s is the seasonal period

&nbsp;

---
  <center>
    <h5>Experiment 4 Estimation of Seasonal Data (M1 in US)</h5>
  </center>

  * This data is the M1 from 1960 to 2012 in US. For many reasons, such as the Chrismas, M1 tend to enlarge in the Q4 of every year. Also, this data also indicates a growing trend in M1. Hence, we take the log-difference to approcximate it's growth rate. And the long-term trend seems not so obevious as before.

  <center>
    <h6> M1 and Its Growth Rate </h6>
    <img src="figures\exp4-1.png" width="800">
  </center>

  * The main difference in ACF between seasonal process and non-seasonal process is that, in seasonal lag period s, 2s,..., **the ACF tend not to decrease to zero**.  After seasonal adjusted, the ACF decrease to Zero, and PACF didn't. This indicates that this data has seasonality and it probably contains in the MA process.

  <center>
  <h6> ACF and PACF of M1's Growth Rate </h6>
  <img src="figures\exp4-2.png" width="800">
  <h6> ACF and PACF of M1's Seasonal Adjusted Growth Rate </h6>
  <img src="figures\exp4-3.png" width="800">
  </center>

  * Hence, we can estimate the following models:
  $$\begin{aligned}
  model \ 1: \ & y_t = a_0 + a_1 y_{t-1} + \varepsilon_t + \beta_4 \varepsilon_{t-4}  & AR(1) \ with \ seasonal \ MA  \\
  model \ 2: \ & (1-a_1 L)(1-a_4 L^4) y_t = a_0 + \varepsilon_t  & ARMA(1,0,0)(1,0,0,4)  \\
  model \ 3: \ & y_t =a_0 +  (1+\beta_1 L)(1 + \beta_4 L^4) \varepsilon_t  & ARMA(0,0,1)(0,0,1,4) \\
  \end{aligned}
  $$

  * The Results are given in the following table.
    - Model 1 performs the best. Both AIC and BIC suggest model 1 is better and the Ljung-Box Q-statistic indicates that the residuals has no more explianable factors.
    - Model 2's residuals are autocorrelated, because SAR term didnot replicate the seasonal factor well enough. SAR suggest that the ACF declines from s period to s+1 period. However, ACF in the picture shows that there is a truncation after 4 period. 
    - Model 3 capture this seasonal factors well but it fails to replicate the autoregressive decline in short-term lags. Also, the Q-statistic indicates that the residuals are autocorrelated.
  <center>
  <h6> Results of Estimations </h6>
  <img src="figures\exp4-4.png" width="500">
  </center>

  ```python
  # load data and detrend
  data = pd.read_excel("./data/quarterly.xls")
  data.index  = pd.date_range('1960','2013',freq="Q")
  m = pd.DataFrame(data["M1NSA"])
  m["ln_M1"] = np.log(m)
  m["g_M1"] = m["ln_M1"].diff()
  m["g_M1_s"] = m['g_M1'].diff(4)
  fig,axes = plt.subplots(figsize = (13,6))
  axes.plot(m["M1NSA"])
  axes1 = axes.twinx()
  axes1.plot(m["g_M1"],'orange',alpha = 0.8)
  plt.show()

  fig,axes = plt.subplots(1,2,figsize= (16,5))
  sm.graphics.tsa.plot_acf(m['g_M1'][5:],ax = axes[0])
  sm.graphics.tsa.plot_pacf(m['g_M1'][5:], ax = axes[1])
  plt.show()

  fig,axes = plt.subplots(1,2,figsize= (16,5))
  sm.graphics.tsa.plot_acf(m['g_M1_s'][5:],ax = axes[0])
  sm.graphics.tsa.plot_pacf(m['g_M1_s'][5:], ax = axes[1])
  plt.show()

  # estimate and evaluation
  y = pd.DataFrame(m['g_M1_s'][5:])
  m1 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(1,0,(0,0,0,1))).fit(disp=False)
  m2 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(1,0,0),seasonal_order=(1,0,0,4)).fit(disp=False)
  m3 = sm.tsa.statespace.SARIMAX(y,trend = "c", order=(0,0,1),seasonal_order=(0,0,1,4)).fit(disp=False)

  ```
---
&nbsp;

##### 2.9 Combination Forecasting
 * Sometimes, we have several available models, and we don't just pick one of them to predict. One reason is that some model may contain the information that others fail to capture. 
 * One way to use all the available model is that we use the weighted average of the predictions:

    $$f_{ct} = w_1 f_{1t}  + w_2 f_{2t} + ... + w_n f_{nt}$$
  where $f_{it} $ is the one-step prediction series of model $i$. and $\sum w_i = 1$.

 * Combination forecasting has a great advantage. consider the predicted error:
   - if the models are unbiased, then the combination prediction are also unbiased, $Ey_{ct} = \sum w_i y_{it} =y_t $
   - then the combination prediction error: $ e_{ct} = \sum w_i e_{it} $. the variation would be $ var(e_{ct}) = \sum  w_i^2 Ee_{it}^2 + \sum\sum e_i e_j  Ee_{it}e_{jt}$. Suppose $ e_{it}, e_{jt} $ are non-correlated and with the same variation $\sigma^2$, then, it must be : $ var(e_{ct}) = \sigma^2 \sum w_i^2 \le \sigma^2 $. **The prediction error decreased.**

 * That leaves one problem to solve: **optimal weight**
    - simple averge
    - minimized $var(e_{ct})$ 
    - Bates and Granger(1969): $w_i^* = \frac{var(e_{it})^{-1}}{\sum var(e_{it})^{-1}}$
    - Granger and Ramanathan(1989): reg $y_t$ on $f_{it}$s, and constrain the coefficient $\sum w_i = 1$
    - use BIC as the weight: let BIS* be the best BIC, and $ a_i = \exp(\frac{BIC^* - BIC}{2}) $, then $w_i = \frac{a_i}{\sum a_i}$

  There is not a "Optimal Weight" for all models, and the researchers must decide the best way to predict by themselves based on the data and the model structures. |
  :--|
&nbsp;

##### 2.10 Parameter Stability
    
  In previous analyse, we assume that the data generating process dosen't change, which is not true in some cases. The data can be significantly changed by some exogenous or endogenous factors, which can lead to the change of coefficient.

  * **Structral Break Test**
    - Chow test can be use to test the structral break in data. Suppose the researcher suspect the data has change at $t_m$, he can use the unchanged ($t = t_1,...,t_m$) and changed ($t = t_{m+1},...,T$) data to estimate the same ARMA model.
    - The null assumption is that the data hasn't changed, hence, the coefficient of the two model should be the same.
    - Denote SSR1 and SSR2 to be the squared sum of residuals. hence, we can construct a F statistic: $$
    F = \frac{ \frac{SSR - SSR_1 - SSR_2}{n} }{ \frac{SSR_1 + SSR_2}{T-2n} } \sim F(n, T-2n) 
    $$ where, n is the number of coefficient($n = p+q$ or $n = p+q+1$), SSR is the original ARMA model's SSR.
    - An equivalent way to test the structral break is to introduce the dummy variable into the model.
  
  * **Endogenous Break**
    - When the researcher does not know the exact time of changing, we call it endogenous break. To indentify the changting period, we need to test all the potential time, and the larger F means closer to the changing time. 

&nbsp;

#### Chapter 3  Volatility Modeling

##### 3.1 ARCH and GARCH
  * We now condider the heteroscedasticity case, where the residuals' variance are not constants.
  * First, we need to address that conditional prediction is better than unconditional predition.
    - consider a AR(1): $y_t = a_0 + a_1 t_{t-1} + \varepsilon_t $
    - then the conditional expectation is $E_t y_{t+1} = a_0 + a_1 t_{t} $, and the variance of prediction error would be: $E_t(y_{t+1} - a_0 - a_1 y_t )^2 = E_t \varepsilon_{t+1}^2 = \sigma^2 $
    - if we use unconsitional prediction, the Expectation would be $ \frac{a_0}{1-a_1}$, hence, the variance would be $ \frac{\sigma^2}{1-a_1^2} $, which is larger than conditional prediction.
  
  * **ARCH model** 
    - We can use a simple model to estimate the volatility. Suppose we have estimate the model with a simple ARMA, and the residuals was $ \{ \hat \varepsilon_t \}$, hence, we can use it to model a AR(q) process: $$ \hat \varepsilon_{t}^2 = a_0 + a_1 \hat \varepsilon_{t-1}^2 + ... + a_q \hat \varepsilon_{t-q}^2 + v_t $$  where $v_t$ is a white noise. 
    - Hence, we can estimate the conditional variance  $Var(y_{t+1} | y_t)= E_t \varepsilon_{t+1}^2 $ by: $E_t \hat \varepsilon_{t+1}^2 = a_0 + a_1 \hat \varepsilon_{t}^2 + ... + a_q \hat \varepsilon_{t+1-q}^2$. We call it **Autoregressive Conditional Heteroskedastic** (ARCH) model (Engle,1982).
    - Engle propose a simplest ARCH model: $ \varepsilon_t = v_t \sqrt{a_0 + a_1 \varepsilon_{t-1}^2} $ , where $v_t$ is white noise and $ a_0 >0$, $0 \le a_1 \le 1$.
      This model has followting properties:
      - Zero mean and non-autocorrelated: $E \varepsilon_t = E(v_t) E (\sqrt{a_0 + a_1 \varepsilon_{t-1}^2}) = 0 $ and $E(\varepsilon_t \varepsilon_{t-1}) = 0$
      - Variance: $E \varepsilon_t^2 = E(v_t^2) E (a_0 + a_1 \varepsilon_{t-1}^2) = \frac{a_0}{1-a_1} $, the same with AR process.
      - Conditional variance: $ E_{t-1} \varepsilon_t = a_0 + a_1 \varepsilon_{t-1}^2 $ depends on known $\varepsilon_{t-1}^2$. In this case, the conditional variance follows AR(1) process but with constrains.
    

    Hence, the conditional variance is a ARCH process, with the larger $\varepsilon_{t-1}$, the larger variance is the $\varepsilon_t$, therefore, leads to heteroskedastic in $ \{ y_t \}$. As a result, ARCH process can show the stability and volatility in $ \{ y_t \}$.|
    :--|

    - Similary, $ \varepsilon_t = v_t \sqrt{a_0 + \sum\limits_{i=1}^{q} a_i \varepsilon_{t-i}^2} $ can be used to model higher order ARCH process, which means shocks from $ \varepsilon_{t-1} $ to $ \varepsilon_{t-q} $ have derect effect on $ \varepsilon_{t} $. Also, it is easy to proof that the conditional variance is AR(q) process.
    
  * **GARCH model**