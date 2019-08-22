
<center>
<h2>Learning Time Series With Python</h2>
<h6>Author: Ocsphy</h6>
<h6>2019</h6>
</center>

[TOC]

### Chapter 1 差分方程和随机差分方程
##### 1.1 （确定性）差分方程

* 形如 $f(y_t,y_{t-1},...,y_{t-n})=x_t $ 的方程称为差分方程。

* 与微分方程一样，差分方程的结构也是由 **通解(complementary solution)** 和 **特解(particular sulution)** 组成的，即$y_t = y_p + y_c$.

* 通解
    - 一阶差分方程的通解：$y_t=ay_{t-1} \Rightarrow y_t = Aa^t$, A是任意常数。

***未完待续***

---

#### Chapter 2 ARMA Model

#### 2.1 white noise, MA & ARMA process

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

#### 2.2 stationary 
  * Covariance Stationary
    1. Stationary often refers to covariance or weakly stationary: **a stochastic process $\{y_t\}$ is staionary iff** 
    a) $E(y_t) = E(y_{t-s}) = \mu$
    b) $R(y_t,y_{t-s}) = cov(y_t,y_{t-s}) = R(s)$, which means that correlation only depends on the time difference but not time.
  
          Another stationary is called strongly or strictly stationary, which requires the same finite distribution over time. 
    2. if $\{y_t\}$ is staionary, then the autocorrelation $\rho(y_t,y_{t-s}) = \rho_s = \frac{R(s)}{R(0)} $
&nbsp;

  * **[Theorem] The stationary condition for ARMA(p,q) is that: a) the characteristic roots of the AR model must lies in the unit circle; b) the MA process must be staionary.**
  &nbsp;
    **[Proof]**
    * First, consider the MA($\infty$) process  $x_t = \sum_{i=0}^\infty \beta_i \varepsilon_{t-i} $
      a) mean: $E(x_t) = 0$, finite and independent of time.
      b) variance: $var(x_t) = E(x_t^2) = E(\beta_0^2 \varepsilon_t^2) + E(\beta_1^2 \varepsilon_{t-1}^2) + ... =  \sigma^2 \sum_{i=0}^{\infty} \beta_i$, independent of time, and finite when $\sum \beta_i$ is bounded.
      c) covariance: for any s>0, $E(x_t x_{t-s}) = E(\beta_0 \varepsilon_t + \beta_1 \varepsilon_{t-1} + ...)(\beta_0 \varepsilon_{t-s} + \beta_1 \varepsilon_{t-s-1} + ...) $. 
      Note that for any $s \ne 0 $, $E(\varepsilon_t \varepsilon_{t-s})=0$, hence, $E(x_t x_{t-s}) =\sigma^2 (\beta_s + \beta_1\beta_{s+1} + \beta_2\beta_{s+2} + ... ) $, independent of time and finite if $(\beta_s + \beta_1\beta_{s+1} + \beta_2\beta_{s+2} + ... ) $ is bounded.
      **hence, MA is staionary if and only if a)  $\sum \beta_i$ is bounded; b) $(\beta_s + \beta_1\beta_{s+1} + \beta_2\beta_{s+2} + ... ) $ is bounded.**

    * for AR model,  $y_t = a_0 + \sum_{i=1}^{p} a_i y_{t-i} + \varepsilon_t$, the sufficient and necessary condition for stationary is that all the characteristic roots lies in the unit circle, hence, along with the MA stationary condition, we have that:
    $$
    y_t = (1-\sum_{i=1}^{p}a_i L^i)^{-1} (a_0 + \sum_{i=0}^q \beta_i \varepsilon_{t-i})
    $$
    is converge to a stable value.

    Note that the complementary solution can cause long-term drift from the stable state. Hence, another assumption is that the complementary solution is zero or the data generate a long time ago($t \rightarrow \infty$)|
    --|

&nbsp;

#### 2.3 ACF & PACF
  * ACF(Auto-Correlation Function): $\rho_s = corr(y_t,y_{t-s})$ is the correlation with $y_t$ and $y_{t-1}$, **for any staionary series, $\rho_s$ should converges to 0 over time.** 

  * However, in $corr(y_t,y_{t-s})$, the effect of $y_{t-s}$ on $y_t$ is confusing, because it concludes not only the derect effect on $y_t$, but also the inderect efforts. In OLS estimation, we know that linear estimation is equal to calculate the correlation. Hence, **the ACF is equivalent to estimate** $y_t =\beta_0 + \beta_1 y_{t-1} + e$.

  * PACF(Partial Auto-Correlation Function) is to identify the derect effect of $y_{t-s}$. It is equivalent to estimate  $y_t = a_0 + \sum_{i=1}^{p} a_i y_{t-i} + \varepsilon_t$, and **the $a_s$ is the partial effect we needed.**

    Note that in AR(p), when s>p, PACF should be zero. hence, it can help us to identify which AR(p) model we should use in estimating.|
    --|
  &nbsp;

  * **[Application]**: for ARMA(p,q), **ACF would decrease from q period, and PACF would decrease from p period.** The convergence process could be oscillatorily or geometrically.  hence, for a unkown staionary process, we can apply this characters to identify some models to be used.

  
  ---
##### Experiment 1 Introduction of StatsModels

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

#### 2.4 Autocorrelation of samples and test
  * Autocorrelation of samples (suppose the series is staionary) can be calculated as:
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


#### 2.5 Model Selection Criteria
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
##### Experiment 2 Estimation of ARMA model

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

#### 2.6 Box-Jenkins Model Selection
  * **Identification Stage** 
    In this stage, researcher needs to observe the data by ploting the scatter diagram, ACF and PACF, to see if the data has a significant trend, and ensure the data is staionary.
    - *parsimony pricipal*: use the least variables to fit the data as possible. And note that different process may generate the same process, AR(1) equals to MA($\infty$) for example, if possible, use the more parsimony form.  **To ensure parsimony, we ensure all the parameters' t-statistic larger than 2.**
    - *stationary and invetible*: stationary is the key assumption in time series analysis. And invertible ensure that the process is converge. 

  * **Estimation stage** 
    Model selection criteria is used in this stage.

  * **Diagnostic Checking**
    Ensure the residuals are white niose is the most important part of this stage. 
&nbsp;

#### 2.7 Prediction
  * After Seleting a model, we need to use it to predict the fulture values. 
  * AR(1):
    - Given $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$, denote the predicted value $E(y_{t+j}|y_t,...,y_{t-j-1},\varepsilon_t,...,\varepsilon_{t-j-1}) = E_t y_{t+j}$
    - It is easy to conduct that $E_t y_{t+j} = a_0 \sum\limits_{i=0}^{j-1} a_1^i + a_1^j y_t $, this is called the **Forecast Function**, and with $ |a_1|<1 $, the prediction is converge to $ \frac{a_0}{1-a_1} $.
    
    This result is profounding: for any staionary ARMA, conditional expectation is converge to unconditional expectation.|
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
##### Experiment 3 Estimation of Interest Spreads
  
  * This Experiment was to demostrate the application of Box-Jenkins method. Here, we use the speads between the interest rates of 5-year U.S. federal bonds and the 3-month U.S. T-Bills to measure the interest spreads ranged from 1960Q1 to 2012Q4.
  * From ACF and PACF, we can observe the following facts:
    - ACF and PACF converge to zero very quickly, hence, the series is basically staionary.
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

#### 2.8 Seasonal Model
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
    - For starters, we need to take difference or log to staionarize the data.
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
##### Experiment 4 Estimation of Seasonal Data (M1 in US)

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

#### 2.9 Combination Forecasting
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

#### 2.10 Parameter Stability
    
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

### Chapter 3  Volatility Modeling

#### 3.1 ARCH and GARCH
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
    - Bollerslev(1986) introduce a model which allows the conditional variance turning into ARMA process: $ \varepsilon_t = v_t \sqrt{h_t} $, where:$$ h_t = a_0 + \sum\limits_{i=1}^{q} a_i \varepsilon_{t-i}^2 + \sum\limits_{i=1}^{p} \beta_i h_{t-i} $$
    - It is easy to proof that the unconditional expectation and conditional expectation is zero, and conditional expectation is $ E_{t-1} \varepsilon_t^2 = h_t $, which is a ARMA(p,q) process.
    - Hence, GARCH(p,q) process is an extension of ARCH process, when p=0, they are equivalent.
  
  * **Test for ARCH or GARCH**
    - we can identify the conditional heteroskedastic in following steps:
      1. fit $ \{ y_t \}$ with optimal ARMA or regression model, the residuals are $ \{ e_t \}$, and calculate the variance: $$ \hat \sigma^2 = \frac{ \sum\limits_{t=1}^{T} e_t^2 }{T} $$
      2. Calculate the ACF of sqared residuals: $$ \rho_i = \frac{ \sum\limits_{t=i+1}^{T} (e_t^2- \hat \sigma^2) (e_{t-i}^2- \hat \sigma^2) }{\sum\limits_{t=1}^{T} (e_t^2- \hat \sigma^2)^2} $$
      3. In large sample, the standard variance of $ \rho_i$ is $ 1/ \sqrt{T}$. Hence, Q-statistic can be used to test the ACF: $$ Q = T(T+2) \sum\limits_{i=1}^{n}\frac{\rho_i^2}{T-i} \sim \chi^2(n)  $$ where the null hypothesis is that there is no ARCH or GARCH process in $ \{ e_t \}$. In pratice, we can use $ n = T/4 $
    
    - Mcleod and Li(1983) propose a formal Lagrangian test for ARCH process:
      1. fit $ \{ y_t \}$ with optimal ARMA or regression model, the squared residuals are $ \{ e_t^2 \}$
      2. estimate model: $$ e_t^2 = a_0 + a_1 e_{t-1}^2 + ... + a_q e_{t-q}^2 $$
    
      if null hypothesis( no ARCH process) hold, then $R^2$ should be relatively small. Hence, under null hypothesis, $TR^2 \sim \chi^2(q)$. Under small sample, F-test should be better than $\chi^2$-test. 

---
##### Experiment 5 Volatility of Real GDP in US (imperfect)

  * This experiment use real GDP in US from 1947 to 2012 to indentify the volatility in some period, such as 1984-2002, or 2008 financial crisis. We use the GDP growth rate $y_t = \log \frac{RGDP}{RGDP_{-1}}$ to analyse.
  * First, the basic ploting:

  <center>
  <h5> GDP growth rate , ACF and PACF </h5>
  <img src="figures\exp5-1.png" width="700">
  </center>

  * It is easy to conduct that the best model is AR(1): $$\begin{aligned}
  y_t = & 0.008&  + \quad &0.37 * y_{t-1}&+\varepsilon_t \\
        & (8.656)&  &(6.467)&
  \end{aligned} 
  $$

  * Apply Mcleod-Li test for this quarterly data:$$\begin{aligned}
  e_t^2 = & 8.315 * 10^{-5}&  + \quad &0.116 * e_{t-1}^2& + \quad &0.129 * e_{t-2}^2& - \quad &0.029 * e_{t-3}^2& + \quad &0.124 * e_{t-4}^2& \\
        & (5.613)&  &(1.897)&  &(2.092)&  &(0.628)& &(2.034)&
  \end{aligned} 
  $$ where the F-test shows that under significant of 0.05, null hypothesis(the parameters are joinly zero) does not hold. That means the data shows volatility.

  * We can also measure a ARCH model: $$\begin{aligned}
  y_t = & 0.005&  + \quad &0.39 * y_{t-1}&+\varepsilon_t \\
        & (5.439)&  &(5.523)&
  \end{aligned} 
  $$

    $$\begin{aligned}
    h_t = & 6.38 * 10^{-5}&  + \quad &0.25 * e_{t-1}^2& \\
          & (7.157)&  &(1.834)&
    \end{aligned} 
    $$
  * To test the sufficient of this model, we can use standardized residuals $ \frac{e_t}{\sqrt h_t} $, which is an estimation of $v_t$ (residuals of GARCH model). Hence, only when $ \frac{e_t}{\sqrt h_t} $ is non-autocorrelated and constant variance, the model is sufficient.
  <center>
  <h5> Standardized residual </h5>
  <img src="figures\exp5-2.png" width="400">
  </center>

  * Specially, we can estimate a GRACH model with exogenous variables: (*to be done*)
    $$\begin{aligned}
    h_t = a_0 + a_1 e_{t-1}^2  + \gamma X \\       
    \end{aligned} 
    $$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch.univariate import ARCH, GARCH, ARX

# load data and plot
data = pd.read_excel("./data/REAL.xls",index_col="DATE")
y = pd.DataFrame(np.log(data["RGDP"]).diff()[1:])

fig,axes = plt.subplots(1,3,figsize = (16,3))
axes[0].plot(y)
sm.graphics.tsa.plot_acf(y,ax = axes[1])
sm.graphics.tsa.plot_pacf(y, ax = axes[2])
plt.show()

# estimate AR(1) and test for residuals 
m1 = sm.tsa.ARMA(y, order=(1,0)).fit(trend="c")
res_2 = m1.resid[1:]**2
m2 = sm.tsa.ARIMA(res_2,order=(4,0,0)).fit(trend="c")
# F-test
A= np.identity(5)[1:,:]
m2.f_test(A,invcov = 300) # questioned!!!
#ARCH(1)
m3 = arch_model(y = y["RGDP"],mean ="ARX",vol ="ARCH",lags = 1).fit(disp="off")
m3.summary()
m3.plot()

# conditional valotility(h_t) and standardized residual
h_t = m4.conditional_volatility
resid = m4.resid
std_resid = resid/h_t
```
---

&nbsp;
#### 3.2 Estimation with MLE
  * Condider a GARCH model:
  $$\begin{aligned}
  y_t & = \alpha_0 + \sum\limits_{i=1}^{P} \alpha_i y_{t-i} + \sum\limits_{i=i}^{Q} \beta_i \varepsilon_{t-i} + \varepsilon_t & (a)\\
  \varepsilon_t &= v_t \sqrt{h_t} & (b)\\
  h_t & = a_0 + \sum\limits_{i=1}^{q} a_i \varepsilon_{t-i}^2 + \sum\limits_{i=i}^{p} b_i h_{t-i} & (c)
  \end{aligned}
  $$ 
  where, $v_t \sim N(0,1)$ 

  * Hence, the likelihood function is 
  $$ \begin{aligned}
   L &= \prod_{t=1}^{T} (\frac{1}{ \sqrt{2 \pi h_t} }) \exp({- \frac{\varepsilon_i^2}{2h_i} })\\
  \ln L &= - \frac{T}{2} \ln(2\pi) - \frac{1}{2} \sum\limits_{t=1}^{T} \ln h_i -  \frac{1}{2} \sum\limits_{t=1}^{T} \frac{\varepsilon_i^2}{h_i}
  \end{aligned}
  $$ replace $h_t$ with (c) and $\varepsilon_t$ with estimated residuals $ e_t $, we can apply some numberical method to estimate the parameters. 
&nbsp;

#### 3.3 ARCH-M Model
  * In asset market, ARCH-M model is very popularm, which usually used to model the excess earning of an assets. 
  $$\begin{aligned} r_t &= \mu_t + \varepsilon_t \\ \mu_t &= \beta + \delta h_t , \delta >0 \\ h_t &=a_0 +  \sum\limits_{i=1}^{q} a_i \varepsilon_{t-i}^2 \end{aligned}$$ where, $r_t$ is the excess earning and  $\mu_t$ measures the risk premium.
&nbsp;

#### 3.4 Other Conditional Variance Model
  * IGARCH (integrated GARCH):  $h_t = a_0 + a_1 \varepsilon_{t-1}^2 + \beta_1 h_{t-1}$, where, $a_1 + \beta_1 = 1$
    In this case, the one-step prediction of conditional variance would be $ E_t h_{t+1} = a_0 + h_t $
  
  * Asymmetric models: TARCH and EGARCH
    - Leverage effect: Earnings of today have strong nagetive relation with the volatility in the fulture. The reason behind it is that a nagetive stock shocks results in low the values related to the debts, hence, risks increase as a result of higher the asset-liability ratio. **However, a positive shock usually causes less volatility than a nagetive shock.**
    - TARCH(Threshold ARCH): $h_t = a_0 + \sum\limits_{i=1}^{q} a_i \varepsilon_{t-i}^2  +\sum\limits_{i=1}^{o} \gamma_i \varepsilon_{t-i}^2 I[ \varepsilon_{t-i} < 0 ] + \sum\limits_{i=i}^{p} \beta_i h_{t-i} $ , if $\gamma_i$ significantly not equal to zero, then we have threshold effct.

    - EGARCH(Exponential GARCH): $\ln h_{t} = a_ 0 + \sum_{i=1}^{p}\alpha_{i} \frac{\varepsilon_{t-i}}{h_{t-i}^{0.5}} +\sum_{i=1}^{o}\gamma_{i} |\frac{\varepsilon_{t-i}}{h_{t-i}^{0.5}}|  +\sum_{i=1}^{q}\beta_{i}\ln h_{t-i} $
```python
GARCH(p=1, o=1, q=1) # TARCH(1,1,1)
EGARCH(p=1, o=1, q=1) # EGARCH(1,1,1)
```

  * Test for Leverage effect
    - Define $s_t = \frac{e_t}{h_t^{0.5}}$ be the standardized residuals. To test leverage effect, we can estimate the following equation: $$ s_t^2 = a_0 + a_1 s_{t-1} + a_2 s_{t-2} + ... $$ if joinly F-statistic shows that $a_1,...$ is zero, then we can rule out leverage effect.

    - further, we can introduce a dummy $d_t = 0 $ if $e_t \ge 0$, else, $d_t=1$. And then estimate: $$ s_t^2 = a_0 + a_1 d_{t-1} + a_2 d_{t-1}s_{t-1} + a_3(1-d_{t-1}) s_{t-1} $$

  * Non-Normal Distributed Error
    Sometime, the distributions of error are fat-tailed, which we wouldn't use normal distribution to estimate. luckly, we can use t,F or even $\chi^2$ to run MLE.

---
##### Experiment 6 Estimation of NYSE100 index

  * We use NYSE100 index to calculate the return rate
  * First, by ACF and PACF, we choose the **AR(2) model**: $$\begin{aligned}
  r_t = & 0.0032&  - \quad &0.0947 * r_{t-1}& - \quad &0.0576 * r_{t-2}& \\
        & (0.163)^*&  &(-5.42)& &(-3.29)&
  \end{aligned} 
  $$ Note that the intercept is insignificant, however, since t-statistic changes depending on the conditional variance, it is better we keep the interception. Once we find the suitable ARCH, we can estimate an new model without interception.

  * **Test for GARCH**
    $$\begin{aligned}
  e_t^2 = & 1.6254 &  + \quad &0.0469 * e_{t-1}^2& + \quad &0.3085 * e_{t-2}^2& - \quad &0.0046 * e_{t-3}^2& + \quad &0.104 * e_{t-4}^2& + \quad &0.2332 * e_{t-5}^2&\\
        & (6.365)&  &(2.756)&  &(18.212)&  &(0.259)^*& &(6.141)& &(13.709)&
  \end{aligned} 
  $$ GARCH effect exist.

  * From the distribution of the data, we can see that the return rate has fat tail, which means that normal distribution may not stand very well. 
  <center>
  <h5> Distribution of return rate </h5>
  <img src="figures\exp6-1.png" width="600">
  </center> 

  * Then, we can use different distribution assumption to fit the ARCH model. 
  With **normal distribution**:
  $$\begin{aligned}
  r_t = & 0.044&  - \quad &0.058 * r_{t-1}& - \quad &0.038 * r_{t-2}& (AIC:9295.35 \quad BIC:9331.90 ) \\
        & (2.939)&  &(-3.236)& &(-2.014)& \\
  h_t = & 0.014&  + \quad &0.084 * e_{t-1}^2&  +  \quad &0.906 * h_{t-1}& \\
  & (2.936)&  &(8.213)& &(85.935)&
    \end{aligned} 
  $$ With **t distribution**:
  $$\begin{aligned}
  r_t = & 0.061&  - \quad &0.062 * r_{t-1}& - \quad &0.045 * r_{t-2}& (AIC: 9162.72 \quad BIC:9205.36) \\
        & (4.326)&  &(-3.946)& &(-2.494)& \\
  h_t = & 0.009&  + \quad &0.089 * e_{t-1}^2&  +  \quad &0.909 * h_{t-1}&\\
  & (2.974)&  &(8.719)& &(94.882)&
    \end{aligned} $$ and the estimated degree of freedom is 6.14 (0.707).Hence, **it is better use t distribution than norm distribution to fit the GARCH model**.

  * However, if we look at the coefficient of conditional variance, we have that $a_1 + \beta_1 \approx 1$. Hence, we can try the **IGARCH model with t-distribution**.$$\begin{aligned}
  r_t  = & 0.061&  - \quad &0.062 * r_{t-1}& - \quad &0.045 * r_{t-2}& (AIC: 9187.16 \quad BIC:9217.62) \\
        & (3.868)&  &(-3.723)& &(-2.336)& \\
  h_t =  &0.08* e_{t-1}^2&  +  \quad &0.92 * h_{t-1}&\\
   &  &  &(119.913)& 
    \end{aligned} $$ (*I can not add the intercept now*)

  * **Reference Testing** for validation of GARCH(1,1): constructing $ s_t = \frac{e_t}{h_t^0.5} $
    1. **Remaining serial correlation**: the actocorrelation is relatively small and Q-statistic suggests that the standardized remaining residuals are not correlated.(Stationary)
    <center>
    <img src="figures\exp6-2.png" width="400">
    </center> 
    
    2. **Remaining GARCH effect**: our goal is to test the remaining residuals have no conditional heteroskedastic. We can test the following equation: $$ s_t^2 = a_0 + a_1s_{t-1}^2 + ... + a_n s_{t-n}^2 $$ for all n, we find that n = 2,3,... are all insigificant. $$ \begin{aligned}
    s_t^2 = &0.987& - & 0.0468  * s_{t-1}^2  \\
    &(30.302)&  &(-2.679)&
    \end{aligned}  $$ t-statistic is 0.007. Hence, we can conclude that **there is no GARCH effect**. We can alway estimate higher order GARCH model to eliminate the GARCH effect, however, it usually causes overfitting problem.

    3. **Leverage effect**: If there is no leverage effect, $s_t^2$ and $s_t$ should not be correlated. consider the following equation: $$ \begin{aligned}
    s_t^2 = &0.9694& - & 0.0962  * s_{t-1}& - & 0.1788  * s_{t-2} & \\
    &(28.419)&  &(-2.806)& &(-5.216)&
    \end{aligned}  $$  $s_{t-1}$ and $s_{t-2}$ are highly significant and F-test for $ a_1 = a_2 = 0 $ shows that the parameters are joinly unequal to zero significantly (F=17.57, p-value=0.000).**The nagetive parameters indicate that nagetive shock can cause higher conditional variation.**
      To extend this idea, we can run Engle-Ng (Sign Bias Test): $$ \begin{aligned}
    s_t^2 = &0.6204& + & 0.2922  * d_{t-1}& +  & 0.2820  *d_{t-2} & + & 0.1405 *d_{t-3}& \\
    &(8.902)&  &(4.3)& &(4.148)& &(2.066)&
    \end{aligned}  $$
    
      $$ \begin{aligned}
    s_t^2 = &1.0071& + & 0.2608  * d_{t-1}& +  & 0.1794  *d_{t-1} s_{t-1} & - & 0.2394  *(1-d_{t-1} )s_{t-1} & \\
    &(13.075)&  &(2.515)& &(2.779)& &(-2.797)&
    \end{aligned}  $$ **This result indicates that there is leverage effect, and  the effect depends on not only sign but also the shock itself.**

  * **Asymmetric Model**: Since leverage effect is proved to exist, we need to estimate the asymmetric model(TARCH or EGARCH):
    **TARCH**: $$\begin{aligned}
  r_t = & 0.0367&  - \quad &0.0615 * r_{t-1}& - \quad &0.0413 * r_{t-2}& &(AIC:9058.37 \quad BIC:9107.10 ) \\
        & (2.737)&  &(-3.836)& &(-1.782)& \\
  h_t = & 0.01&  + \quad &0.000 * e_{t-1}^2&  +  \quad &0.1374 * d_{t-1} e_{t-1}^2& + \quad &0.9211 * h_{t-1}&\\
  & (3.683)&  &(0.000)& &(7.736)& &(77.652)&
    \end{aligned}  $$ It is obvious that $e_{t-1}^2$ can not be identified (because $a_1$ needs to be positive)
    **EGARCH**:$$\begin{aligned}
  r_t = & 0.0385&  - \quad &0.0604 * r_{t-1}& - \quad &0.0317 * r_{t-2}& & (AIC:9055.64 \quad BIC:9104.38 ) \\
        & (2.737)&  &(-3.836)& &(-1.782)& \\
  \ln h_t = & 0.0004&  + \quad &0.1080 * |\frac{e_{t-1}^2}{h_{t-1}^{0.5}}| &  -  \quad &0.1292 * \frac{e_{t-1}^2}{h_{t-1}^{0.5}} & + \quad &0.9861 * \ln h_{t-1}&\\
  & (-0.165)^*&  &(7.845)& &(-9.192)& &(336.425)&
    \end{aligned}  $$ where eatimated degree of freedom is 6.88. EGARCH indicates that **good news(positive shock) decrease the volatility and nagetive shock has massive impact on the volatility( $0.108+0.129 > 0.18-0.129$)**
  
  * Finally, we need to justify the EGARCH model. 
    1. **Remaining Residuals**: We run AR(1) on the remaining residuals, and find that even though there is a significant effect, it is very small so that we can ignore it. $$ \begin{aligned}
    s_t^2 = &0.9943 & - &0.0537 * s_{t-1}^2 &  \\
            &(31.562)& &(-3.074)&
    \end{aligned}   $$

    2. **Q-Q graph**: We can test normality $\{ s_t \}$ using Q-Q graph, and it shows that rather than normal distribution, it appears to be t-distribution.
    <center>
    <img src="figures\exp6-3.png" width="800">
    </center> 
  
  * We can draw the predicted conditional variance and compare with the original data. Even though the one-step predition(Forcast) is not accurate, but the conditional variance can reflect the valotility pretty well. It is justify that the volatility from 2008 to 2009 raise significantly due to the financial crisis.


```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch.univariate import ARCH, GARCH, ARX
from arch import arch_model

date = pd.date_range(start="2000-1-4",end="2012-7-16",freq="B")
data = pd.read_excel("./data/NYSE.xlsx")
rate = data.loc[1:,"RATE"]
rate.index = date
# Observe the data
fig,axes = plt.subplots(1,3,figsize = (20,5))
axes[0].plot(rate)
sm.graphics.tsa.plot_acf(rate,ax = axes[1])
sm.graphics.tsa.plot_pacf(rate, ax = axes[2])
plt.show()
# distribution
from scipy.stats import t,norm
x = np.arange(-10,10,0.01)
fig,axes = plt.subplots(figsize = (16,9))
axes.hist(rate,bins=200,facecolor = "gray",edgecolor= "black",density=1,label = "rate",alpha=0.5)
axes.plot(x,norm.pdf(x,0,1),label="norm")
axes.plot(x,t.pdf(x,3),label = "t(3)")
plt.legend(fontsize=20)
# axes.set_ylim(0,0.6)
axes.set_xlim(-5,5)

# simple AR(2) model and test for valotality
m1 = sm.tsa.ARMA(rate,order=(2,0)).fit()
m1.summary()
res_2 = m1.resid **2
test = sm.tsa.ARMA(res_2,order=(5,0)).fit()
test.summary()

# GARCH models
garch = arch_model(y = rate,mean ="ARX",vol ="GARCH",lags = 2,dist='normal').fit(disp="off")
garch_t = arch_model(y = rate,mean ="ARX",vol ="GARCH",lags = 2,dist='t').fit(disp="off")
from arch.univariate import EWMAVariance,StudentsT,FIGARCH
igarch_t = ARX(y=rate,lags = 2,distribution=StudentsT())
igarch_t.volatility=EWMAVariance(None)
igarch_t.fit(disp=False).summary()

# test for remaining serial correlation
s_t = garch_t.resid[2:] / garch_t.conditional_volatility[2:]
q_stat = pd.DataFrame(sm.tsa.acf(s_t,nlags=15,qstat=True, fft=False),index=["acf",'Q','p-value'])
print(q_stat.T)

# test for remaining GARCH effects
s_t_2 = s_t ** 2
q_stat = pd.DataFrame(sm.tsa.acf(s_t_2,nlags=15,qstat=True, fft=False),index=["acf",'Q','p-value'])
print(q_stat.T)
test_2 = sm.tsa.ARMA(s_t_2,order=(1,0)).fit(disp=False)
print(test_2.summary())

# test for Leverage effect
# since there is no direct method, we construct OLS
X = np.column_stack((s_t[1:-1], s_t[:-2]))
X = sm.add_constant(X)
Y = s_t_2[2:]
Lev = sm.OLS(Y, X).fit()
print(Lev.summary())
R = np.identity(3)[1:,:]
print(Lev.f_test(R))
# sign test
d = []
for e in s_t:
    if e>0:
        d.append(0)
    else:
        d.append(1)
d_t = np.array(d)
X = np.column_stack((d_t[2:-1], d_t[1:-2],d_t[:-3]))
X = sm.add_constant(X)
Y = s_t_2[3:]
Lev = sm.OLS(Y, X).fit()
print(Lev.summary())
R = np.identity(4)[1:,:]
print(Lev.f_test(R))

X = np.column_stack((d_t[:-1],d_t[:-1]*s_t[:-1] ,(1-d_t[:-1])*s_t[:-1] ))
X = sm.add_constant(X)
Y = s_t_2[1:]
Lev = sm.OLS(Y, X).fit()
print(Lev.summary())
R = np.identity(4)[1:,:]
print(Lev.f_test(R))

# Asymmetric model
tgarch_t = arch_model(y = rate,mean ="ARX",vol ="GARCH",lags = 2,dist='t',p=1,q=1,o=1).fit(disp="off")
egarch_t = arch_model(y = rate,mean ="ARX",vol ="EGARCH",lags = 2,dist='t',p=1,q=1,o=1).fit(disp="off")
# test the remaining residuals of EGARCH model
s_t = egarch_t.resid[2:] / egarch_t.conditional_volatility[2:]
s_t_2 = s_t **2
test = sm.tsa.ARMA(s_t_2,order=(1,0)).fit(disp=False)
print(test.summary())
# Q-Q graph and the predition
forecasts = egarch_t.forecast(horizon=1, start="2007-1-1", method='simulation') # one step forcast

fig,axes = plt.subplots(1,2,figsize = (20,5))
sm.graphics.qqplot(s_t,line ='q',ax = axes[0]) # QQ graph
axes[1].plot(rate,label = "Return Rate",alpha= 0.8)
axes[1].plot(egarch_t.conditional_volatility,label = "Conditional Volatility",alpha = 0.8)
axes[1].plot(forecasts.mean,label = "Forcast",c = "b",alpha =0.5)
axes[1].legend(fontsize = 15)
axes[0].set_title("Normality Test", fontsize = 20)
axes[1].set_title("Predicted Volatility", fontsize =  20)
```
---
&nbsp;
#### 3.5 Multivariate GARCH

*To be continue*

&nbsp;

### Chapter 4 Trend
#### 4.1 Deterministic and Stochastic Trend

 * Modeling the trend
   For any linear stochastic difference equation, it's general solultion usually consists of three part: $ y_t = trend + staionary \ part + noise $. In the ARMA model, we know how to model the staionary part, and how to model the variation of errors in volatility modeling. **The key feature of a trend is that it has permanent effect on the series**.
* **Deterministic trend and trend-stationary**
  - consider a simple model: $ y_t = t_{t-1} + a_0 \Rightarrow \Delta y_t = a_0 $
  - the solution is $ y_t = y_0 + a_0 t $, where $a_0 t $ is the **deterministic trend** of this series. If we add a stationary part 

    $B(L) \varepsilon_t$ into this equation, we have a more general model: $$ y_t = y_0 + a_0 t + B(L) \varepsilon_t $$

  - this stochastic process means that even though $y_t$ would be different from its trend, but due to the stationary of $B(L) \varepsilon_t $, the long term predition would converge to its trend. We call it **Trend-stationary (TS)**. *Remember, trend must have long term effect on the series, hence, even though the past $\varepsilon_t $ has effect on the series, it fades in time. Hence, we don't consider them as trends.*

* **Stochastic trend and difference stationary**
  - Suppose we add a white noise on the simple model(Unit root process, random walk with drift): $\Delta y_t = a_0 + \varepsilon_t $, then the solution would now be: 

    $$ y_t = y_0 + a_0 t + \sum_{i=1}^{t} \varepsilon_i $$

  - In addition to the deterministic part, we have a stochastic part which would not fades in time (each stochastic shock would affect the series permanently). Hence, we call $\sum_{i=1}^{t} \varepsilon_i $ **stochastic trend**. 
  - Stochastic trend can usually be eliminate by taking difference, hence, we call the series with stochastic trend as the **difference stationary** series. For example, in the model above, we can take first order difference : $ \Delta y_t =  a_0  + \varepsilon_t $ and we have a stationary process.
&nbsp;

#### 4.2 Detrending
  * **Difference**: ARIMA model
    - Consider a more general model, random walk plus noise(stochastic trend): $y_t = y_0  + \sum_{i=1}^{t} \varepsilon_i  + \eta_t $, where $ \varepsilon_t $ and $ eta_t $ are both white noise and independent. 
    - Hence, we have that $ \Delta y_t = \varepsilon_t + \Delta \eta_t $, and it can be proved that it is a stationary process. And since it is a MA(1) process after first-order difference, hence, it is a ARIMA(0,1,1) model. Also, since adding a constant doesn't change the stationary of a process, trend plus noise model ($ y_t = y_0 + a_0 t + \sum_{i=1}^{t} \varepsilon_i $) is also a ARIMA(0,1,1) process.
    - **ARIMA(p,d,q)**: $A(L)y_t = B(L)\varepsilon_t$ , where $A(L) = (1-L)^d A^*(L) $ has $d$ unit roots, and $B(L)$ is stationary. Hence, ARIMA(p,d,q) can be written as : $$ A^*(L) \Delta^d y_t = B(L)\varepsilon_t $$ 

      Hence, an ARIMA(p,d,q) model means that after taking d-order difference, the process is an ARMA(p,q) process, and we also called it **d-order integrated**,$I(d)$.
  
  * **Detrending**
    - For those has the deterministic trend, difference may not be able to obtain a stationary series. For example: $$ y_t = y_0 + a_1 t + \varepsilon_t \Rightarrow \Delta y_t = a_1 + \Delta \varepsilon_t $$
    - Note that, this is an irreversible because the MA process is has unit root. 

    Because in estimating MA process, we use MLE and $ \Delta y_t = a_1 + (1-L) \varepsilon_t $ can not be written as $ \varepsilon_t = \frac{ \Delta y_t - a_1}{1-L}  $ |
    :--|

    - Hence, a proper way to detrend is to estimate the equation $ y_t = a_0 + a_1t + \varepsilon_t $ and obtain the estimated $ \{ \varepsilon_t \} $.
    - For a more general model, we can test for $ y_t = a_0 + a_1t + a_2 t^2 + ... + a_n t^n + e_t $ and the residuals $ \{ e_t \} $ is a stationary process. To decide n, we can try for a proper maximun n, and decrease untill the equation pass the tests(t-test, F-test, AIC, BIC, etc.)

  To conclude, *Difference Stationary(DS)* has *stochastic trend*, and need to be differenced to obtain stationary; *Trend Stationary(TS)* has *deterministic trend* which need to be eliminate to obtain stationary.|
  :--|   
&nbsp;

#### 4.3 Stationary and Spurious Regression
  * Stationary is one of the most important setting in time series analysis. One resason is that Stationary can replace the assumption of **random sampling**, which can hence justify the CLT and normality, therefore, we can still apply causal inference in time series data.
  * Another important reason for stationary is that it can **reduce the effect of spurious regression**. spurious regression is a phenomenon that regression suggest some correlation among variables which is not suppose to happend. 
  * **Spurious regression happends not only in unstationary series, but also in stationary series and cross section data. Because regression is a statistic analysis, it concentrate only on the statistic relation and not economic relation.** Hence, statistic relation only the necessary consition for economic relation. That's why causal effect can only comes from beyond statistic. Of course, unstationay series have greater chances in spurious regression due to the fact that they may have some common trend, such as time. **The only way to avoid spurious is setting the model properly.**

  ---
##### Experiment 7 Example of Spurious Regression

   * **Case 1** 
     - We construct two random walk:
       $$\begin{aligned}
       y_t &= y_{t-1} + \varepsilon_t \\
       z_t &= z_{t-1} + \eta_t
       \end{aligned} \quad
       $$ where $\{ \varepsilon_t \}$,$\{ \eta_t \}$ independly normal distributed. Hence, they have no meaningful relation. 
     - If we run the OLS of them, we would find that they are highly correlated: $$ y_t = 5.9089 - 0.7856 z_t $$ however, the residual is also highly autocorrelated, and even with unit root:$$ e_t = 0.9e_{t-1} $$ 
     - Note that both the series are unstationary, hence, the correlation OLS suggested is invalid.
  <center>
  <img src="figures\exp7-1.png" width="900">
  </center> 

  * **Case 2**
    - We now construct a random walk with drift: 
      $$\begin{aligned}
       y_t &=0.2 + y_{t-1} + \varepsilon_t \\
       z_t &=-0.1 + z_{t-1} + \eta_t
       \end{aligned} \quad
      $$
    - Since the drift term (end up with deterministic trend) dominated the process, we can predict that the correlation would be nagetive.
  <center>
  <img src="figures\exp7-2.png" width="900">
  </center> 

  It can be proved that in both random walk and random walk with drift, the correlation of residual would converge to 1.|
  :--|

```python
# Generate data
rand1 = np.random.normal(0,1,100)
rand2 = np.random.normal(0,1,100)
z_t = [0]
y_t = [0]
for i in range(1,100):
    # random walk
    y_t.append(y_t[i-1] + rand1[i-1] )
    z_t.append(z_t[i-1] + rand2[i-1] )
    # random walk with drift
    # y_t.append(0.2+y_t[i-1] + rand1[i-1] )
    # z_t.append(-0.1+z_t[i-1] + rand2[i-1] )
    
# run ols on the y and z
X = sm.add_constant(z_t)
m1 = sm.OLS(y_t,X).fit()
print(m1.summary())

# test the residuals for correlation
m2 = sm.tsa.ARMA(m1.resid,order=(1,0)).fit(trend="nc")
print(m2.summary())

fig,axes = plt.subplots(1,3,figsize = (24,4))
axes[0].plot(z_t,label = r'$z_t$')
axes[0].plot(y_t,label = r"$y_t$" )
axes[0].set_title(r"$ y_t \ and \ z_t $",fontsize = 20)
axes[0].legend(fontsize = 15)
axes[1].scatter(z_t,y_t)
axes[1].plot(z_t,m1.predict(),c="r")
axes[1].set_title("Regression",fontsize = 20)
axes[2].scatter(m1.resid[:-1],m1.resid[1:])
axes[2].set_title("Residuals",fontsize = 20)
axes[2].set_xlabel(r"$e_{t-1}$",fontsize = 15)
axes[2].set_ylabel(r"$e_t$",fontsize = 15)
```
---
&nbsp;

#### 4.4 Monte-Carlo Simulation
  * We already know that for a unit root process, we need to difference, and for a deterministic trend process, we need to detrend. The problem is how do we know whether the series is DS or TS.
  * **t-Test and ACF graphs**
    - Consider the an AR(1) process: $ y_t = a_0 + a_1 y_{t-1} + \varepsilon_t $
    - **Under $a_1=0$ assumption**: we can run OLS and estimate $a_1$ and its standard error, then we can apply t-test for it.
    - However, **Under $ a_1 =1$ assumption**, $y_t = y_0 + \sum_{i=1}^{t} \varepsilon_i$, the variation tends to $\infty$ when t increase. Hence, we can run t-test on $a_1$.
    <center>
    <img src="figures\fig4.4.png" width="900">
    </center> 

    - As we can see from the figure, a unit root process and a stationary AR(1) process can be very similar in ACF. Hence, it is not accurate to conclude the stationarity by the ACF graphs.
  * **Monte-Carlo and DF distribution**
    - Even though we can apply t-test on $a_1$, but if we know the distribution of a_1 under the assumption of $a_1 =1$, we can still apply a similar test on $a_1$.
    - Unlike the stardard distribution(t,F,$\chi^2$), the distribution of $a_1$ doesn't have an analytical form. Hence, we need to apply Monte-Carlo simulation to calculate its distribution.
    - The idea of Monte-Carlo simulation is to apply the **Law of Large Number**, with large enough sample, we can generate the distribution of an unknow and complex distribution.
      1. Generate a random series with T samples (it can be any known distribution), usually we use standard normal distribution. 
      2. Consruct the series we wants to test(for example:$y_t = y_{t-1} + \varepsilon_t$). If we don't want the initial value to affect the result, we can generate T+N samples, and use only the last T of the series to calculate.
      3. Estimate the model($ \Delta y_t = a_0 + \gamma y_{t-1} + \varepsilon_t $, $\gamma = a_1-1$) **under the alternative assumption**($\gamma < 0$), and calculate the t-statistic.
      4. Repeat 1-3 many time(10000 or more), and we have a distribution on $a_1$.

  ---
##### Experiment 8 Calculate CF distribution with Monte-Carlo Simulation

  * According to algrithm above, I write a python programme to calculate the distribution of $\gamma$ in $ \Delta y_t = a_0 + \gamma y_{t-1} + \varepsilon_t $. 
  * As we can see, the value are very close, and if I set the iteration(repeat) larger, I would have a much accurate distribution.

  <center class = 'third'>
    <img src="figures\exp8-1.png" width="200"><img src="figures\exp8-3.png" width="300"><img src="figures\exp8-2.png" width="300">
  </center>

```python
def get_pct(T,iteration = 10000):
    t_value = []
    index = range(0,iteration)
    while iteration>0:
        iteration -=1
        # generate 150 random number 
        rand1 = np.random.normal(0,1,T+50)
        y_t = [0]
        # construct series y_t
        for i in range(1,T+51):
            y_t.append(y_t[i-1] + rand1[i-1] )
        # abandon the first 50 number to obtain accuracy
        y_t = np.array(y_t[-T:])
        # construct \delta_y and y_{t-1}
        y_t_1 = y_t[:-1]
        d_y = y_t[1:] - y_t_1
        x = sm.add_constant(y_t_1)
        # estimate \delta y_t = a_0 + \gamma y_{t-1} + e_t
        m = sm.OLS(d_y,x).fit()
        t_value.append(m.tvalues[1])
    # sort the t_value
    DF = pd.DataFrame(t_value).sort_values(by=0)
    DF.index = index
    # obtain the thresholds
    pct = [0.01,0.025,0.05,0.1]
    tre = []
    for p in pct:
        n = int(DF.shape[0] * (1-p))
        tre.append(DF.iloc[-n,0])
    thred = pd.DataFrame(tre).T
    thred.columns = pct
    thred.index = [T]
    return thred.round(2),DF.round(2)

distribution = pd.DataFrame(columns=[0.01,0.025,0.05,0.1])
sample = [25,50,100,250,300,5000]
for s in sample:
    thred,DF = get_pct(s,iteration=30000)
    distribution = distribution.append(thred)
```
---
&nbsp;

#### 4.5 DF and ADF Test
  * Dickey and Fuller(1979) propese three different equation to test unit root: $$\begin{aligned}  \Delta y_t &= \gamma y_{t-1} + \varepsilon_t  &(1) \\
    \Delta y_t &= a_0 + \gamma y_{t-1} + \varepsilon_t &(2) \\
    \Delta y_t &= a_0 +  \gamma y_{t-1} + a_2 t + \varepsilon_t &(3)
    \end{aligned}$$ The difference among them is whether include deterministic trend $a_0$ and $a_2 t$, and each of the equation has a corresponding(different) distritionof $\gamma$, and **if the t-statistic of estimated $\gamma$: $\tau_{\gamma} > t_{\alpha} $, then we can't reject the existence of unit root, under siginican level of $\alpha$.(at leat exist one unit root)**

  * Not all series can fit the AR(1) process well, hence, to extend the idea of DF test, we need to estimate higher order model. Dickey and Fuller(1981) propose an **augmented DF tests(ADF)**, in order to fit higher order model. ADF also constructed with three model: $$\begin{aligned}  \Delta y_t &= \gamma y_{t-1} + \sum_{i=2}^{p} \beta_i \Delta y_{t-i+1} + \varepsilon_t  &(4) \\
    \Delta y_t &= a_0 + \gamma y_{t-1}  + \sum_{i=2}^{p} \beta_i \Delta y_{t-i+1} + \varepsilon_t &(5) \\
    \Delta y_t &= a_0 +  \gamma y_{t-1} + a_2 t + \sum_{i=2}^{p} \beta_i \Delta y_{t-i+1} + \varepsilon_t &(6)
    \end{aligned}$$ To understand ADF, we can consider a AR(p) process: $$ y_t = a_0 + a_1 y_{t-1} +a_2 y_{t-2} + ... + a_{p-1} y_{t-p+1} + a_p y_{t-p} $$ Transform this equation by add and minus $  a_p y_{t-p+1}  $ simultaneously: $$ y_t = a_0 + a_1 y_{t-1} +a_2 y_{t-2} + ... + (a_{p-1} + a_p) y_{t-p+1} - a_p \Delta y_{t-p+1}  $$ We can repeat this process and obtain $$\Delta y_t = \gamma y_{t-1} + \sum_{i=2}^{p} \beta_i \Delta y_{t-i+1} + \varepsilon_t $$  where, $ \gamma = \sum\limits_{i=1}^{p} a_i -1 $ and $ \beta_i = -\sum\limits_{j=i}^{p} a_j $
    Hence, test for $\gamma=0$ is equivalent to test the existence of unit root.(can not specify which term though)

    **It can be proved that ADF tests have the same critical value as the DF tests.**

  * In addition, Dickey and Fuller(1981) added three F-statistic to test the joinly assumption of the equation, it is calculated by the traditional way: $$ F = \frac{ \frac{SSR_r - SSR_u}{r} }{ \frac{SSR_u}{T-k} } \sim F(r,T-k)  $$ where ${SSR}_r$ and $SSR_u$ is the restricted and unrestricted SSR, r is the number of restriction, T is the number of samples, and k is the parameters of unrestricted model. The main statistic of DF test:
  
  <center>
  <img src="figures\fig4.5.1.png" width="600">
  </center>
  <center>
    <img src="figures\fig4.5.png" width="800">
  </center>

  * **In practice, we test the series start from (6) or (1), and then (5) or (2), finally we test (4) or (1). We stop until the model has no unit root**. For example, if we test (3) and (2), (3) suggests unit root and (2) suggests not, then we can suspect the series has deterministic trend.

---
##### Experiment 9 ADF test for real consumption per capita in China

  * We use the data of consumption from 1980 to 2018. The raw data comes from [National Bureau of Statistic](http://www.stats.gov.cn/), and we compute the real consumption per capita using the nominal consumption per capita and CPI(1978=100) $ rcon = Consumption/CPI \times 100 $
  * First, we observe the basic feature of the data. The rcon has obvious trend and the ACF is very typical: easily confused with AR(1) process. And if we estimate AR(1), we have an equation: $$ rcon_t = 707.51 + 0.9968 rcon_{t-1} $$ And the residuals are highly autocorrelated, hence, it is reasonal to suspect the existence of unit root.

  <center>
    <img src="figures\exp9-1.png" width="800">
  </center>

  * **ADF for rcon**
    First, we estimate the model with constant and trend: $$M3: \quad \Delta y_t =6.1912 -0.2129 \times y_{t-1} + 0.6851 \times \Delta y_{t-1} + 6.1912 \times t $$ Where $ t_{0.05} < \tau_{\gamma} < t_{0.1}$,hence, under significant level of 5%, we **can't reject the existence of unit root**. And the t-statiscic for trend is 3.37, which is larger than the  critical value under significant level of 5% (our observation is 37,and the critical value of 25 is 2.85, of 50 is 2.81), hence, we **reject null assumption of no trend**.
    Problem raise here, the trend term is significant in this model and unit root also exist, hence, **it is not reasonable to estimate model 2**(with constant only). Then, we can try to relex the assumption by **compare to t-distribution**. And we find that even in t-distribution, the trend term is still significant, hence, we can conclude that **the series is not stationary**.
    <center class='half'>
    <img src="figures\exp9-2.png" width="380"><img src="figures\exp9-3.png" width="400">
    </center>
  * **Intergation**
    We want to further know that if the series is I(d) series, we can difference the serise and test for unit root. We run the following model: $$M3: \quad \Delta y_t =20.9642 -0.6929 \times y_{t-1} + 0.511 \times \Delta y_{t-1} -0.0129 \times \Delta y_{t-2} + 0.3488 \times \Delta y_{t-3} - 0.0514\times t  $$ Where $\tau_{\gamma} > t_{0.05}$ and $\tau_{a_2} < t_{0.05} $, hence, **we can not reject the existence of unit root but can reject the existent of trend**. We can further estimate the model 2: $$M2: \quad \Delta y_t =20.1196 -0.6949 \times y_{t-1} + 0.514 \times \Delta y_{t-1} -0.0111 \times \Delta y_{t-2} + 0.3512 \times \Delta y_{t-3} $$ Where $\tau_{\gamma} < t_{0.05}$, hence we can reject the existence of unit root, and conclude that **the first-order difference of rcon is stationary, i.e., $I(1)$.**
  
  * We can also test for the growth rate and the logarithm of the real consumption. And we have that **the growth rate of real consumption is stationary, and the logarithmic real consumption is not.** Since the first-order difference of logarithmic real consumption is approximatly the growth rate, hence, it is obvious that the logarithmic real consumption is I(1).
  <center class='third'>
  <img src="figures\exp9-4.png" width="250"><img src="figures\exp9-5.png" width="245"><img src="figures\exp9-6.png" width="280">
  </center>

```python
cons = pd.read_excel("./data/Consumption.xls",index_col="Year")
rcon = cons['Rcon']

# graphs of the data
fig,axes = plt.subplots(1,3,figsize=(24,4))
axes[0].plot(rcon)
axes[0].set_title("Real Consumption per capita",fontsize = 15)
sm.graphics.tsa.plot_acf(rcon,ax=axes[1])
axes[1].set_title("Autocorrelation",fontsize = 15)

ar1 = sm.tsa.ARMA(rcon,order=(1,0)).fit()
axes[2].scatter(ar1.resid[:-1],ar1.resid[1:])
axes[2].set_title("Residuals",fontsize = 15)
axes[2].set_xlabel(r"$e_{t-1}$")
axes[2].set_ylabel(r"$e_t$")
axes[2].set_xlim(-20,100)

# ADF test for Rcon
"""
statsmodels.tsa.stattools.adfuller(
    x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)

  regression{‘c’,’ct’,’ctt’,’nc’}
    Constant and trend order to include in regression
        ‘c’ : constant only (default)
        ‘ct’ : constant and trend
        ‘ctt’ : constant, and linear and quadratic trend
        ‘nc’ : no constant, no trend
  regresults: y_{t-1}, \Delta y_{t-1},...,\Delta y_{t-1},constant,t,t^2
"""
# test for contant and trend
def adf_test(data,title):
    m3 = sm.tsa.stattools.adfuller(data,regression="ct")
    m2 = sm.tsa.stattools.adfuller(data,regression="c")
    m1 = sm.tsa.stattools.adfuller(data,regression="nc")
    print("{:^40}".format("Results of " + title) )
    print("-"*40)
    print("{:^10}{:^10}{:^10}{:^10}".format('','M3',"M2","M1"))
    print("{:^10}{:^10}{:^10}{:^10}".format('lags',m3[2],m2[2],m1[2]))
    print("{:^10}{:^10}{:^10}{:^10}".format(
          'stats',m3[0].round(3),m2[0].round(3),m1[0].round(3))
          )
    print("{:^10}{:^10}{:^10}{:^10}".format(
          'p-value',m3[1].round(3),m2[1].round(3),m1[1].round(3))
          )
    print("{:^10}{:^10}{:^10}{:^10}".format(
          '1%',m3[4]["1%"].round(3),m2[4]["1%"].round(3),m1[4]["1%"].round(3))
          )
    print("{:^10}{:^10}{:^10}{:^10}".format(
          '5%',m3[4]["5%"].round(3),m2[4]["5%"].round(3),m1[4]["5%"].round(3))
          )
    print("{:^10}{:^10}{:^10}{:^10}".format(
          '10%',m3[4]["10%"].round(3),m2[4]["10%"].round(3),m1[4]["10%"].round(3))
          )
    print("-"*40)
adf_test(rcon,"rcon")

# check for regression
m3 = sm.tsa.stattools.adfuller(rcon,regression="ct",regresults=True)[-1]
m3.resols.summary()
```
---

---
##### Experiment 10 Application of ADF on Economic Theory: Business Cycle and PPP
  * **Busniness Cycle**
    In traditional economic thoery, people tends to decompose the real variables into two parts: **a secular(growth) component** and **a cyclical component**. And the growth part is usually considered decided by population, education, capital or technology. Especially in neoclassical macroeconomics(RBC theory), the long-term trend is determined by technology. However, One problem is that if we assume the technology shock is stochastic, it shouldn't have a deterministic trend in long-term growth.
  
    <center >
    <img src="figures\exp10-1.png" width="800">
    </center>

    - **Nelson and Plosser(1982) prove that some important macroeconomic series tends to be DS(stochastic trend) rather than TS(deterministic trend)**.
    &nbsp;
    - **Evidence from autocorrelation**
      1. Most of the series(except unemployment rate) is consistent with the behavior of unit root process. The autocorrelation of unemployment rate  decrese rapidly, hence, it can be deemed as a stationary series.
      2. The first order difference of real GNP, nominal GNP, real per capita GNP, employment, nominal and real wages, and common stock prices also indicate the DS process, which is the behavior of the first order difference of MA process.((Recall that first order difference of DS process is an inversible MA process)
      Hence, they are no necessary TS process intuitively.
      3. Also, if these series indeed TS process, then after detreding, we should obtain stationary series. However, the autocorrelation of the deviation still implies that these series(except the unemplyment rate) are probably DS process, which also proof that detrend the DS process would not eliminate the non-stationarity.

    <center >
    <img src="figures\exp10-2-1.png" width="600">
    <img src="figures\exp10-2-2.png" width="600">
    <img src="figures\exp10-3.png" width="600">
    </center>

     
    - **Evidence from ADF test**
      We use ADF test with constant and trend to test for the unit root. According to the result, only unemployment rate can reject the existence of unit root, which consolidates our analysis before. As for the trend part, we can't reject the absence of trend in real GNP,real per capita GNP,industrial production, employment, real wages, money stock and common stock prices, which means that they are probably not TS series.
    
    <center >
    <img src="figures\exp10-4.png" width="800">
    </center>

```python
# Business Cycle (Nelson and Plosser,1982)
# Detrend and cylical graph
Data = pd.read_excel('./data/nelson-plosser 1982.xls',index_col='year')
rgnp = Data['rgnp'].dropna()
time = rgnp.index
x = np.column_stack((time,time**2))
x = sm.add_constant(x)
m = sm.OLS(rgnp,x).fit()

fig,axes = plt.subplots(1,2,figsize=(20,5))
axes[0].plot(time,rgnp,label="rgnp")
axes[0].plot(time,m.predict(),
              label=r"$rgnp = 732647.89 -764.61 *t +0.2*t^2$")
axes[0].legend(fontsize=15)
axes[0].set_title('Deterministic Trend ?',fontsize=20)
axes[1].plot(m.resid)
axes[1].plot(time,np.zeros_like(time))
axes[1].set_title('Business Cycle ?',fontsize=20)

# Evidence from autocorrelation
# autocorrelation of natural logs
var = Data.columns[14:]
result = pd.DataFrame(columns=var)
for v in var:
    tmp = Data[v].dropna()
    n = tmp.shape[0]
    a = sm.tsa.acf(tmp,fft=True)
    b = np.hstack(([n],a[1:7].round(2)))
    result[v] = b
result.index = ["T",'r1','r2','r3','r4','r5','r6']
result = result.T

# autocorrelation of first-order difference natural logs
d_Data = Data.diff()
result = pd.DataFrame(columns=var)
for v in var:
    tmp = d_Data[v].dropna()
    n = tmp.shape[0]
    a = sm.tsa.acf(tmp,fft=True)
    b = np.hstack(([n],a[1:7].round(2)))
    result[v] = b
result.index = ["T",'r1','r2','r3','r4','r5','r6']
result = result.T

# autocorrelation of deviation from trend
deviation = pd.DataFrame(columns=var)
for v in var:
    tmp = Data[v].dropna()
    time = tmp.index
    n = tmp.shape[0]
    x = sm.add_constant(time)
    m = sm.OLS(tmp,x).fit().resid
    a = sm.tsa.acf(m,fft=True)
    b = np.hstack(([n],a[1:7].round(2)))
    deviation[v] = b
deviation.index = ["T",'r1','r2','r3','r4','r5','r6']
deviation = deviation.T

# evidence from ADF test
ADF = pd.DataFrame(columns=var)
for v in var:
    tmp = Data[v].dropna()
    n = tmp.shape[0]
    m1 = sm.tsa.adfuller(tmp,regression="ct")
    m2 = sm.tsa.adfuller(tmp,regression="ct",regresults=True)[-1].resols
    k = m1[2]
    a_0 = m2.params[-2].round(3)
    a_2 = m2.params[-1].round(3)
    gamma = m2.params[0].round(3)
    a_0_t = m2.tvalues[-2].round(3)
    a_2_t = m2.tvalues[-1].round(3)
    a_2_p = m2.pvalues[-1].round(3)
    gamma_t = m2.tvalues[0].round(3)
    
    lit = [n,k,a_0,a_0_t,a_2,a_2_t,a_2_p,gamma,gamma_t,1+gamma]
    ADF[v] = lit
index = ["T","lags","a_0",'t(a_0)',"a_2",
        't(a_2)','p(a_2)',"gamma",'t(gamma)','a_1']
ADF.index=index
ADF = ADF.T
```
---


  * In DF or ADF tests, since we don't know the data generation process(DGP), we may cause some important problems: 
    1. The lags terms, unless we include all the proper lags terms, we can estimate the model properly.
    2. DGP may include MA process
    3. DF and ADF only consider one unit root
    4. Some series contains seasonal lags
    5. Data structral breaks may change the trends
    6. When to conclude the intercept and the time trend.

