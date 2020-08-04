---
title: "Campus Recruitment Statistics"
date: 2020-08-04
tags: [data wrangling, statistic, messy data, data science]
header:
  image: ""
excerpt: "data wrangling, statistic, messy data, data science"
mathjax: "true"
---


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import ttest_1samp
import math
```


```python
data = pd.read_csv('Placement_Data_Full_Class.csv')
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sl_no</th>
      <th>gender</th>
      <th>ssc_p</th>
      <th>ssc_b</th>
      <th>hsc_p</th>
      <th>hsc_b</th>
      <th>hsc_s</th>
      <th>degree_p</th>
      <th>degree_t</th>
      <th>workex</th>
      <th>etest_p</th>
      <th>specialisation</th>
      <th>mba_p</th>
      <th>status</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>M</td>
      <td>67.00</td>
      <td>Others</td>
      <td>91.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>58.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>55.0</td>
      <td>Mkt&amp;HR</td>
      <td>58.80</td>
      <td>Placed</td>
      <td>270000.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>M</td>
      <td>79.33</td>
      <td>Central</td>
      <td>78.33</td>
      <td>Others</td>
      <td>Science</td>
      <td>77.48</td>
      <td>Sci&amp;Tech</td>
      <td>Yes</td>
      <td>86.5</td>
      <td>Mkt&amp;Fin</td>
      <td>66.28</td>
      <td>Placed</td>
      <td>200000.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>M</td>
      <td>65.00</td>
      <td>Central</td>
      <td>68.00</td>
      <td>Central</td>
      <td>Arts</td>
      <td>64.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>75.0</td>
      <td>Mkt&amp;Fin</td>
      <td>57.80</td>
      <td>Placed</td>
      <td>250000.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>M</td>
      <td>56.00</td>
      <td>Central</td>
      <td>52.00</td>
      <td>Central</td>
      <td>Science</td>
      <td>52.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>66.0</td>
      <td>Mkt&amp;HR</td>
      <td>59.43</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>M</td>
      <td>85.80</td>
      <td>Central</td>
      <td>73.60</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>73.30</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>96.8</td>
      <td>Mkt&amp;Fin</td>
      <td>55.50</td>
      <td>Placed</td>
      <td>425000.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>210</td>
      <td>211</td>
      <td>M</td>
      <td>80.60</td>
      <td>Others</td>
      <td>82.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>77.60</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>91.0</td>
      <td>Mkt&amp;Fin</td>
      <td>74.49</td>
      <td>Placed</td>
      <td>400000.0</td>
    </tr>
    <tr>
      <td>211</td>
      <td>212</td>
      <td>M</td>
      <td>58.00</td>
      <td>Others</td>
      <td>60.00</td>
      <td>Others</td>
      <td>Science</td>
      <td>72.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>74.0</td>
      <td>Mkt&amp;Fin</td>
      <td>53.62</td>
      <td>Placed</td>
      <td>275000.0</td>
    </tr>
    <tr>
      <td>212</td>
      <td>213</td>
      <td>M</td>
      <td>67.00</td>
      <td>Others</td>
      <td>67.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>73.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>59.0</td>
      <td>Mkt&amp;Fin</td>
      <td>69.72</td>
      <td>Placed</td>
      <td>295000.0</td>
    </tr>
    <tr>
      <td>213</td>
      <td>214</td>
      <td>F</td>
      <td>74.00</td>
      <td>Others</td>
      <td>66.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>58.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>70.0</td>
      <td>Mkt&amp;HR</td>
      <td>60.23</td>
      <td>Placed</td>
      <td>204000.0</td>
    </tr>
    <tr>
      <td>214</td>
      <td>215</td>
      <td>M</td>
      <td>62.00</td>
      <td>Central</td>
      <td>58.00</td>
      <td>Others</td>
      <td>Science</td>
      <td>53.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>89.0</td>
      <td>Mkt&amp;HR</td>
      <td>60.22</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>215 rows × 15 columns</p>
</div>




```python
data.isnull().sum(axis=0)
```




    sl_no              0
    gender             0
    ssc_p              0
    ssc_b              0
    hsc_p              0
    hsc_b              0
    hsc_s              0
    degree_p           0
    degree_t           0
    workex             0
    etest_p            0
    specialisation     0
    mba_p              0
    status             0
    salary            67
    dtype: int64




```python
index = data.index
num = len(index)
num
```




    215




```python
male = []
female = []

for i in range(num):
    if data['gender'][i] == 'M':
        male = data.loc[(data['gender'] == 'M')]

    elif data['gender'][i] == 'F':
        female = data.loc[(data['gender'] == 'F')]

```


```python
female
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sl_no</th>
      <th>gender</th>
      <th>ssc_p</th>
      <th>ssc_b</th>
      <th>hsc_p</th>
      <th>hsc_b</th>
      <th>hsc_s</th>
      <th>degree_p</th>
      <th>degree_t</th>
      <th>workex</th>
      <th>etest_p</th>
      <th>specialisation</th>
      <th>mba_p</th>
      <th>status</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>F</td>
      <td>46.00</td>
      <td>Others</td>
      <td>49.2</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>79.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>74.28</td>
      <td>Mkt&amp;Fin</td>
      <td>53.29</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>12</td>
      <td>13</td>
      <td>F</td>
      <td>47.00</td>
      <td>Central</td>
      <td>55.0</td>
      <td>Others</td>
      <td>Science</td>
      <td>65.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>62.00</td>
      <td>Mkt&amp;HR</td>
      <td>65.04</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>13</td>
      <td>14</td>
      <td>F</td>
      <td>77.00</td>
      <td>Central</td>
      <td>87.0</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>59.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>68.00</td>
      <td>Mkt&amp;Fin</td>
      <td>68.63</td>
      <td>Placed</td>
      <td>218000.0</td>
    </tr>
    <tr>
      <td>15</td>
      <td>16</td>
      <td>F</td>
      <td>65.00</td>
      <td>Central</td>
      <td>75.0</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>69.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>72.00</td>
      <td>Mkt&amp;Fin</td>
      <td>64.66</td>
      <td>Placed</td>
      <td>200000.0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>18</td>
      <td>F</td>
      <td>55.00</td>
      <td>Central</td>
      <td>67.0</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>64.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>60.00</td>
      <td>Mkt&amp;Fin</td>
      <td>67.28</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>197</td>
      <td>198</td>
      <td>F</td>
      <td>83.96</td>
      <td>Others</td>
      <td>53.0</td>
      <td>Others</td>
      <td>Science</td>
      <td>91.0</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>59.32</td>
      <td>Mkt&amp;HR</td>
      <td>69.71</td>
      <td>Placed</td>
      <td>260000.0</td>
    </tr>
    <tr>
      <td>198</td>
      <td>199</td>
      <td>F</td>
      <td>67.00</td>
      <td>Central</td>
      <td>70.0</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>65.0</td>
      <td>Others</td>
      <td>No</td>
      <td>88.00</td>
      <td>Mkt&amp;HR</td>
      <td>71.96</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>204</td>
      <td>205</td>
      <td>F</td>
      <td>74.00</td>
      <td>Others</td>
      <td>73.0</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>73.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>80.00</td>
      <td>Mkt&amp;Fin</td>
      <td>67.69</td>
      <td>Placed</td>
      <td>210000.0</td>
    </tr>
    <tr>
      <td>208</td>
      <td>209</td>
      <td>F</td>
      <td>43.00</td>
      <td>Central</td>
      <td>60.0</td>
      <td>Others</td>
      <td>Science</td>
      <td>65.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>92.66</td>
      <td>Mkt&amp;HR</td>
      <td>62.92</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>213</td>
      <td>214</td>
      <td>F</td>
      <td>74.00</td>
      <td>Others</td>
      <td>66.0</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>58.0</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>70.00</td>
      <td>Mkt&amp;HR</td>
      <td>60.23</td>
      <td>Placed</td>
      <td>204000.0</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 15 columns</p>
</div>




```python
male
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sl_no</th>
      <th>gender</th>
      <th>ssc_p</th>
      <th>ssc_b</th>
      <th>hsc_p</th>
      <th>hsc_b</th>
      <th>hsc_s</th>
      <th>degree_p</th>
      <th>degree_t</th>
      <th>workex</th>
      <th>etest_p</th>
      <th>specialisation</th>
      <th>mba_p</th>
      <th>status</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>M</td>
      <td>67.00</td>
      <td>Others</td>
      <td>91.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>58.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>55.0</td>
      <td>Mkt&amp;HR</td>
      <td>58.80</td>
      <td>Placed</td>
      <td>270000.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>M</td>
      <td>79.33</td>
      <td>Central</td>
      <td>78.33</td>
      <td>Others</td>
      <td>Science</td>
      <td>77.48</td>
      <td>Sci&amp;Tech</td>
      <td>Yes</td>
      <td>86.5</td>
      <td>Mkt&amp;Fin</td>
      <td>66.28</td>
      <td>Placed</td>
      <td>200000.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>M</td>
      <td>65.00</td>
      <td>Central</td>
      <td>68.00</td>
      <td>Central</td>
      <td>Arts</td>
      <td>64.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>75.0</td>
      <td>Mkt&amp;Fin</td>
      <td>57.80</td>
      <td>Placed</td>
      <td>250000.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>M</td>
      <td>56.00</td>
      <td>Central</td>
      <td>52.00</td>
      <td>Central</td>
      <td>Science</td>
      <td>52.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>66.0</td>
      <td>Mkt&amp;HR</td>
      <td>59.43</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>M</td>
      <td>85.80</td>
      <td>Central</td>
      <td>73.60</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>73.30</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>96.8</td>
      <td>Mkt&amp;Fin</td>
      <td>55.50</td>
      <td>Placed</td>
      <td>425000.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>209</td>
      <td>210</td>
      <td>M</td>
      <td>62.00</td>
      <td>Central</td>
      <td>72.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>65.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>67.0</td>
      <td>Mkt&amp;Fin</td>
      <td>56.49</td>
      <td>Placed</td>
      <td>216000.0</td>
    </tr>
    <tr>
      <td>210</td>
      <td>211</td>
      <td>M</td>
      <td>80.60</td>
      <td>Others</td>
      <td>82.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>77.60</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>91.0</td>
      <td>Mkt&amp;Fin</td>
      <td>74.49</td>
      <td>Placed</td>
      <td>400000.0</td>
    </tr>
    <tr>
      <td>211</td>
      <td>212</td>
      <td>M</td>
      <td>58.00</td>
      <td>Others</td>
      <td>60.00</td>
      <td>Others</td>
      <td>Science</td>
      <td>72.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>74.0</td>
      <td>Mkt&amp;Fin</td>
      <td>53.62</td>
      <td>Placed</td>
      <td>275000.0</td>
    </tr>
    <tr>
      <td>212</td>
      <td>213</td>
      <td>M</td>
      <td>67.00</td>
      <td>Others</td>
      <td>67.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>73.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>59.0</td>
      <td>Mkt&amp;Fin</td>
      <td>69.72</td>
      <td>Placed</td>
      <td>295000.0</td>
    </tr>
    <tr>
      <td>214</td>
      <td>215</td>
      <td>M</td>
      <td>62.00</td>
      <td>Central</td>
      <td>58.00</td>
      <td>Others</td>
      <td>Science</td>
      <td>53.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>89.0</td>
      <td>Mkt&amp;HR</td>
      <td>60.22</td>
      <td>Not Placed</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>139 rows × 15 columns</p>
</div>




```python
female = female.fillna(female.mean())
male = male.fillna(male.mean())
```


```python
female_mean = round(female['salary'].mean(), 2)
male_mean = round(male['salary'].mean(), 2)
print(female_mean, male_mean)
print(round(male_mean - female_mean, 2))
```

    267291.67 298910.0
    31618.33


the difference in starting salary between men and woman is 31618.33


```python
sns.pairplot(data)
```

    C:\Users\44738\AppData\Local\Continuum\anaconda3\lib\site-packages\numpy\lib\histograms.py:824: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    C:\Users\44738\AppData\Local\Continuum\anaconda3\lib\site-packages\numpy\lib\histograms.py:825: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)





    <seaborn.axisgrid.PairGrid at 0x1ec7f330848>




![png](Campus_Recruitment_T_test_files/Campus_Recruitment_T_test_10_2.png)


Q1: Is there statistical evidence to suggest we can reject the null hypothesis i.e 60%.


```python
data['degree_p'].hist()
plt.axvline(60, 0, 100, label='pyplot vertical line', c='r')
plt.show()
```


![png](Campus_Recruitment_T_test_files/Campus_Recruitment_T_test_12_0.png)



```python
sample_data = data.sample(100)
```


```python
sample_mean = sample_data['degree_p'].mean()
sample_mean
```




    66.76410000000001




```python
mean1 = sample_data['degree_p'].mean()
mean1
std = np.std(sample_data['degree_p'])
std
```




    6.962074210319797




```python
def ttest(mean_data, mu, std_dev, n):
    t = (mean_data-mu)/(std_dev/math.sqrt(n-1))
    return t
```


```python
ttest(mean1, 60, std, 100)
```




    9.677228695570832




```python
t_sample = []
sample_mean = []
for i in range(100):
    sample_data = data['degree_p'].sample(30)
    sample_mean = sample_data.mean()
    t_sample.append(ttest(sample_mean, 60, std, 30))
```


```python
np.min(t_sample)
```




    2.9645678898706813




```python

```


```python
plt.hist(t_sample, bins = 50)
```




    (array([1., 1., 0., 0., 2., 0., 0., 1., 1., 2., 1., 4., 2., 4., 2., 5., 2.,
            1., 3., 5., 1., 5., 3., 5., 2., 1., 3., 5., 3., 2., 3., 3., 2., 5.,
            1., 4., 3., 0., 1., 2., 1., 2., 1., 2., 0., 1., 1., 0., 0., 1.]),
     array([2.96456789, 3.04284094, 3.12111399, 3.19938704, 3.27766009,
            3.35593314, 3.43420619, 3.51247923, 3.59075228, 3.66902533,
            3.74729838, 3.82557143, 3.90384448, 3.98211753, 4.06039058,
            4.13866363, 4.21693668, 4.29520973, 4.37348278, 4.45175583,
            4.53002887, 4.60830192, 4.68657497, 4.76484802, 4.84312107,
            4.92139412, 4.99966717, 5.07794022, 5.15621327, 5.23448632,
            5.31275937, 5.39103242, 5.46930547, 5.54757851, 5.62585156,
            5.70412461, 5.78239766, 5.86067071, 5.93894376, 6.01721681,
            6.09548986, 6.17376291, 6.25203596, 6.33030901, 6.40858206,
            6.48685511, 6.56512815, 6.6434012 , 6.72167425, 6.7999473 ,
            6.87822035]),
     <a list of 50 Patch objects>)




![png](Campus_Recruitment_T_test_files/Campus_Recruitment_T_test_21_1.png)



```python

```


```python
print("95 percent confidence interval : ",st.t.interval(0.95,len(sample_data)-1, loc=np.mean(sample_data['degree_p']), scale=st.sem(sample_data['degree_p'])) )
```

    95 percent confidence interval :  (65.42589693112755, 68.10230306887247)



```python

```
