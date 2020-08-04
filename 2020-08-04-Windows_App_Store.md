---
title: "Windows App Store Analysis"
date: 2020-08-04
tags: [data wrangling, statistic, messy data, data visualization]
header:
  image:
excerpt: "data wrangling, statistic, messy data, data visualization"
mathjax: "True"
---


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv("msft.csv")
```


```python
#data = data.drop([5321], axis = 0)
```


```python
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
      <th>Name</th>
      <th>Rating</th>
      <th>No of people Rated</th>
      <th>Category</th>
      <th>Date</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Dynamic Reader</td>
      <td>3.5</td>
      <td>268</td>
      <td>Books</td>
      <td>07-01-2014</td>
      <td>Free</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Chemistry, Organic Chemistry and Biochemistry-...</td>
      <td>3.0</td>
      <td>627</td>
      <td>Books</td>
      <td>08-01-2014</td>
      <td>Free</td>
    </tr>
    <tr>
      <td>2</td>
      <td>BookViewer</td>
      <td>3.5</td>
      <td>593</td>
      <td>Books</td>
      <td>29-02-2016</td>
      <td>Free</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Brick Instructions</td>
      <td>3.5</td>
      <td>684</td>
      <td>Books</td>
      <td>30-01-2018</td>
      <td>Free</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Introduction to Python Programming by GoLearni...</td>
      <td>2.0</td>
      <td>634</td>
      <td>Books</td>
      <td>30-01-2018</td>
      <td>Free</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>5316</td>
      <td>Get Color</td>
      <td>3.0</td>
      <td>785</td>
      <td>Developer Tools</td>
      <td>08-08-2019</td>
      <td>₹ 54.50</td>
    </tr>
    <tr>
      <td>5317</td>
      <td>JS King</td>
      <td>1.0</td>
      <td>720</td>
      <td>Developer Tools</td>
      <td>19-07-2018</td>
      <td>₹ 269.00</td>
    </tr>
    <tr>
      <td>5318</td>
      <td>MQTTSniffer</td>
      <td>2.5</td>
      <td>500</td>
      <td>Developer Tools</td>
      <td>10-04-2017</td>
      <td>₹ 64.00</td>
    </tr>
    <tr>
      <td>5319</td>
      <td>Dev Utils - JSON, CSV and XML</td>
      <td>4.0</td>
      <td>862</td>
      <td>Developer Tools</td>
      <td>18-11-2019</td>
      <td>₹ 269.00</td>
    </tr>
    <tr>
      <td>5320</td>
      <td>Simply Text</td>
      <td>4.0</td>
      <td>386</td>
      <td>Developer Tools</td>
      <td>23-01-2014</td>
      <td>₹ 219.00</td>
    </tr>
  </tbody>
</table>
<p>5321 rows × 6 columns</p>
</div>




```python
data.hist(figsize=(10, 4))
plt.show()
```


![png](2020-08-04-Windows_App_Store_files/2020-08-04-Windows_App_Store_4_0.png)



```python
index = data.index
num_of_rows = len(index)
num_of_rows
```




    5321




```python
free = (data['Price'] == 'Free').sum()/num_of_rows*100
free
```




    97.03063333959781




```python
paid = (data['Price'] != 'Free').sum()/num_of_rows*100
paid
```




    2.96936666040218




```python
categories = data['Category'].unique()
categories
```




    array(['Books', 'Business', 'Developer Tools', 'Social',
           'Food and Dining', 'Government and Politics', 'Health and Fitness',
           'Kids and Family', 'Lifestyle', 'Multimedia Design', 'Music',
           'Navigation and Maps', 'News and Weather'], dtype=object)




```python
categories_dict = dict()
for item in categories:
    categories_dict[item] = (data['Category'] == item).sum()
```


```python
categories_dict
```




    {'Books': 707,
     'Business': 679,
     'Developer Tools': 500,
     'Social': 328,
     'Food and Dining': 166,
     'Government and Politics': 70,
     'Health and Fitness': 527,
     'Kids and Family': 345,
     'Lifestyle': 492,
     'Multimedia Design': 117,
     'Music': 753,
     'Navigation and Maps': 154,
     'News and Weather': 483}




```python
categories_list =  list(categories_dict.values())
categories_list
```




    [707, 679, 500, 328, 166, 70, 527, 345, 492, 117, 753, 154, 483]




```python
categories_keys =  list(categories)
categories_keys
```




    ['Books',
     'Business',
     'Developer Tools',
     'Social',
     'Food and Dining',
     'Government and Politics',
     'Health and Fitness',
     'Kids and Family',
     'Lifestyle',
     'Multimedia Design',
     'Music',
     'Navigation and Maps',
     'News and Weather']




```python
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

def func(pct):

    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(categories_list, autopct=lambda pct: func(pct),
                                  textprops=dict(color="w"))

ax.legend(wedges, categories_keys,
          title="Distribution of Categories",
          loc="center",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=20, weight="bold")

ax.set_title("Distribution of  Categories")
plt.show()
```


![png](2020-08-04-Windows_App_Store_files/2020-08-04-Windows_App_Store_13_0.png)



```python
years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
```


```python
years_dict = dict()
for year in years:
    years_dict[year] =(data['Date'].str.contains(year)).sum()
```


```python
years_dict
```




    {'2011': 32,
     '2012': 321,
     '2013': 532,
     '2014': 594,
     '2015': 580,
     '2016': 1013,
     '2017': 794,
     '2018': 796,
     '2019': 468,
     '2020': 186}




```python
plt.pie(years_dict.values(),labels=years_dict.keys(),autopct=lambda pct: func(pct))
plt.title("Year wise Distribution")
```




    Text(0.5, 1.0, 'Year wise Distribution')




![png](2020-08-04-Windows_App_Store_files/2020-08-04-Windows_App_Store_17_1.png)



```python

```
