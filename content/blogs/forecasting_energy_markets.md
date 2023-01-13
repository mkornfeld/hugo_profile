---
title: "Predicting Energy Futures"
date: 2023-01-03T23:29:21+05:30
draft: false
github_link: "https://github.com/gurusabarish/hugo-profile"
author: "Myles Kornfeld"
tags:
  - Forecasting
  - Energy
  - Regression
image: /images/post.jpg
description: ""
toc:
---

# Forecasting Energy Markets

## Introduction

Over the past several years, global events have rocked the energy markets: the Covid-19 pandemic caused the price of oil to crash, while a combination of inflation and the sanctions placed on Russia due to its invasion of Ukraine have caused the price of oil and natural gas to skyrocket. As a result, it is important to be able to forecast energy markets accurately despite global uncertainty.

With this in mind, my project aims at developing a tool to forecast energy markets consistently and accurately in a time-effective manner. This tool can help governments allocate resources efficiently and help investors optimize trading futures. Specifically, by pulling publicly available data, I will train different univariate models on past values to forecast future values.

## Tools
I will be creating these models using Python. In particular, I will be using the pandas and numpy libraries to clean the data, the requests and json libraries to pull and parse the data, and the sklearn library to prepare the data. I will be using a XGBoost regressor, a Support Vector regressor, and a Random Forest regressor to forecast the data.


```python
import requests
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
import math
import datetime

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pylab import rcParams
rcParams['figure.figsize'] = 12,6
```

## Pulling the Data

### U.S. Energy Information Administration
The U.S. Energy Information Administration (EIA) collects and disseminates energy data to the public to promote efficient markets, sound policy making, and public education. The EIA makes its data available through dashbarods available here: https://www.eia.gov/tools/. For people who want to pull data locally, the EIA provides documentation for their API here: https://www.eia.gov/opendata/documentation.php. To pull the data, one needs to register an API key. I blurred mine out, but one can easily be obtained from their website. However, their API has several problems. The first is that when trying to determine which data to pull, the options are often hidden, making querying data tedious. Additionally, the maximum number of points that can be pulled per request is 5000 datapoints. If one wants to pull more data, they would either have to paginate or or filter their data in the API call, as opposed to in a dataframe tool, like Pandas, which would be typically more well known by the user. My first class, LoadQuery, addresses these issues.

The class LoadQuery begins by looping through the different pages available, from which the user can select which data they want while having all options available, and then choose which dataset they want to pull. Next, the user will choose the periodicity of the data, if it comes in multiple formats, as well as the type of data that they want to pull, again, if the data comes in mulitple formats. Once chosen, the program will then paginate through the queries until all data is pulled. The data is then turned into a Pandas dataframe format, which can be accessed through the .df attribute of the LoadQuery object. The dataframe will be indexed by date.

```Python
date_switch = {'annual':'%Y',
                   'monthly':'%Y-%m',
                   'daily':'%Y-%m-%d'}

class LoadQuery:
    def __init__(self, header = "",page = "", key = "",data = None, frequency = None,response = None, df = None, format = None):
        self.header = 'https://api.eia.gov/v2'
        self.page = page ##Which page you want to access. Created by running load_page
        self.key= #XXXXXXXXX
        self.data = data ##The data you want to access with the query. Created by running choose_query
        self.frequency = frequency
        self.response = response
        self.df = df
        self.format = format

    def load_page(self):
        ##When running load_page, this creates
        api_url = self.header+self.page+self.key
        response = requests.get(api_url)#, verify = False)
        response = response.json()
        if 'frequency' not in response['response'].keys():
            print('Choose your id')
            check = []
            for i in response['response']['routes']:
                print("id: ", i['id'], " Name: ", i['name'],'\r')
                check.append(i['id'])
            next_page = input()
            if next_page not in check:
                print('What you typed was not an option')
                self.load_page()
            else:
                self.page+='/'+next_page
                self.load_page()
        api_url = self.header+self.page+self.key
        response = requests.get(api_url)
        self.response = response.json()


    def choose_frequency(self):
        if len(self.response['response']['frequency']) == 1:
            self.frequency = self.response['response']['frequency'][0]['id']
        else:
            print('Choose the frequency')
            check = []
            for i in self.response['response']['frequency']:
                check.append(i['id'])
                print(i['id'])
            frequency = input()
            if frequency in check:
                self.frequency = frequency
            else:
                print('What you typed was not an option')
                self.choose_frequency()

    def choose_data(self):
        if len(self.response['response']['data'].keys()) == 1:
            self.data = list(self.response['response']['data'].keys())[0]
        else:
            print('Choose the data type')
            check = []
            for i in self.response['response']['data'].keys():
                check.append(i)
                print(i)
            data = input()
            if data in check:
                self.data = data
            else:
                print('What you typed was not an option')
                self.choose_data()

    def create_info(self):
        api_url = self.header+self.page+self.key+'&data[]='+self.data+'&frequency='+self.frequency
        response = requests.get(api_url)
        self.response = response.json()    

    def create_every_response(self):
        self.format = date_switch[self.frequency]
        api_url = self.header+self.page+"/data"+self.key+'&data[]='+self.data+'&frequency='+self.frequency
        response = requests.get(api_url)
        data = response.json()
        final = data['response']['data']
        total = data['response']['total']
        offset = 5000
        if total <= 5000:
            self.response = final
        else:
            num_loops = math.floor(total/5000)
            offset = 5000
            for i in range(0,num_loops):
                api_url = self.header+self.page+"/data"+self.key+'&data[]='+self.data+'&frequency='+self.frequency+'&offset='+str(offset)
                response = requests.get(api_url)
                data = response.json()
                final += data['response']['data']
                offset += 5000
                percent = (offset-5000)/total*100
                print("%.2f"%percent,'% of the way done')
            self.response = final

    def get_total_number_of_data_points(self):
        api_url = self.header+self.page+"/data"+self.key+'&data[]='+self.data+'&frequency='+self.frequency
        response = requests.get(api_url)
        response = response.json()
        return response['response']['total']

    def create_df(self):
        api_url = self.header+self.page+"/data"+self.key+'&data[]='+self.data+'&frequency='+self.frequency
        response = requests.get(api_url)
        data = response.json()
        self.df = pd.DataFrame(data['response']['data'])

    def return_url(self):
        url = self.header+self.page+"/data"+self.key+'&data[]='+self.data+'&frequency='+self.frequency
        return url

    def create_data_df(self):
        self.df = pd.DataFrame(self.response)

    def create_index(self):
        self.df['period'] = pd.to_datetime(self.df['period'],format = self.format)
        self.df['time_idx'] = pd.DatetimeIndex(self.df['period'])
        self.df = self.df.set_index('time_idx')
        self.df = self.df.sort_values('time_idx')

    def run_all(self):
        self.load_page()
        self.choose_frequency()
        self.choose_data()
        self.create_info()
        self.create_every_response()
        self.df = pd.DataFrame(self.response).drop_duplicates()
        self.create_index()
        print('Data has been obtained')
```

## Building the Models

To build and determine the best model, I built the ForecastModel class below. A ForecastModel object takes four parameters: the data one wants to forecast which was pulled from the query, the number of past periods to train the models on, the number of periods to forecast, and the name of the column in the dataframe being passed that contains the data.

I perform a GridSearch across a Support Vector regressor, a XGBoost regressor, and a RandomForest regressor. First, I

To build the models, I first call the shift_values() method to shift the values being predicted up a row so that the features in each row will forecast the value for the next day. Next, I call train_models(), which calls the models and performs a GridSearch, trains each model on a 80/20 split, and then performs a forecast by using the datapoints forecasted as the features. I then store the error and the predictions for each model. Afterwards, I align the predictions with the dates they correspond to through update_df().

Finally, I call plot_best_model() to plot with the smallest MAE, while plot_data() plots all three models trained from the GridSearch. Calling find_best_model() works through the above steps and plots the best model. Calling the regerssors attribute will display the final models created for each regressor. Below, I work through two examples.

```Python
params = {
'Support Vector Regressor':{"C": [0.5, 0.7], 'kernel':['rbf', 'poly']},
'RandomForest':{'max_features':[1.0,'sqrt'],'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],'bootstrap':[True,False]},
'XGBoost':{'min_child_weight': [5, 7, 9],'colsample_bytree': [0.2, 0.4, 0.6],'max_depth': [5, 7, 9]}
}

class ForecastModel:
    def __init__(self, df=None, num_shifts = None, num_predict = None, col = None,
                 error = None, test=None, predictions = None, final_df = None, final_date = None, first_date = None, regressors=None):
        self.df = df
        self.col = col
        self.num_shifts = num_shifts
        self.num_predict = num_predict
        self.data = self.shift_values()
        self.error = error
        self.test = test
        self.predictions = predictions
        self.final_df = final_df
        self.final_date = final_date
        self.first_date = first_date
        self.regressors = regressors

    def create_regressors(self):
        regressors = {
            'Support Vector Regressor': [SVR()],
            'RandomForest':[RandomForestRegressor()],
            'XGBoost': [XGBRegressor()]
        }
        self.regressors = regressors

    def shift_values(self):
        df_dummy = self.df[[self.col]].copy()
        for i in range(1,self.num_shifts+1):
            name = self.col + "_shift " + str(i)
            df_dummy[name] = df_dummy[self.col].shift(i)
        df_dummy[self.col] = df_dummy[self.col].shift(-1)
        df_dummy = df_dummy.dropna(how='any',axis=0)
        y = df_dummy[self.col]
        X = df_dummy.drop(self.col,axis=1)
        return X,y

    def train_models(self):
        X,y = self.shift_values()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        X_pred = X_test.copy()
        self.create_regressors()
        for reg in self.regressors:
            print('Training '+reg)
            model = self.regressors[reg][0]
            #model.fit(X_train, y_train)
            grid = GridSearchCV(model, params[reg])
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            self.regressors[reg][0] = model
            for i in range(0,len(X_pred)-1):
                new = model.predict([X_pred[i]])
                if i == 0:
                    predictions = np.array([new[0]])
                else:
                    predictions = np.append(predictions,np.array([new[0]]))
                X_pred[i+1] = np.concatenate([new,X_test[i+1][:-1]])
            error = mean_absolute_error(y_test[:-1], predictions)
            self.regressors[reg].append(error)
            self.regressors[reg].append(predictions)   
            print(reg+' trained')
        self.test = y_test

    def update_df(self):
        first_date = None
        final_date = None
        length = len(self.df[self.df[self.col] == self.test[0]].index)
        for i in range(0,length):
            x = self.df.index.get_loc(self.df[self.df[self.col] == self.test[0]].index[i])
            new = x + len(self.test) - 1
            if self.df.iloc[new]['value'] == self.test[-1]:
                first_date = x
                final_date = new + 1
                break
        self.first_date = first_date
        self.final_date = final_date
        self.final_df = self.df.iloc[first_date:final_date][[self.col]].copy()[:-1]
        self.final_df['test'] = self.test[:-1]
        for reg in self.regressors:
            self.final_df[reg+' preds'] = self.regressors[reg][2]


    def plot_best_model(self):
        left = 500
        right = 100
        min_error = self.regressors['Support Vector Regressor'][1]
        best_model_name = 'Support Vector Regressor'
        for reg in self.regressors:
            if self.regressors[reg][1] < min_error:
                min_error = self.regressors[reg][1]
                best_model_name = reg

        rcParams['figure.figsize'] = 20,10
        plt.plot(self.df[self.col], label = 'Full Dataset', color = 'black')
        plt.plot(self.final_df[best_model_name+' preds'], label = best_model_name+' Predictions', color = 'blue')
        corr = str(round(self.final_df.corr()[best_model_name+' preds']['value'],3))
        min_error_name = ' MAE: %.3f' % min_error
        plt.title(best_model_name+' Predictions r^2: '+corr+min_error_name)
        plt.legend()
        plt.show()

        plt.plot(self.df[self.col][self.first_date - left:self.final_date + right], label = 'Previous', color = 'black')
        plt.plot(self.final_df[best_model_name+' preds'], label = best_model_name+' Predictions', color = 'blue')
        corr = str(round(self.final_df.corr()[reg+' preds']['value'],3))
        plt.title(best_model_name+' Predictions r^2: '+corr+min_error_name)
        plt.legend()
        plt.show()

    def find_best_model(self):
        self.train_models()
        self.update_df()
        self.plot_best_model()

    def plot_data(self):
        left = 500
        right = 100

        for reg in self.regressors:
            min_error = ' MAE: %.3f' % self.regressors[reg][1]
            rcParams['figure.figsize'] = 20,10
            plt.plot(self.df[self.col][self.first_date - left:self.final_date + right], label = 'Previous', color = 'black')
            plt.plot(self.final_df[reg+' preds'], label = reg+' Predictions', color = 'blue')
            corr = str(round(self.final_df.corr()[reg+' preds']['value'],3))
            plt.title(reg+' Predictions r^2: '+corr+min_error)
            plt.legend()
            plt.show()
```

## Forecasting Natural Gas Futures
Below, I walk through the process outlined above to forecast natural gas futures. Specifically, I will be forecasting the Contract 1 futures prices. Because natural gas contracts expire three business days prior to the first calendar day of the delivery month, Contract 1 contains the calendar month following the trade date. Information can be found at this link: https://www.eia.gov/dnav/ng/TblDefs/ng_pri_fut_tbldef2.asp.

Below, I create a LoadQuery object to pull the data that I want to forecast.

``` Python
natural_gas_query = LoadQuery()
natural_gas_query.run_all()
```

| time_idx            | period              | duoarea   | area-name     | product   | product-name   | process   | process-name      | series   | series-description                                       |   value | units   |
|:--------------------|:--------------------|:----------|:--------------|:----------|:---------------|:----------|:------------------|:---------|:---------------------------------------------------------|--------:|:--------|
| 1993-12-20 00:00:00 | 1993-12-20 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.894 | $/MMBTU |
| 1993-12-21 00:00:00 | 1993-12-21 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.83  | $/MMBTU |
| 1993-12-22 00:00:00 | 1993-12-22 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.859 | $/MMBTU |
| 1993-12-23 00:00:00 | 1993-12-23 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.895 | $/MMBTU |
| 1993-12-27 00:00:00 | 1993-12-27 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.965 | $/MMBTU |

<!--
## Paragraph

Xerum, quo qui aut unt expliquam qui dolut labo. Aque venitatiusda cum, voluptionse latur sitiae dolessi aut parist aut dollo enim qui voluptate ma dolestendit peritin re plis aut quas inctum laceat est volestemque commosa as cus endigna tectur, offic to cor sequas etum rerum idem sintibus eiur? Quianimin porecus evelectur, cum que nis nust voloribus ratem aut omnimi, sitatur? Quiatem. Nam, omnis sum am facea corem alique molestrunt et eos evelece arcillit ut aut eos eos nus, sin conecerem erum fuga. Ri oditatquam, ad quibus unda veliamenimin cusam et facea ipsamus es exerum sitate dolores editium rerore eost, temped molorro ratiae volorro te reribus dolorer sperchicium faceata tiustia prat.

Itatur? Quiatae cullecum rem ent aut odis in re eossequodi nonsequ idebis ne sapicia is sinveli squiatum, core et que aut hariosam ex eat.

## Blockquotes

The blockquote element represents content that is quoted from another source, optionally with a citation which must be within a `footer` or `cite` element, and optionally with in-line changes such as annotations and abbreviations.

### Blockquote without attribution

> Tiam, ad mint andaepu dandae nostion secatur sequo quae.
> **Note** that you can use _Markdown syntax_ within a blockquote.

### Blockquote with attribution

> Don't communicate by sharing memory, share memory by communicating.</p>
> — <cite>Rob Pike[^1]</cite>

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.

## Tables

Tables aren't part of the core Markdown spec, but Hugo supports supports them out-of-the-box.

| Name  | Age |
| ----- | --- |
| Bob   | 27  |
| Alice | 23  |

### Inline Markdown within tables

| Inline&nbsp;&nbsp;&nbsp; | Markdown&nbsp;&nbsp;&nbsp; | In&nbsp;&nbsp;&nbsp;                | Table  |
| ------------------------ | -------------------------- | ----------------------------------- | ------ |
| _italics_                | **bold**                   | ~~strikethrough~~&nbsp;&nbsp;&nbsp; | `code` |

## Code Blocks

### Code block with backticks

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Example HTML5 Document</title>
  </head>
  <body>
    <p>Test</p>
  </body>
</html>
```

### Code block indented with four spaces

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Example HTML5 Document</title>
    </head>
    <body>
      <p>Test</p>
    </body>
    </html>

### Code block with Hugo's internal highlight shortcode

{{< highlight html >}}

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Example HTML5 Document</title>
</head>
<body>
  <p>Test</p>
</body>
</html>
{{< /highlight >}}

## List Types

### Ordered List

1. First item
2. Second item
3. Third item

### Unordered List

- List item
- Another item
- And another item

### Nested list

- Item
  1. First Sub-item
  2. Second Sub-item

## Headings

The following HTML `<h1>`—`<h6>` elements represent six levels of section headings. `<h1>` is the highest section level while `<h6>` is the lowest.

# H1

## H2

### H3

#### H4

##### H5

###### H6

## Other Elements — abbr, sub, sup, kbd, mark

<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd><kbd>CTRL</kbd>+<kbd>ALT</kbd>+<kbd>Delete</kbd></kbd> to end the session.

Most <mark>salamanders</mark> are nocturnal, and hunt for insects, worms, and other small creatures. -->
