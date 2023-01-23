---
title: "Negation Sentiment with NLP"
date: 2023-01-18T23:29:21+05:30
draft: false
github_link: "https://github.com/mkornfeld/negation_nlp"
author: "Myles Kornfeld"
tags:
  - NLP
  - Sentiment Analysis
  - Machine Learning
image: /images/nat_gas_fut_1f.png
description: ""
toc:
---

# Analyzing Negation Sentiments with NLP

## Introduction

Throughout various industries, Natural Language Processing (NLP) models can be used to solve many problems: organizations can use NLP analysis to understand employee satisfaction; companies can use NLP to understand consumer sentiment for a project; investments firms can use NLP to understand the market's sentiment around a specific stock or company. By performing NLP, organizations can gain quantitative insights from qualitative data.

One problem that currently faces the field is analyzing sentiment when negations are present. For example, the sentence "Don't do that again!" can be interpreted in a positive context (if, for example, a person pranked another person) or in a negative context (if a person caused harm to another person). Denials, imperatives, questions, and rejections are all forms have negations that can decrease accuracy. Ultimately, creating a model that can determine sentiment with negations present is important as organizations can use this model alongside other models to gain more accurate insights.

With this in mind, my project aims at developing a model that analyzes sentiment analysis with negations present. By analyzing tweets that specifically contain any or all of the words "no", "not", "never", and "didn't", I attempt to build a machine learning model that can accurately classify these sentiments as either positive or negative.

## Tools
I will be creating these models using Python. In particular, I will be using the ```pandas``` and ```numpy``` libraries to clean the data, the ```sklearn``` library to prepare the data, and ```keras``` to build and train the models. I also use the ```nltk``` library to provide stopwords.


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('brown')
nltk.download('stopwords')
stop = stopwords.words('english')
stop = set(stopwords.words('english')) - {'no', 'not','never',"don't",'never','nor'}
```
When preparing data for NLP, commonly used words, such as "our", "such", and "out", are typically removed from datasets as these are words that do not provide the true sentiment regarding a statement. For this project, I am making sure to include the negations because I want my model to be able to recognize those words, as shown in redefining the set above.

## Data Collection and Preparation
I collected data from [Sentiment140](http://help.sentiment140.com/home), a group run by former Computer Science graduate students at Stanford. They offer machine learning sentiment analysis services on tweets, and offer their [training and testing](http://help.sentiment140.com/for-students/) datasets free to use. The datasets contain 6 fields:
1. The sentiment of the tweet, where 0 is negative, 2 is neutral, and 4 is positive
2. The id of the tweet
3. The date of the tweet
4. The query of the tweet
5. The twitter user who created the tweet
6. The text contained in the tweet

First I load the data, and then run value counts to see how much of each sentiment each dataset contains:
```Python
df_train = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='latin-1')
df_test = pd.read_csv('testdata.manual.2009.06.14.csv',encoding='latin-1')
df_train['sentiment'].value_counts().reset_index()
```
|    |   index |   sentiment |
|---:|--------:|------------:|
|  0 |       0 |      800000 |
|  1 |       4 |      800000 |

```Python
df_test['sentiment'].value_counts().reset_index()
```
|    |   index |   sentiment |
|---:|--------:|------------:|
|  0 |       4 |         182 |
|  1 |       0 |         177 |
|  2 |       2 |         139 |

Looking at the datasets, we see that the training datasets have a large number of only negative and positive sentiments and no neutral statements, while the testing datasets have a small number of negative and positive sentiments, and a small number of neutral statements. Because the training dataset does not contain any neutral sentiment tweets nor does the testing dataset contain large numbers of validation data, I decide to drop the neutral tweets from the dataset and combine the two datasets. When building my models, I run a 80/20 split on the training/testing data so that the size of the validation dataset increases.

```python
df_test = df_test[df_test['sentiment']!=2]
df_concat = pd.concat([df_train,df_test],axis=0)
df_concat.head()
```
|    |   sentiment |    user_id | date                         | query    | user            | text                                                                                                                |
|---:|------------:|-----------:|:-----------------------------|:---------|:----------------|:--------------------------------------------------------------------------------------------------------------------|
|  0 |           0 | 1467810369 | Mon Apr 06 22:19:45 PDT 2009 | NO_QUERY | _TheSpecialOne_ | @switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D |
|  1 |           0 | 1467810672 | Mon Apr 06 22:19:49 PDT 2009 | NO_QUERY | scotthamilton   | is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!     |
|  2 |           0 | 1467810917 | Mon Apr 06 22:19:53 PDT 2009 | NO_QUERY | mattycus        | @Kenichan I dived many times for the ball. Managed to save 50%  The rest go out of bounds                           |
|  3 |           0 | 1467811184 | Mon Apr 06 22:19:57 PDT 2009 | NO_QUERY | ElleCTF         | my whole body feels itchy and like its on fire                                                                      |
|  4 |           0 | 1467811193 | Mon Apr 06 22:19:57 PDT 2009 | NO_QUERY | Karoli          | @nationwideclass no, it's not behaving at all. i'm mad. why am i here? because I can't see you all over there.      |

Once I combine the datasets, I want to clean them up. First, I make all the letters lowercase so all the words are homogenous. Next, I get rid of twitter usernames and links by filtering out words that contain an "@" and "http", respectively, as those words do not contain sentiment. I also replace 4 to 1 in the sentiment column as it's generally good practice.

```Python
df_concat['text'] = df_concat['text'].str.lower()
df_concat['text'] = df_concat['text'].apply(lambda x: ' '.join([word for word in x.split() if '@' not in word]))
df_concat['text'] = df_concat['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df_concat['text'] = df_concat['text'].apply(lambda x: ' '.join([word for word in x.split() if 'http' not in word]))
df_concat['text'] = df_concat['text'].str.replace(r'[^\w\s]','',regex=True)
df_concat['sentiment'] = df_concat['sentiment'].replace(4,1)
df_concat.head()
```
|    |   sentiment |    user_id | date                         | query    | user            | text                                                                       |
|---:|------------:|-----------:|:-----------------------------|:---------|:----------------|:---------------------------------------------------------------------------|
|  0 |           0 | 1467810369 | Mon Apr 06 22:19:45 PDT 2009 | NO_QUERY | _TheSpecialOne_ | awww thats bummer shoulda got david carr third day                         |
|  1 |           0 | 1467810672 | Mon Apr 06 22:19:49 PDT 2009 | NO_QUERY | scotthamilton   | upset cant update facebook texting might cry result school today also blah |
|  2 |           0 | 1467810917 | Mon Apr 06 22:19:53 PDT 2009 | NO_QUERY | mattycus        | dived many times ball managed save 50 rest go bounds                       |
|  3 |           0 | 1467811184 | Mon Apr 06 22:19:57 PDT 2009 | NO_QUERY | ElleCTF         | whole body feels itchy like fire                                           |
|  4 |           0 | 1467811193 | Mon Apr 06 22:19:57 PDT 2009 | NO_QUERY | Karoli          | no not behaving im mad cant see                                            |

Now, because I removed the apostrophes in each word, I filter to only contain tweets that contain the words 'no', 'not', 'never', 'never', or 'nor'.

```Python
df_neg = df_concat[df_concat['text'].str.contains(r"no|not|never|didnt|dont|nor",regex=True)]
df_neg = df_neg.reset_index()
df_neg.head()
```
|    |   index |   sentiment |    user_id | date                         | query    | user            | text                                                           |
|---:|--------:|------------:|-----------:|:-----------------------------|:---------|:----------------|:---------------------------------------------------------------|
|  0 |       4 |           0 | 1467811193 | Mon Apr 06 22:19:57 PDT 2009 | NO_QUERY | Karoli          | no not behaving im mad cant see                                |
|  1 |       5 |           0 | 1467811372 | Mon Apr 06 22:20:00 PDT 2009 | NO_QUERY | joy_wolf        | not whole crew                                                 |
|  2 |       7 |           0 | 1467811594 | Mon Apr 06 22:20:03 PDT 2009 | NO_QUERY | coZZ            | hey long time no see yes rains bit bit lol im fine thanks hows |
|  3 |       8 |           0 | 1467811795 | Mon Apr 06 22:20:05 PDT 2009 | NO_QUERY | 2Hood4Hollywood | nope                                                           |
|  4 |      10 |           0 | 1467812416 | Mon Apr 06 22:20:16 PDT 2009 | NO_QUERY | erinx3leannexo  | spring break plain city snowing                                |


### Class Imbalance
When performing a classification analysis, it is important to make sure that the classes you are differentiating are of similar magnitude, i.e. balanced. I will visually check this by plotting the number of negative and positive tweets below.

```Python
x_count = list(df_neg['sentiment'].unique())
y_count = list(df_neg['sentiment'].value_counts())
fig = plt.figure(figsize = (12, 8))
 
# creating the bar plot
plt.bar(x_count, y_count, color ='maroon',
        width = .8, tick_label=[0,1])
 
plt.xlabel("Sentiments")
plt.ylabel("Value Counts")
plt.title("Number of Each Sentiment")
plt.show()
```

<p align="center">
  <img src="/images/sentiment_count.png" width="700">
</p>

Clearly, the number of negative and positive tweets with negation are on the same order of magnitude, and so the classes are balanced.

### Identifying Outliers

## Building the Models

To build and determine the best model, I built the ForecastModel class below. A ForecastModel object takes four parameters: the data one wants to forecast which was pulled from the query, the number of past periods to train the models on, the number of periods to forecast, and the name of the column in the dataframe being passed that contains the data.

I perform a GridSearch across a Support Vector regressor, a XGBoost regressor, and a RandomForest regressor. First, I

To build the models, I first call the ```shift_values()``` method to shift the values being predicted up a row so that the features in each row will forecast the value for the next day. Next, I call ```train_models()```, which calls the models and performs a GridSearch, trains each model on a 80/20 split, and then performs a forecast by using the datapoints forecasted as the features. I then store the error and the predictions for each model. Afterwards, I align the predictions with the dates they correspond to through ```update_df()```.

Finally, I call ```plot_best_model()``` to plot with the smallest MAE, while plot_data() plots all three models trained from the GridSearch. Calling ```find_best_model()``` works through the above steps and plots the best model. Calling the regressors attribute will display the final models created for each regressor. Below, I work through two examples.

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


```
Choose your id
id:  coal  Name:  Coal
id:  crude-oil-imports  Name:  Crude Oil Imports
id:  electricity  Name:  Electricity
id:  international  Name:  International
id:  natural-gas  Name:  Natural Gas
id:  nuclear-outages  Name:  Nuclear Outages
id:  petroleum  Name:  Petroleum
id:  seds  Name:  State Energy Data System (SEDS)
id:  steo  Name:  Short Term Energy Outlook
id:  densified-biomass  Name:  Densified Biomass
id:  total-energy  Name:  Total Energy
id:  aeo  Name:  Annual Energy Outlook
id:  ieo  Name:  International Energy Outlook
id:  co2-emissions  Name:  State CO2 Emissions
natural-gas
Choose your id
id:  sum  Name:  Summary
id:  pri  Name:  Prices
id:  enr  Name:  Exploration and Reserves
id:  prod  Name:  Production
id:  move  Name:  Imports and Exports/Pipelines
id:  stor  Name:  Storage
id:  cons  Name:  Consumption / End Use
pri
Choose your id
id:  sum  Name:  Natural Gas Prices
id:  fut  Name:  Natural Gas Spot and Futures Prices (NYMEX)
id:  rescom  Name:  Average Price of Natural Gas Delivered to Residential and Commercial Consumers by Local Distribution and Marketers in Selected States
fut
Choose the frequency
weekly
monthly
daily
annual
daily
11.82 % of the way done
23.64 % of the way done
35.46 % of the way done
47.28 % of the way done
59.10 % of the way done
70.92 % of the way done
82.74 % of the way done
94.56 % of the way done
Data has been obtained
```

Now that the data has been obtained, I call the dataframe to see the different features and how to filter for Contract 1.

```Python
natural_gas_query.df['series-description'].unique()
```

```
array(['Natural Gas Futures Contract 4 (Dollars per Million Btu)',
       'Natural Gas Futures Contract 2 (Dollars per Million Btu)',
       'Natural Gas Futures Contract 1 (Dollars per Million Btu)',
       'Natural Gas Futures Contract 3 (Dollars per Million Btu)',
       'Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'],
      dtype=object)
```

I will filter on the series description 'Natural Gas Futures Contract 1 (Dollars per Million Btu)'.

```Python
series_desc = 'Natural Gas Futures Contract 1 (Dollars per Million Btu)'
natural_gas_data = natural_gas_query.df[(natural_gas_query.df['series-description'] == series_desc)]
natural_gas_data = natural_gas_data.dropna()
natural_gas_data.head()
```


| time_idx            | period              | duoarea   | area-name     | product   | product-name   | process   | process-name      | series   | series-description                                       |   value | units   |
|:--------------------|:--------------------|:----------|:--------------|:----------|:---------------|:----------|:------------------|:---------|:---------------------------------------------------------|--------:|:--------|
| 1993-12-20 00:00:00 | 1993-12-20 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.894 | $/MMBTU |
| 1993-12-21 00:00:00 | 1993-12-21 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.83  | $/MMBTU |
| 1993-12-22 00:00:00 | 1993-12-22 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.859 | $/MMBTU |
| 1993-12-23 00:00:00 | 1993-12-23 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.895 | $/MMBTU |
| 1993-12-27 00:00:00 | 1993-12-27 00:00:00 | Y35NY     | NEW YORK CITY | EPG0      | Natural Gas    | PE4       | Future Contract 4 | RNGC4    | Natural Gas Futures Contract 4 (Dollars per Million Btu) |   1.965 | $/MMBTU |


I create the model by passing the dataframe ```natural_gas_data```, train the model by having the previous 50 days predict the next day, forecast the next 150 days, and use the data in the column "value".

After training the models, the best model is graphed below. The Support Vector regressor was the best model, and had an r<sup>2</sup> value of 0.982 with the testing data and a mean average error of 0.190.

```Python
natural_gas_forecast = ForecastModel(df = natural_gas_data, num_shifts = 50, num_predict = 150, col = 'value')
natural_gas_forecast.find_best_model()
```
```
Training Support Vector Regressor
Support Vector Regressor trained
Training RandomForest
RandomForest trained
Training XGBoost
XGBoost trained
```

<!-- ![title](/images/nat_gas_fut_1.png)
![title](/images/nat_gas_fut_2.png) -->
  <p align="center">
    <img src="/images/nat_gas_fut_1f.png" width="700">
  </p>

  <p align="center">
    <img src="/images/nat_gas_fut_2f.png" width="700">
  </p>

Below I plot the best predictions for each of the different regressors. In addition to the Suport Vector regressor, the XGBoost regressor closely predicted the test data, with an r<sup>2</sup> of 0.979 and an MAE of 0.218. The RandomForest regressor, however, clearly did not capture the trends in the data.

```Python
natural_gas_forecast.plot_data()
```

<p align="center">
  <img src="/images/nat_gas_fut_1f.png" width="700">
</p>
<p align="center">
  <img src="/images/nat_gas_fut_rff.png" width="700">
</p>
<p align="center">
  <img src="/images/nat_gas_fut_xgboostf.png" width="700">
</p>


## Forecasting Petroleum Futures
I'll now run through the same process by forecasting New York Harbor No. 2 Heating Oil Future Contract 1.

```Python
petroleum_forecast = LoadQuery()
petroleum_forecast.run_all()
```

```
Choose your id
id:  coal  Name:  Coal
id:  crude-oil-imports  Name:  Crude Oil Imports
id:  electricity  Name:  Electricity
id:  international  Name:  International
id:  natural-gas  Name:  Natural Gas
id:  nuclear-outages  Name:  Nuclear Outages
id:  petroleum  Name:  Petroleum
id:  seds  Name:  State Energy Data System (SEDS)
id:  steo  Name:  Short Term Energy Outlook
id:  densified-biomass  Name:  Densified Biomass
id:  total-energy  Name:  Total Energy
id:  aeo  Name:  Annual Energy Outlook
id:  ieo  Name:  International Energy Outlook
id:  co2-emissions  Name:  State CO2 Emissions
petroleum
Choose your id
id:  sum  Name:  Summary
id:  pri  Name:  Prices
id:  crd  Name:  Crude Reserves and Production
id:  pnp  Name:  Refining and Processing
id:  move  Name:  Imports/Exports and Movements
id:  stoc  Name:  Stocks
id:  cons  Name:  Consumption/Sales
pri
Choose your id
id:  gnd  Name:  Weekly Retail Gasoline and Diesel Prices
id:  spt  Name:  Spot Prices
id:  fut  Name:  NYMEX Futures Prices
id:  wfr  Name:  Weekly Heating Oil and Propane Prices (October - March)
id:  refmg  Name:  Refiner Gasoline Prices by Grade and Sales Type
id:  refmg2  Name:  U.S. Refiner Gasoline Prices by Formulation, Grade, Sales Type
id:  refoth  Name:  Refiner Petroleum Product Prices by Sales Type
id:  allmg  Name:  Gasoline Prices by Formulation, Grade, Sales Type
id:  dist  Name:  No. 2 Distillate Prices by Sales Type
id:  prop  Name:  Propane (Consumer Grade) Prices by Sales Type
id:  resid  Name:  Residual Fuel Oil Prices by Sales Type
id:  dfp1  Name:  Domestic Crude Oil First Purchase Prices by Area
id:  dfp2  Name:  Domestic Crude Oil First Purchase Prices for Selected Crude Streams
id:  dfp3  Name:  Domestic Crude Oil First Purchase Prices by API Gravity
id:  rac2  Name:  Refiner Acquisition Cost of Crude Oil
id:  imc1  Name:  F.O.B. Costs of Imported Crude Oil by Area
id:  imc2  Name:  F.O.B. Costs of Imported Crude Oil for Selected Crude Streams
id:  imc3  Name:  F.O.B. Costs of Imported Crude Oil by API Gravity
id:  land1  Name:  Landed Costs of Imported Crude by Area
id:  land2  Name:  Landed Costs of Imported Crude for Selected Crude Streams
id:  land3  Name:  Landed Costs of Imported Crude by API Gravity
id:  ipct  Name:  Percentages of Total Imported Crude Oil by API Gravity
fut
Choose the frequency
weekly
daily
monthly
annual
daily
3.68 % of the way done
7.36 % of the way done
11.05 % of the way done
14.73 % of the way done
18.41 % of the way done
22.09 % of the way done
25.78 % of the way done
29.46 % of the way done
33.14 % of the way done
36.82 % of the way done
40.50 % of the way done
44.19 % of the way done
47.87 % of the way done
51.55 % of the way done
55.23 % of the way done
58.91 % of the way done
62.60 % of the way done
66.28 % of the way done
69.96 % of the way done
73.64 % of the way done
77.33 % of the way done
81.01 % of the way done
84.69 % of the way done
88.37 % of the way done
92.05 % of the way done
95.74 % of the way done
99.42 % of the way done
Data has been obtained
```


```Python
petroleum_forecast.df.head()
```

| time_idx            | period              | duoarea   | area-name     | product   | product-name                | process   | process-name      | series                  | series-description                                                       |   value | units   |
|:--------------------|:--------------------|:----------|:--------------|:----------|:----------------------------|:----------|:------------------|:------------------------|:-------------------------------------------------------------------------|--------:|:--------|
| 1980-01-02 00:00:00 | 1980-01-02 00:00:00 | Y35NY     | NEW YORK CITY | EPD2F     | No 2 Fuel Oil / Heating Oil | PE1       | Future Contract 1 | EER_EPD2F_PE1_Y35NY_DPG | New York Harbor No. 2 Heating Oil Future Contract 1 (Dollars per Gallon) |   0.821 | $/GAL   |
| 1980-01-02 00:00:00 | 1980-01-02 00:00:00 | Y35NY     | NEW YORK CITY | EPD2F     | No 2 Fuel Oil / Heating Oil | PE3       | Future Contract 3 | EER_EPD2F_PE3_Y35NY_DPG | New York Harbor No. 2 Heating Oil Future Contract 3 (Dollars per Gallon) |   0.89  | $/GAL   |
| 1980-01-03 00:00:00 | 1980-01-03 00:00:00 | Y35NY     | NEW YORK CITY | EPD2F     | No 2 Fuel Oil / Heating Oil | PE1       | Future Contract 1 | EER_EPD2F_PE1_Y35NY_DPG | New York Harbor No. 2 Heating Oil Future Contract 1 (Dollars per Gallon) |   0.827 | $/GAL   || 1980-01-03 00:00:00 | 1980-01-03 00:00:00 | Y35NY     | NEW YORK CITY | EPD2F     | No 2 Fuel Oil / Heating Oil | PE3       | Future Contract 3 | EER_EPD2F_PE3_Y35NY_DPG | New York Harbor No. 2 Heating Oil Future Contract 3 (Dollars per Gallon) |   0.866 | $/GAL   |
| 1980-01-04 00:00:00 | 1980-01-04 00:00:00 | Y35NY     | NEW YORK CITY | EPD2F     | No 2 Fuel Oil / Heating Oil | PE3       | Future Contract 3 | EER_EPD2F_PE3_Y35NY_DPG | New York Harbor No. 2 Heating Oil Future Contract 3 (Dollars per Gallon) |   0.88  | $/GAL   |

```Python
petroleum_forecast.df['series-description'].unique()
```

```
array(['New York Harbor No. 2 Heating Oil Future Contract 3 (Dollars per Gallon)',
       'New York Harbor No. 2 Heating Oil Future Contract 1 (Dollars per Gallon)',
       'Cushing, OK Crude Oil Future Contract 3 (Dollars per Barrel)',
       'Cushing, OK Crude Oil Future Contract 1 (Dollars per Barrel)',
       'New York Harbor Regular Gasoline Future Contract 3 (Dollars per Gallon)',
       'New York Harbor Regular Gasoline Future Contract 1 (Dollars per Gallon)',
       'Cushing, OK Crude Oil Future Contract 2 (Dollars per Barrel)',
       'Cushing, OK Crude Oil Future Contract 4 (Dollars per Barrel)',
       'Mont Belvieu, Tx Propane Future Contract 1 (Dollars per Gallon)',
       'Mont Belvieu, Tx Propane Future Contract 4 (Dollars per Gallon)',
       'New York Harbor No. 2 Heating Oil Future Contract 4 (Dollars per Gallon)',
       'New York Harbor Regular Gasoline Future Contract 2 (Dollars per Gallon)',
       'New York Harbor Regular Gasoline Future Contract 4 (Dollars per Gallon)',
       'New York Harbor No. 2 Heating Oil Future Contract 2 (Dollars per Gallon)',
       'Mont Belvieu, Tx Propane Future Contract 2 (Dollars per Gallon)',
       'Mont Belvieu, Tx Propane Future Contract 3 (Dollars per Gallon)',
       'New York Harbor Reformulated RBOB Regular Gasoline Future Contract 2 (Dollars per Gallon)',
       'New York Harbor Reformulated RBOB Regular Gasoline Future Contract 1 (Dollars per Gallon)',
       'New York Harbor Reformulated RBOB Regular Gasoline Future Contract 4 (Dollars per Gallon)',
       'New York Harbor Reformulated RBOB Regular Gasoline Future Contract 3 (Dollars per Gallon)'],
      dtype=object)
```

```Python
petreoleum_forecast_data.df = petroleum_forecast.df[petroleum_forecast.df['series-description'] == 'New York Harbor No. 2 Heating Oil Future Contract 1 (Dollars per Gallon)']
petreoleum_forecast_data.df = petreoleum_forecast_data.df.dropna()
```

After training the different models, this time the XGBoost regressor had the smallest MAE at 0.088 and the largest r<sup>2</sup> at 0.982 for the testing data.

```Python
petroleum_forecast_model = ForecastModel(df = petreoleum_forecast_data.df, num_shifts = 50, num_predict = 150, col = 'value')
petroleum_forecast_model.find_best_model()
```

```
Training Support Vector Regressor
Support Vector Regressor trained
Training RandomForest
RandomForest trained
Training XGBoost
XGBoost trained
```
<p align="center">
  <img src="/images/pet_fut_11.png" width="700">
</p>
<p align="center">
  <img src="/images/pet_fut_zoom.png" width="700">
</p>


Looking at the rest of the models, we see that the Support Vector regressor also predicted the data very well, while the RandomForest regressor was unable to capture the trend.

```Python
petroleum_forecast_model.plot_data()
```


<p align="center">
  <img src="/images/pet_fut_svr.png" width="700">
</p>
<p align="center">
  <img src="/images/pet_fut_rf.png" width="700">
</p>
<p align="center">
  <img src="/images/pet_fut_zoom.png" width="700">
</p>


## Conclusions and Next Steps

Taking a look at the natural gas and petroleum futures predictions, we are able to see that the models captured the volatility due to the Covid-19 pandemic and the invasion of Ukraine in their forecasting. As a result, it seems that univariate time-series models can be useful and cost-effective in forecasting markets.

I believe that there are several ways to improve this project. First, when testing the models across other datasets, the models were occasionally unable to capture seasonality in the data. As a result, when data appears more seasonal, performing a seasonality decomposition could increase model performance. Secondly, there were some datasets where models had more difficulty making predictions. Creating a multivariate model that incorporates features such as GDP, local and international conflicts, different weather events, and location could improve model performance as well.

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
