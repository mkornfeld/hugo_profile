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
image: /images/statement_len.png
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
Another step in cleaning the data is to eliminate outliers. If a tweet is too small, then it will not contain enough context to tell us how the negations affect the sentiment. On the other hand, if a tweet is too long, the model can pick up trends that fall outside of how the negations impact the sentiment. Below, I plot the length of each tweet against how frequently that length tweet appears.

<p align="center">
  <img src="/images/statement_len.png" width="700">
</p>

Here, I describe some statistics about the data.

|    | index   |         text |
|---:|:--------|-------------:|
|  0 | count   | 391092       |
|  1 | mean    |      9.50027 |
|  2 | std     |      4.04891 |
|  3 | min     |      1       |
|  4 | 25%     |      6       |
|  5 | 50%     |      9       |
|  6 | 75%     |     13       |
|  7 | max     |     29       |



## Building the Models

To build and determine the best model, I built the ForecastModel class below. A ForecastModel object takes four parameters: the data one wants to forecast which was pulled from the query, the number of past periods to train the models=
```
