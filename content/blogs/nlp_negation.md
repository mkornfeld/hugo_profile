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

```Python 
df_neg['text'].apply(lambda n: len(n.split())).describe().reset_index()
```

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


We see that the max length is 29 words, the minimum is 1 word, the average is 9.5 words, the standard deviation is 4.05 words, and the data has a skew of 0.318. To account for the skew and remove outliers, we will make the minimum number of words required in a tweet to be 3 and the maximum number of words required to be 26, which results in a skew of right skewed with a skew of -0.0312.

```Python
df_gauss = df_neg[(df_neg['text'].apply(lambda n: len(n.split()))>=3) &
       (df_neg['text'].apply(lambda n: len(n.split()))<=26)]
df_gauss['text'].apply(lambda n: len(n.split())).value_counts().reset_index().describe()
```

|       |    index |    text |
|:------|---------:|--------:|
| count | 24       |    24   |
| mean  | 14.5     | 15929.7 |
| std   |  7.07107 | 13442   |
| min   |  3       |    12   |
| 25%   |  8.75    |   993   |
| 50%   | 14.5     | 16573.5 |
| 75%   | 20.25    | 29967   |
| max   | 26       | 32067   |

```Python
x_len_new = list(df_gauss['text'].apply(lambda n: len(n.split())).value_counts().reset_index()['index'])
y_len_new = list(df_gauss['text'].apply(lambda n: len(n.split())).value_counts().reset_index()['text'])
fig = plt.figure(figsize = (12, 8))
 
plt.bar(x_len_new, y_len_new, color ='maroon',
        width = 1)
 
plt.xlabel("Number of Words")
plt.ylabel("Value Counts")
plt.title("Length of Each Tweet")
plt.show()
```

<p align="center">
  <img src="/images/counts_no_outliers.png" width="700">
</p>

### Tokenizing the Data

Now that the data has been cleaned, we need to tokenize it. I tokenize the data with a maximum vocabulary size of 5000, have the max length be the length of the longest tweet, which I restricted to 26, use post padding, and finally join the padded sequences to `df_gauss` and drop the rest of the columns.

```Python
vocab_size = 5000
max_length = df_gauss['text'].apply(lambda n: len(n.split())).max()

myTokenizer = Tokenizer(num_words=vocab_size)
myTokenizer.fit_on_texts(df_concat['text'])
sequences = myTokenizer.texts_to_sequences(df_gauss['text'])
padded = pad_sequences(sequences, maxlen=max_length, padding="post")

df_data = df_gauss.join(pd.DataFrame(padded))
df_data = df_data.drop(['index','user_id','date','query','user','text'],axis=1)
df_data.head()

```

|    |   sentiment |    0 |   1 |    2 |   3 |   4 |   5 |    6 |   7 |    8 |   9 |   10 |   11 |   12 |   13 |   14 |   15 |   16 |   17 |   18 |   19 |   20 |   21 |   22 |   23 |   24 |   25 |
|---:|------------:|-----:|----:|-----:|----:|----:|----:|-----:|----:|-----:|----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
|  0 |           0 |    7 |   2 |  303 |   1 | 505 | 182 |   14 |  23 |  213 |   0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
|  1 |           0 |    2 | 330 | 2143 |   0 |   0 |   0 |    0 |   0 |    0 |   0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
|  2 |           0 |   83 |  93 |   16 |   7 |  23 |  86 | 2481 | 165 | 2454 | 165 |   18 |    1 |  434 |   32 |  715 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
|  4 |           0 | 4625 | 764 |  375 |  88 | 241 | 311 |    0 |   0 |    0 |   0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
|  5 |           0 | 2178 |  94 | 3919 |   2 |  22 |  67 | 4230 |   0 |    0 |   0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |

## Building the Models

### Splitting the Data
When training the models, one problem that I ran into was the length of time it took to train the models: because there were `362116` datapoints, it took a long time to fully train the machine learning model. As a result, I sampled the data dowwn to `10%` of that size by randomly choosing a subset of the data. I think split the data up with an `80/20` split.

```Python
df_sample = df_data.sample(frac=0.1,random_state=42)
y = df_sample['sentiment']
X = df_sample.drop('sentiment',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
```

### Determining the Learning Rate
When training neural nets, the learning rate determines the speed at which the model trains. If a learning rate is set too fast, then the model will miss the minimum of the loss function. If a learning rate is set too slow, then the loss function model never reaches the minimum. In order to find the learning rate for each model, I defined `determine_lr()`, which takes in a model, an epoch number, and a guess for the learning rate. The function slightly increases the learning rate with each epoch, and outputs the model's training history.

```Python
def determine_lr(model, guess, epoch_num):
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: guess * 10**(epoch/15))
    optimizer = tf.keras.optimizers.SGD(lr=guess, momentum=0.9) #Stochastic gradient descent optimizer
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epoch_num, callbacks = lr_schedule)
    return history
```

The first model I will train will be a model with 1 Bidirectional LSTM layer that has a dropout of 0.3, an embedding dimension of 16, and a dense layer before the output layer.

```Python
embedding_dim = 16
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, dropout = 0.3, return_sequences = True)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

history1 = determine_lr(model1, 'Model 1',1e-6,100)
```

```
Epoch 1/100
478/478 [==============================] - 19s 36ms/step - loss: 0.6942 - accuracy: 0.4127 - lr: 1.0000e-06
Epoch 2/100
478/478 [==============================] - 17s 36ms/step - loss: 0.6939 - accuracy: 0.4270 - lr: 1.1659e-06
Epoch 3/100
478/478 [==============================] - 17s 35ms/step - loss: 0.6935 - accuracy: 0.4469 - lr: 1.3594e-06
Epoch 4/100
478/478 [==============================] - 17s 35ms/step - loss: 0.6930 - accuracy: 0.4874 - lr: 1.5849e-06
Epoch 5/100
478/478 [==============================] - 16s 33ms/step - loss: 0.6918 - accuracy: 0.5928 - lr: 1.8478e-06
Epoch 6/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6901 - accuracy: 0.6245 - lr: 2.1544e-06
Epoch 7/100
478/478 [==============================] - 16s 33ms/step - loss: 0.6884 - accuracy: 0.6255 - lr: 2.5119e-06
Epoch 8/100
478/478 [==============================] - 16s 33ms/step - loss: 0.6865 - accuracy: 0.6257 - lr: 2.9286e-06
Epoch 9/100
478/478 [==============================] - 16s 33ms/step - loss: 0.6844 - accuracy: 0.6257 - lr: 3.4145e-06
Epoch 10/100
478/478 [==============================] - 16s 33ms/step - loss: 0.6819 - accuracy: 0.6257 - lr: 3.9811e-06
Epoch 11/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6792 - accuracy: 0.6257 - lr: 4.6416e-06
Epoch 12/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6760 - accuracy: 0.6257 - lr: 5.4117e-06
Epoch 13/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6727 - accuracy: 0.6257 - lr: 6.3096e-06
Epoch 14/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6692 - accuracy: 0.6257 - lr: 7.3564e-06
Epoch 15/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6661 - accuracy: 0.6257 - lr: 8.5770e-06
Epoch 16/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6638 - accuracy: 0.6257 - lr: 1.0000e-05
Epoch 17/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6623 - accuracy: 0.6257 - lr: 1.1659e-05
Epoch 18/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6615 - accuracy: 0.6257 - lr: 1.3594e-05
Epoch 19/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6610 - accuracy: 0.6257 - lr: 1.5849e-05
Epoch 20/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6605 - accuracy: 0.6257 - lr: 1.8478e-05
Epoch 21/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6598 - accuracy: 0.6257 - lr: 2.1544e-05
Epoch 22/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6589 - accuracy: 0.6257 - lr: 2.5119e-05
Epoch 23/100
478/478 [==============================] - 16s 32ms/step - loss: 0.6570 - accuracy: 0.6257 - lr: 2.9286e-05
Epoch 24/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6524 - accuracy: 0.6257 - lr: 3.4145e-05
Epoch 25/100
478/478 [==============================] - 16s 33ms/step - loss: 0.6333 - accuracy: 0.6274 - lr: 3.9811e-05
Epoch 26/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6079 - accuracy: 0.6595 - lr: 4.6416e-05
Epoch 27/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5895 - accuracy: 0.6813 - lr: 5.4117e-05
Epoch 28/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5705 - accuracy: 0.6984 - lr: 6.3096e-05
Epoch 29/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5554 - accuracy: 0.7135 - lr: 7.3564e-05
Epoch 30/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5360 - accuracy: 0.7312 - lr: 8.5770e-05
Epoch 31/100
478/478 [==============================] - 16s 32ms/step - loss: 0.5188 - accuracy: 0.7508 - lr: 1.0000e-04
Epoch 32/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5024 - accuracy: 0.7663 - lr: 1.1659e-04
Epoch 33/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4886 - accuracy: 0.7747 - lr: 1.3594e-04
Epoch 34/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4745 - accuracy: 0.7851 - lr: 1.5849e-04
Epoch 35/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4629 - accuracy: 0.7918 - lr: 1.8478e-04
Epoch 36/100
478/478 [==============================] - 16s 32ms/step - loss: 0.4494 - accuracy: 0.8011 - lr: 2.1544e-04
Epoch 37/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4422 - accuracy: 0.8025 - lr: 2.5119e-04
Epoch 38/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4292 - accuracy: 0.8114 - lr: 2.9286e-04
Epoch 39/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4249 - accuracy: 0.8116 - lr: 3.4145e-04
Epoch 40/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4216 - accuracy: 0.8147 - lr: 3.9811e-04
Epoch 41/100
478/478 [==============================] - 16s 33ms/step - loss: 0.4166 - accuracy: 0.8166 - lr: 4.6416e-04
Epoch 42/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4122 - accuracy: 0.8192 - lr: 5.4117e-04
Epoch 43/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4099 - accuracy: 0.8187 - lr: 6.3096e-04
Epoch 44/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4046 - accuracy: 0.8230 - lr: 7.3564e-04
Epoch 45/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4047 - accuracy: 0.8221 - lr: 8.5770e-04
Epoch 46/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4001 - accuracy: 0.8236 - lr: 0.0010
Epoch 47/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3924 - accuracy: 0.8288 - lr: 0.0012
Epoch 48/100
478/478 [==============================] - 16s 32ms/step - loss: 0.3838 - accuracy: 0.8299 - lr: 0.0014
Epoch 49/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3748 - accuracy: 0.8348 - lr: 0.0016
Epoch 50/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3620 - accuracy: 0.8408 - lr: 0.0018
Epoch 51/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3518 - accuracy: 0.8459 - lr: 0.0022
Epoch 52/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3401 - accuracy: 0.8540 - lr: 0.0025
Epoch 53/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3346 - accuracy: 0.8525 - lr: 0.0029
Epoch 54/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3253 - accuracy: 0.8582 - lr: 0.0034
Epoch 55/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3187 - accuracy: 0.8585 - lr: 0.0040
Epoch 56/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3131 - accuracy: 0.8634 - lr: 0.0046
Epoch 57/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3137 - accuracy: 0.8617 - lr: 0.0054
Epoch 58/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3064 - accuracy: 0.8658 - lr: 0.0063
Epoch 59/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3037 - accuracy: 0.8656 - lr: 0.0074
Epoch 60/100
478/478 [==============================] - 15s 32ms/step - loss: 0.2960 - accuracy: 0.8708 - lr: 0.0086
Epoch 61/100
478/478 [==============================] - 16s 33ms/step - loss: 0.2952 - accuracy: 0.8710 - lr: 0.0100
Epoch 62/100
478/478 [==============================] - 16s 33ms/step - loss: 0.2933 - accuracy: 0.8729 - lr: 0.0117
Epoch 63/100
478/478 [==============================] - 15s 32ms/step - loss: 0.2916 - accuracy: 0.8739 - lr: 0.0136
Epoch 64/100
478/478 [==============================] - 15s 32ms/step - loss: 0.2984 - accuracy: 0.8681 - lr: 0.0158
Epoch 65/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3080 - accuracy: 0.8684 - lr: 0.0185
Epoch 66/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3089 - accuracy: 0.8653 - lr: 0.0215
Epoch 67/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3246 - accuracy: 0.8573 - lr: 0.0251
Epoch 68/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3479 - accuracy: 0.8466 - lr: 0.0293
Epoch 69/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3666 - accuracy: 0.8376 - lr: 0.0341
Epoch 70/100
478/478 [==============================] - 15s 32ms/step - loss: 0.3844 - accuracy: 0.8296 - lr: 0.0398
Epoch 71/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4212 - accuracy: 0.8085 - lr: 0.0464
Epoch 72/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4611 - accuracy: 0.7888 - lr: 0.0541
Epoch 73/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4684 - accuracy: 0.7865 - lr: 0.0631
Epoch 74/100
478/478 [==============================] - 15s 32ms/step - loss: 0.4945 - accuracy: 0.7659 - lr: 0.0736
Epoch 75/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5201 - accuracy: 0.7471 - lr: 0.0858
Epoch 76/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5494 - accuracy: 0.7237 - lr: 0.1000
Epoch 77/100
478/478 [==============================] - 15s 32ms/step - loss: 0.5901 - accuracy: 0.6915 - lr: 0.1166
Epoch 78/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6006 - accuracy: 0.6893 - lr: 0.1359
Epoch 79/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6049 - accuracy: 0.6802 - lr: 0.1585
Epoch 80/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6153 - accuracy: 0.6690 - lr: 0.1848
Epoch 81/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6309 - accuracy: 0.6543 - lr: 0.2154
Epoch 82/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6452 - accuracy: 0.6452 - lr: 0.2512
Epoch 83/100
478/478 [==============================] - 15s 31ms/step - loss: 0.6556 - accuracy: 0.6347 - lr: 0.2929
Epoch 84/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6657 - accuracy: 0.6252 - lr: 0.3415
Epoch 85/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6654 - accuracy: 0.6252 - lr: 0.3981
Epoch 86/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6704 - accuracy: 0.6179 - lr: 0.4642
Epoch 87/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6695 - accuracy: 0.6229 - lr: 0.5412
Epoch 88/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6760 - accuracy: 0.6056 - lr: 0.6310
Epoch 89/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6733 - accuracy: 0.6084 - lr: 0.7356
Epoch 90/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6754 - accuracy: 0.6072 - lr: 0.8577
Epoch 91/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6764 - accuracy: 0.6140 - lr: 1.0000
Epoch 92/100
478/478 [==============================] - 15s 32ms/step - loss: 0.7012 - accuracy: 0.6072 - lr: 1.1659
Epoch 93/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6944 - accuracy: 0.5885 - lr: 1.3594
Epoch 94/100
478/478 [==============================] - 15s 32ms/step - loss: 0.6846 - accuracy: 0.6036 - lr: 1.5849
Epoch 95/100
478/478 [==============================] - 15s 32ms/step - loss: 0.7246 - accuracy: 0.5933 - lr: 1.8478
Epoch 96/100
478/478 [==============================] - 15s 32ms/step - loss: 0.7066 - accuracy: 0.5802 - lr: 2.1544
Epoch 97/100
478/478 [==============================] - 15s 32ms/step - loss: 0.7098 - accuracy: 0.5897 - lr: 2.5119
Epoch 98/100
478/478 [==============================] - 15s 32ms/step - loss: 0.7355 - accuracy: 0.5692 - lr: 2.9286
Epoch 99/100
478/478 [==============================] - 15s 32ms/step - loss: 0.7078 - accuracy: 0.5792 - lr: 3.4145
Epoch 100/100
478/478 [==============================] - 15s 32ms/step - loss: 0.7194 - accuracy: 0.5664 - lr: 3.9811
```

Once the `determine_lr` finishes, I plot the loss against the learning rate on a semilogx plot to determine where the loss reaches a minimum as a function of the learning rate.

```Python
min_lr = "{:.3e}".format(history1.history["lr"][history1.history["loss"].index(min(history1.history["loss"]))])
fig = plt.figure(figsize = (12, 8))
plt.semilogx(history1.history["lr"], history1.history["loss"],color="maroon")
plt.xlabel("Learning Rate (log)")
plt.ylabel("Loss")
plt.title("Learning Rate Determination for Model1\n Minimum Learning Rate = "+str(min_lr)[0:])
plt.show()
```
<p align="center">
  <img src="/images/lr_min1.png" width="700">
</p>

The loss function reaches a minimum at a learning rate of `1.359e-02`, and so now I train `model1` with that learning rate. 


### Training the Model
To automate this, I define the `run_model()` method, which takes in the a model, a learning rate, and an epoch number, and returns the model's training history. To see how the model predicts testing data and training data, I call `plot_graphs()` to plot how the model's loss and accuracy for both datasets.

```Python
def run_model(model, lrate, epoch_num):
    optimizer = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9) #Stochastic gradient descent optimizer
    model.compile(optimizer = 'adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(x=X_train, y=y_train, epochs=epoch_num, validation_data=(X_test, y_test))
    model.summary()
    return history

def plot_graphs(history, model_name):
    plt.plot(history.history[model_name])
    plt.plot(history.history['val_'+model_name])
    plt.xlabel("Epochs")
    plt.ylabel(model_name)
    plt.title(model_name + " over " + str(num_epochs) + " epochs")
    plt.legend([model_name, 'val_'+model_name])
    plt.show()
```

```python
history1_train = run_model(model1,1.359e-2, 50)
plot_graphs(history1_train, "accuracy")
plot_graphs(history1_train, "loss")
```

```
Epoch 1/50
478/478 [==============================] - 21s 40ms/step - loss: 0.6202 - accuracy: 0.6661 - val_loss: 0.5754 - val_accuracy: 0.7022
Epoch 2/50
478/478 [==============================] - 17s 36ms/step - loss: 0.5202 - accuracy: 0.7545 - val_loss: 0.5634 - val_accuracy: 0.7162
Epoch 3/50
478/478 [==============================] - 17s 36ms/step - loss: 0.4794 - accuracy: 0.7818 - val_loss: 0.5702 - val_accuracy: 0.7141
Epoch 4/50
478/478 [==============================] - 17s 36ms/step - loss: 0.4499 - accuracy: 0.8008 - val_loss: 0.5939 - val_accuracy: 0.7070
Epoch 5/50
478/478 [==============================] - 17s 37ms/step - loss: 0.4239 - accuracy: 0.8163 - val_loss: 0.6021 - val_accuracy: 0.7084
Epoch 6/50
478/478 [==============================] - 17s 37ms/step - loss: 0.4055 - accuracy: 0.8220 - val_loss: 0.6544 - val_accuracy: 0.7048
Epoch 7/50
478/478 [==============================] - 18s 37ms/step - loss: 0.3870 - accuracy: 0.8321 - val_loss: 0.6907 - val_accuracy: 0.7067
Epoch 8/50
478/478 [==============================] - 18s 37ms/step - loss: 0.3704 - accuracy: 0.8376 - val_loss: 0.7043 - val_accuracy: 0.6885
Epoch 9/50
478/478 [==============================] - 18s 37ms/step - loss: 0.3563 - accuracy: 0.8425 - val_loss: 0.7370 - val_accuracy: 0.7021
Epoch 10/50
478/478 [==============================] - 17s 36ms/step - loss: 0.3440 - accuracy: 0.8488 - val_loss: 0.6977 - val_accuracy: 0.7009
Epoch 11/50
478/478 [==============================] - 17s 37ms/step - loss: 0.3318 - accuracy: 0.8539 - val_loss: 0.7649 - val_accuracy: 0.6921
Epoch 12/50
478/478 [==============================] - 19s 39ms/step - loss: 0.3248 - accuracy: 0.8547 - val_loss: 0.7893 - val_accuracy: 0.7006
Epoch 13/50
478/478 [==============================] - 18s 39ms/step - loss: 0.3138 - accuracy: 0.8606 - val_loss: 0.7853 - val_accuracy: 0.6996
Epoch 14/50
478/478 [==============================] - 18s 37ms/step - loss: 0.3042 - accuracy: 0.8657 - val_loss: 0.8560 - val_accuracy: 0.6976
Epoch 15/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2960 - accuracy: 0.8652 - val_loss: 0.9709 - val_accuracy: 0.6977
Epoch 16/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2866 - accuracy: 0.8688 - val_loss: 0.9408 - val_accuracy: 0.6905
Epoch 17/50
478/478 [==============================] - 17s 37ms/step - loss: 0.2785 - accuracy: 0.8722 - val_loss: 0.9833 - val_accuracy: 0.6942
Epoch 18/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2696 - accuracy: 0.8763 - val_loss: 0.9856 - val_accuracy: 0.6973
Epoch 19/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2629 - accuracy: 0.8778 - val_loss: 0.9774 - val_accuracy: 0.6880
Epoch 20/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2555 - accuracy: 0.8816 - val_loss: 0.9803 - val_accuracy: 0.6938
Epoch 21/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2469 - accuracy: 0.8858 - val_loss: 1.0402 - val_accuracy: 0.6836
Epoch 22/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2414 - accuracy: 0.8860 - val_loss: 1.0516 - val_accuracy: 0.6884
Epoch 23/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2350 - accuracy: 0.8892 - val_loss: 1.0598 - val_accuracy: 0.6916
Epoch 24/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2283 - accuracy: 0.8906 - val_loss: 1.1724 - val_accuracy: 0.6747
Epoch 25/50
478/478 [==============================] - 18s 38ms/step - loss: 0.2269 - accuracy: 0.8928 - val_loss: 1.0596 - val_accuracy: 0.6698
Epoch 26/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2230 - accuracy: 0.8958 - val_loss: 1.1966 - val_accuracy: 0.6792
Epoch 27/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2126 - accuracy: 0.8985 - val_loss: 1.1960 - val_accuracy: 0.6785
Epoch 28/50
478/478 [==============================] - 18s 38ms/step - loss: 0.2125 - accuracy: 0.8992 - val_loss: 1.1695 - val_accuracy: 0.6851
Epoch 29/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2046 - accuracy: 0.9031 - val_loss: 1.3259 - val_accuracy: 0.6813
Epoch 30/50
478/478 [==============================] - 18s 38ms/step - loss: 0.2032 - accuracy: 0.9025 - val_loss: 1.2916 - val_accuracy: 0.6735
Epoch 31/50
478/478 [==============================] - 18s 37ms/step - loss: 0.2000 - accuracy: 0.9062 - val_loss: 1.3189 - val_accuracy: 0.6835
Epoch 32/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1987 - accuracy: 0.9070 - val_loss: 1.2896 - val_accuracy: 0.6801
Epoch 33/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1866 - accuracy: 0.9130 - val_loss: 1.3800 - val_accuracy: 0.6736
Epoch 34/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1858 - accuracy: 0.9136 - val_loss: 1.2356 - val_accuracy: 0.6703
Epoch 35/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1883 - accuracy: 0.9143 - val_loss: 1.4163 - val_accuracy: 0.6709
Epoch 36/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1807 - accuracy: 0.9167 - val_loss: 1.3581 - val_accuracy: 0.6722
Epoch 37/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1771 - accuracy: 0.9172 - val_loss: 1.3777 - val_accuracy: 0.6608
Epoch 38/50
478/478 [==============================] - 17s 37ms/step - loss: 0.1743 - accuracy: 0.9203 - val_loss: 1.4809 - val_accuracy: 0.6734
Epoch 39/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1731 - accuracy: 0.9204 - val_loss: 1.4263 - val_accuracy: 0.6662
Epoch 40/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1709 - accuracy: 0.9223 - val_loss: 1.3935 - val_accuracy: 0.6658
Epoch 41/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1665 - accuracy: 0.9235 - val_loss: 1.3850 - val_accuracy: 0.6577
Epoch 42/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1665 - accuracy: 0.9256 - val_loss: 1.4827 - val_accuracy: 0.6670
Epoch 43/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1607 - accuracy: 0.9272 - val_loss: 1.6532 - val_accuracy: 0.6704
Epoch 44/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1571 - accuracy: 0.9305 - val_loss: 1.6128 - val_accuracy: 0.6595
Epoch 45/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1600 - accuracy: 0.9281 - val_loss: 1.5414 - val_accuracy: 0.6674
Epoch 46/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1526 - accuracy: 0.9318 - val_loss: 1.5113 - val_accuracy: 0.6679
Epoch 47/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1485 - accuracy: 0.9344 - val_loss: 1.6543 - val_accuracy: 0.6681
Epoch 48/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1485 - accuracy: 0.9355 - val_loss: 1.4974 - val_accuracy: 0.6506
Epoch 49/50
478/478 [==============================] - 18s 38ms/step - loss: 0.1433 - accuracy: 0.9383 - val_loss: 1.6149 - val_accuracy: 0.6614
Epoch 50/50
478/478 [==============================] - 18s 37ms/step - loss: 0.1410 - accuracy: 0.9394 - val_loss: 1.6766 - val_accuracy: 0.6669
```

<p align="center">
  <img src="/images/accuracy1.png" width="700">
</p>

<p align="center">
  <img src="/images/loss1.png" width="700">
</p>

Clearly, the model overfits as the testing accuracy and loss and the validation accuracy and loss do not line up.

<!-- ## Training Other Models

Because the first model overfits the data, I attempted to build a simplier model to hopefully reduce the overfitting. -->


## Conclusions and Next Steps
Taking a look at the data, there are several steps I can take to find a model that predicts the sentiment of tweets with negations in more accurately. First of all, I can perform a grid-search across different hyperparameters, which would include varying the dropout fraction, the lasso and ridge regression coefficients, the vocabularly size, and the learning rate for each subset. I could also look into simpler models to reduce overfitting, such as a Naive Bayes' analysis. Finally, I could also perform several other data manipulation techniques, including lemmitization, tf-idf analysis, and n-gram analysis. This portfolio project is still a work in progress, so come back to see updates as they come!