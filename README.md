# EXPERIMENT - 06 Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The task involves developing an LSTM-based neural network model using Bidirectional Recurrent Neural Networks for recognizing the named entities in the text.
The dataset used has a number of sentences, and each words have their tags.Vectorization of words is performed using Embedding techniques to train our model.

## DESIGN STEPS
1. Import the necessary libraries.
2. Analyse the dataset and fill the null values using forward fill.
3. Create a list of words and tags. Also find the number of unique words and tags in the dataset.
4. Create a dictionary for the words and their Index values.
5. Repeat the same for the tags as well.
6. Padding is done to make the input data of same length.
7. Compile and fit the model using training data.
8. Validate the model using validation dataset - training data.

## PROGRAM
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)

data = data.fillna(method="ffill")
# ffill -- forward fill

print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())

print("Unique tags are:", tags)

num_words = len(words)
num_tags = len(tags)

num_words

print(num_words)
print(num_tags)

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
sentences[0]

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
# Assignment of uniques number in order for the words and the tags
word2idx

plt.hist([len(s) for s in sentences], bins=50)
plt.show()
X1 = [[word2idx[w[0]] for w in s] for s in sentences]
type(X1[0])
X1[0]
max_len = 50
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
X[0]

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
X_train[0]
y_train[0]
input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim = num_words,
                                   output_dim = 50,
                                   input_length = max_len)(input_word)
dropout_layer = layers.SpatialDropout1D(0.13)(embedding_layer)
bidirectional_lstm = layers.Bidirectional(layers.LSTM(
    units=100, return_sequences=True,recurrent_dropout=0.13))(dropout_layer)
output = layers.TimeDistributed(
    layers.Dense(num_tags, activation="softmax"))(bidirectional_lstm)
model = Model(input_word, output)
model.summary()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=45,
    epochs=3,)

metrics = pd.DataFrame(model.history.history)

metrics.head()

print("Shavedha Y - 212221230095")
metrics[['accuracy','val_accuracy']].plot()


print("Shavedha Y - 212221230095")
metrics[['loss','val_loss']].plot()

print("Shavedha Y - 212221230095")
i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Shavedha/named-entity-recognition/assets/93427376/de3f9426-e2dc-4ed1-8c41-f2d56fc61c6d)

![image](https://github.com/Shavedha/named-entity-recognition/assets/93427376/e9ecf5e5-a0d5-4c42-950b-b885bf6e96e8)

### Sample Text Prediction
![image](https://github.com/Shavedha/named-entity-recognition/assets/93427376/09d64e75-2c03-464b-bb1f-b271112cff43)


## RESULT
Thus an LSTM-based model for recognizing the named entities in the text is successfully developed.
