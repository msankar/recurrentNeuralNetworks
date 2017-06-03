import numpy as np
import string
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop

import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    # For the series, loop through the series minus the window size.
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])  # insert our inputs into the input array
        y.append(series[i])  # insert our output pairs into the output array

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # build an RNN to perform regression on our time series input/output data
    model = Sequential()
    # Layer1 add a LSTM layer with 5 hidden units and shape of (window_size, 1)
    model.add(LSTM(5, input_shape=(window_size, step_size)))
    # Layer2 Dense layer with 1 output node and linear activation function
    model.add(Dense(step_size, input_dim=window_size, activation='linear'))

    # Optimizer
    # https://keras.io/getting-started/sequential-model-guide/
    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=opt)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique_chars = (list(set(text)))

    # remove as many non-english characters and character sequences as you can
    import string
    all_chars = string.ascii_lowercase
    all_punctuation_marks = [' ', '!', ',', '.', ':', ';', '?']
    # Use a list to check if each unique character is a non-english character
    # Maintain unique non-english characters in a list
    non_english_chars = list(set([i for i in unique_chars if i not in all_chars and i not in all_punctuation_marks]))
    # For loop over each unique non-english character and replace with a space
    for i in non_english_chars:
        text = text.replace(i, ' ')

    # shorten any extra dead space created above
    text = text.replace('  ', ' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # For each window, determine input and output.
    for i in range(window_size, len(text), step_size):
        inputs.append(text[i - window_size:i])
        outputs.append(text[i])

    return inputs, outputs
