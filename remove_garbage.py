import pandas as pd

# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical

raw_data = pd.read_csv("D:/Research/1Action_Recognition/Experiment/test/ACC_Standard.csv", encoding='utf-8')
# x : shape(n, m, 3) y : (n, 6)
# [ [1,0,0,0,0,0],   -> 1번 동작의 정답
#   [0,0,0,0,0,1].....]    -> 2동작의 정답
Step1 = raw_data[993:1474]

Step2 = raw_data[2313:3213]

Step3 = raw_data[3373:4013]

Step4 = raw_data[4141:4825]

Step5 = raw_data[5066:5843]

Step6 = raw_data[6490:6910]

#데이터에다가 1/64을 해줘야한다. /g는 HAR에서 기본 단위로 하므로 g는 곱하지말자.

X = pd.read_csv("D:/Research/1Action_Recognition/Experiment/test/ACC_Standard.csv", encoding='utf-8', usecols=[0:2])
Y = Step1+ Step2+ Step3+ Step4+ Step5+ Step6

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test, Z_train, Z_test=train_test_split(X,Y,Z, test_size=0.2, shuffle=True, random_state=1)

def load_dataset():
    return X_train, Y_train, Z_train, X_test, Y_test, Z_test

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', X_train.shape)
print('Y_test shape:', X_test.shape)
print('Z_train shape:', X_train.shape)
print('Z_test shape:', X_test.shape)

def evaluate_model(X_Train, Y_Train, Z_Train, X_test, Y_test, Z_test):

    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[0], X_train.shape[0]
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=10, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=128, kernel_size=10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_Train, Y_train, Z_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, Y_test, Z_test, batch_size=batch_size, verbose=0)
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10):
    # load data
    X_train, Y_train, Z_train, X_test, Y_test, Z_test = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, Y_train, Z_train, X_test, Y_test, Z_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
run_experiment()