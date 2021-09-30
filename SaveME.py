import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import RobustScaler
from matplotlib.animation import FuncAnimation
PWD = list(glob.glob('data/*'))


SUBJECT_NUM = 4
SAMPLING_RATE = 8 ## 112번째 줄에서 [asdasd:asdfasdf:1] 1이면 1씩 건너뛰기(원본) 2면 2씩건너뛰어서 32/2 = 16이 된다.
INTERVAL = 1/SAMPLING_RATE
ACTION_TIME = 4
WINDOW_SIZE = 4# action의 지속시간 (단위는 초)
LOOT_AT = int(WINDOW_SIZE * SAMPLING_RATE)
TESTSET_SIZE = 12 # test set 개수
TEST_RATIO = 0.2


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    fig.tight_layout()
    # fig.savefig("test.svg")
    return ax
def get_sample_number(array, window_size, freq, overlap=0.75):
    return int((array.shape[0]-int(window_size*freq))/int(window_size*freq*(1-overlap)))
def returnRobust(array):
  array = np.asarray(array)
  reshaped = array.reshape((-1,1))
  robust_scaler = RobustScaler()
  robust_scaler.fit(reshaped)
  return robust_scaler.transform(reshaped)
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
Move = raw_data[994:1475:4] #Step1

Check = raw_data[2314:3214:4]

Mortar = raw_data[3374:4014:4]

Lay = raw_data[4142:4826:4]

Position = raw_data[5067:5844:4]

Cut = raw_data[6491:6911:4]

#데이터에다가 1/64을 해줘야한다. /g는 HAR에서 기본 단위로 하므로 g는 곱하지말자.

data_frame_Move = pd.DataFrame(Move)
data_frame_Check = pd.DataFrame(Check)
data_frame_Mortar = pd.DataFrame(Mortar)
data_frame_Lay = pd.DataFrame(Lay)
data_frame_Position = pd.DataFrame(Position)
data_frame_Cut = pd.DataFrame(Cut)

mergedData = {
    'Move' : data_frame_Move,
    'Check' : data_frame_Check,
    'Mortar' : data_frame_Mortar,
    'Lay' : data_frame_Lay,
    'Position' : data_frame_Position,
    'Cut' : data_frame_Cut,
}
for actionName in list(mergedData.keys()):
    mergedData[actionName]['acc_X_value'] = returnRobust(mergedData[actionName]['acc_X_value'])
    mergedData[actionName]['acc_Y_value'] = returnRobust(mergedData[actionName]['acc_Y_value'])
    mergedData[actionName]['acc_Z_value'] = returnRobust(mergedData[actionName]['acc_Z_value'])
print(mergedData['Move'])

data = dict()
dataTest = dict()

numOfLabel = dict()
numOfLabelTest = dict()

label = dict()
labelTest = dict()

for index, action_name in enumerate(list(mergedData.keys())):
    start = int(0)
    startTest = int(0)

    data_tmp = []
    walk_data_tmp = []
    label_tmp = []
    label_walk_tmp = []
    data_tmpTest = []
    label_tmpTest = []

    numOfLabel[action_name] = get_sample_number(mergedData[action_name], WINDOW_SIZE, SAMPLING_RATE)
    # numOfLabelTest[action_name] = get_sample_number(mergedDataTest[action_name], WINDOW_SIZE, SAMPLING_RATE)

    for i in range(get_sample_number(mergedData[action_name], WINDOW_SIZE, SAMPLING_RATE)):
        end = int(start + LOOT_AT)
        data_tmp.append(np.asarray(mergedData[action_name])[start:end, :])
        '''if action_name == "backward":###########
            label_tmp.append(0)
        else:
            label_tmp.append(1)'''
        # if action_name == "walking":
        #     continue
        label_tmp.append(np.asarray(index))
        start = int(start + LOOT_AT / 4)

    # for i in range(get_sample_number(mergedDataTest[action_name], WINDOW_SIZE, SAMPLING_RATE)):
    #     endTest = int(startTest + LOOT_AT)
    #     data_tmpTest.append(np.asarray(mergedDataTest[action_name])[startTest:endTest, :])
    #     '''if action_name == "backward":##############
    #         label_tmpTest.append(0)
    #     else:
    #         label_tmpTest.append(1)'''
    #     if action_name == "walking":
    #         continue
    #     label_tmpTest.append(np.asarray(index))
    #     startTest = int(startTest + LOOT_AT / 4)

    data[action_name] = np.vstack(data_tmp)
    label[action_name] = np.vstack(label_tmp)
    data[action_name] = np.reshape(data[action_name], (-1, LOOT_AT, 3))

    # dataTest[action_name] = np.vstack(data_tmpTest)
    # labelTest[action_name] = np.vstack(label_tmpTest)
    # dataTest[action_name] = np.reshape(dataTest[action_name], (-1, LOOT_AT, 3))

    if True:
        print("%s total train data size : " % action_name, data[action_name].shape)
        # print("%s total test data size : " % action_name, dataTest[action_name].shape)
    #    print("%s total label size : "%action_name, label[action_name].shape)

train_data = np.vstack([data[action_name] for action_name in list(mergedData.keys())])
train_label = np.vstack([label[action_name] for action_name in list(mergedData.keys())])
print(train_data.shape, train_label.shape)
train_label = to_categorical(train_label)

train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=TEST_RATIO, random_state=1)

print("Train data size: ", train_data.shape)
print("Train label size: ", train_label.shape)
print("Test data size: ", test_data.shape)
print("Test label size: ", test_label.shape)


###########################################
###########################################
import os

from tensorflow.keras.layers import Dense,Dropout,Conv1D,BatchNormalization,MaxPool1D,Flatten, LeakyReLU, TimeDistributed, LSTM, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding1D#, addpadding
from tensorflow.keras.callbacks import EarlyStopping

#history = History()
inputs = Input(shape=(train_data.shape[1],3), name="input")
#
x = Conv1D(80, kernel_size=3, strides=2, input_shape=(LOOT_AT , 3), padding="same")(inputs)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv1D(40, kernel_size=3, strides=2, padding="same")(x)

x = ZeroPadding1D(padding=(0,1))(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = BatchNormalization(momentum=0.8)(x)

x = Conv1D(16, kernel_size=3, strides=1, padding="same")(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)

pool = MaxPool1D(pool_size=2,strides=2,padding='same')(x)
flat = Flatten()(pool)
lstm1 = LSTM(16, return_sequences=False)(pool)

output = Dense(6,activation='softmax', name='classification')(lstm1)
model = Model(inputs, [output])
print(model.summary())
adam=tensorflow.keras.optimizers.Adam(lr=0.0001)


model.compile(loss=['categorical_crossentropy'],metrics=['accuracy'],optimizer=adam)


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 100)


hist=model.fit(train_data, train_label ,batch_size=25,epochs=1000, verbose=2,shuffle=True)
# validation_data=(test_data, [test_label, test_zero_label])
print(hist.history.keys())
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'r', label='train loss')
loss_ax.plot(hist.history['accuracy'], 'g', label='train acc')
# loss_ax.plot(hist.history['validation_loss'], 'r', label='test loss')
#
# acc_ax.plot(hist.history['validation_output_acc'], 'b', label='train acc')
# acc_ax.plot(hist.history['validation_output_acc'], 'g', label='test acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

result = model.predict(test_data)
result = np.argmax(result, axis=1)
print(result)
#print(test_label1)
test_label = np.argmax(test_label, axis=1)
test_label = np.reshape(test_label, (-1,1))
print(classification_report(test_label, result, target_names=list(mergedData.keys())))
plot_confusion_matrix(test_label, result, normalize=True, classes=np.array(list(mergedData.keys())))
plt.show()
