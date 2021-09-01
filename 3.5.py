#뉴스 기사 분류: 다중 분류 문제
#로이터 데이터셋 로드하기
from tensorflow.keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.lad_data(num_words=10000)

#로이터 데이터셋을 텍스트로 디코딩하기
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decorded_newswire = ''.join([reverse_word_index.get(i -3, '?') for i in train_data])

#데이터 인코딩하기 -> 원핫인코딩(범주형 인코딩): 각 레이블의 인덱스 자리는 1이고 나머지는 모두 0인 벡터
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#모델 정의하기
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

#모델 컴파일하기
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#검증 세트 준비하기
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[:1000]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

#훈련과 검증 손실 그리기
import matplotlib.pyplot as plt

loss = history.hi