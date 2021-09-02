#주택 가격 예측: 회귀 문제
#보스턴 주택 데이터셋 로드하기
from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

