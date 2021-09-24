#
import pandas as pd

raw_data = pd.read_csv("D:/Research/1Action_Recognition/Experiment/test/ACC_data.csv", encoding='utf-8')

Step1 = raw_data[993:1474]

Step2 = raw_data[2313:3213]

Step3 = raw_data[3371:4013]

Step4 = raw_data[4141:4825]

Step5 = raw_data[5066:5843]

Step6 = raw_data[6490:6910]

#데이터에다가 1/64을 해줘야한다. /g는 HAR에서 기본 단위로 하므로 g는 곱하지말자.

print(Step1.shape)