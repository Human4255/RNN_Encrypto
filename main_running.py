import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from anal_data import get_url,GetCandleData,GenerateData,confirm_data,ScatterAnal
from utility import cv_format,cv_mill2date,cv_date2milli,cv_str2date,GetCompare

#1.화폐이름목록 추출
names = get_url()
print(names[0])

#2.화폐 캔들 데이터 수신
candle_datas = GetCandleData(names[0]["symbol"],times="24h",pyament="KRW")
print(candle_datas.keys())
if candle_datas["status"]=="0000":
    #3.훈련데이터생성
    # [    기준시간.     시가,       종가,     고가,      저가,     거래량]
    source_datas = np.array(candle_datas["data"])
    x_data,y_data = GenerateData(source_datas[:,1],30) #(sorce_data,timeslot)
    print(x_data.shape) #(3895, 30)
    print(y_data.shape) #(3895,)
    
    #4.데이터 일치성확인
    res = confirm_data(x_data,y_data,source_datas[:,1])
    if res:
        print("모든데이터 정답과 일치")
    else:
        print("데이터 혼합이 잘못됨")

else:
    print("데이터 수신 실패")