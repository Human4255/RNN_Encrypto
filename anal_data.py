from os import times
import requests
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np


def get_url():
    nameurl = "https://api.bithumb.com/v1/market/all?isDetails=false"
    headers = {"accept": "application/json"}
    response = requests.get(nameurl, headers=headers)
    #print(type(response.text)) #<class 'str'>
    res_dict = json.loads(response.text) #json.dumps
    # print(res_dict[0].keys()) #[마켓, 한국이름, 영어이름]
    # print(res_dict[0]["market"]) #KRW-BTC -> KRW마켓
    # print(res_dict[0]["korean_name"]) #비트코인
    # print(res_dict[0]["english_name"]) #Bitcoin
    # print(res_dict[5]["market"]) #KRW-QTUM
    # print(res_dict[5]["korean_name"]) #퀀텀
    # print(res_dict[5]["english_name"]) #Qtum
    # print()
    encrypto_names = []
    for data in res_dict:
        if not "KRW-" in data["market"]:
            #print(data["market"]) #BTC-ETH -> BTC마켓
            encrypto_names.append({"symbol":data["market"].split("-")[1],"eng":data["english_name"],"kor":data["korean_name"]})
    return encrypto_names

# 1. 데이터유형 - [{sumbol:BTC,kor:한국이름,eng}]
# print(encrypto_names)
def GetCandleData(currency="BTC", times="24h", pyament="KRW"):
    order_currency = "BTC" # order_currency: 화폐명
    payment_currency = "KRW" # payment_currency: 지불화폐
    chart_intervals = "24h" # chart_intervals: 데이터간격(시간)
    candle_url = f"https://api.bithumb.com/public/candlestick/{order_currency}_{payment_currency}/{chart_intervals}"
    # print(candle_url)
    headers = {"accept": "application/json"}
    response = requests.get(candle_url, headers=headers)
    candle_data = json.loads(response.text)
    if candle_data["status"]=="0000":
        #print(len(candle_data["data"])) #3925
        #print(type(candle_data["data"][0][0])) #<class 'int'> --> json은 문자는문자로, 숫자는 숫자로 바꿔줌
        #[1388070000000, '737000', '755000', '755000', '737000', '3.78']
        #[    기준시간.     시가,       종가,     고가,      저가,     거래량]
        return candle_data
    else:
        return False

#현재 정렬순서가 최근 데이터가 맨 뒤에 위치함
def GenerateData(sorce_data,timeslot):
    x_data = []
    y_data = [] #정답데이터
    for ix in range(len(sorce_data)-timeslot):
        slot_data = []
        for cur_ix in range(ix,timeslot+ix):
            slot_data.append(sorce_data[cur_ix])
        x_data.append(slot_data)
        y_data.append(sorce_data[timeslot+ix])
    return np.array(x_data),np.array(y_data)

#문제 데이터와 정답데이터 일치성 확인
def confirm_data(x_data,y_data,sorce_data):
    result_bool = True
    if y_data[0] != x_data[1][-1]: #정답 파일 확인
        result_bool = False
    if y_data[1] != x_data[2][-1]:
        result_bool = False
    if y_data[-1] != sorce_data[-1]: #마지막 데이터 확인
        result_bool = False
    if y_data[-2] != sorce_data[-2]:
        result_bool = False
    return result_bool

#산점도 분석
def ScatterAnal(x_data,y_data):
    cvdata = np.average(x_data,axis=1,weights=(0.02,0.04,0.04,0.08,0.08,0.1,0.13,0.15,0.16,0.2))
    plt.scatter(x_data,y_data)
    plt.show()

#print(type(candle_data["data"][:][1])) #<class 'list'>
# raw_data = np.array(candle_data["data"])
# sorce_data = raw_data[:,1:-1].astype("float")
# x_data,y_data = GenerateData(sorce_data,10)
# print(x_data.shape) #(3925, 10)
# print("정답문제일치성:",y_data[0]==x_data[1][-1])
# print("정답문제일치성:",y_data[1]==x_data[2][-1])
# print("정답문제일치성:",y_data[0]==x_data[2][-2])
# #맨마지막데이터
# print(sorce_data[-1]) #143226000
# print("마지막데이터일치설:",y_data[-1]==sorce_data[-1])
# print("마지막데이터일치설:",y_data[-2]==sorce_data[-2])
# #scatter는 1차원만을 받기에 x_data 10개를 하나의 값 즉 평균으로 넣어줄것이다.
# print(x_data.shape)

if __name__ == "__main__":
    pass

