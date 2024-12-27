import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
import tensorflow as tf
from datetime import datetime
from anal_data import get_url,GetCandleData,GenerateData,confirm_data,ScatterAnal
from utility import cv_format,cv_mill2date,cv_date2milli,cv_str2date,GetCompare
from RNN_constructor import ConstructorModel
from RNN_evaluation import Run_grahp,ConvertValue,Today_predict,EvaluationModel

def Train_running(coinname,korname,timesArr,payment):
    for ix,times in enumerate(timesArr):
        print(f"{len(timesArr)-ix}회 남음:{coinname}({korname}_{timesArr}훈련시작)")
        #2.화폐 캔들 데이터 수신
        candle_datas = GetCandleData(coinname,times=times,payment=payment)
        print(candle_datas.keys())
        if candle_datas["status"]=="0000":
            #3.훈련데이터생성
            # [    기준시간.     시가,       종가,     고가,      저가,     거래량]
            source_datas = np.array(candle_datas["data"])
            x_data_start,y_data_start = GenerateData(source_datas[:,1],timeslot) #(sorce_data,timeslot)
            print(x_data_start.shape) #(4970, 30)
            print(y_data_start.shape) #(4970,)

            #4.데이터 일치성확인
            res = confirm_data(x_data_start,y_data_start,source_datas[:,1])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합이 잘못됨")
            #ScatterAnal(x_data_start,y_data_start,weight_avg,"start_price")

            x_data_end, y_data_end = GenerateData(source_datas[:, 2], timeslot)  # (sorce_data,timeslot)
            print(x_data_end.shape)  # (4970, 30)
            print(y_data_end.shape)  # (4970,)
            res = confirm_data(x_data_end, y_data_end, source_datas[:, 2])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합이 잘못됨")
            print("Length of x_data:", len(x_data_end))
            print("Length of y_data:", len(y_data_end))
            #ScatterAnal(x_data_end, y_data_end, weight_avg, "end_price")

            x_data_high, y_data_high = GenerateData(source_datas[:, 3], timeslot)  # (sorce_data,timeslot)
            print(x_data_high.shape)  # (4970, 30)
            print(y_data_high.shape)  # (4970,)
            res = confirm_data(x_data_high, y_data_high, source_datas[:, 3])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합이 잘못됨")
            print("Length of x_data:", len(x_data_high))
            print("Length of y_data:", len(y_data_high))
            #ScatterAnal(x_data_high, y_data_high, weight_avg, "MAX_price")

            x_data_low, y_data_low = GenerateData(source_datas[:, 4], timeslot)  # (sorce_data,timeslot)
            print(x_data_low.shape)  # (4970, 30)
            print(y_data_low.shape)  # (4970,)
            res = confirm_data(x_data_low, y_data_low, source_datas[:, 4])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합이 잘못됨")
            print("Length of x_data:", len(x_data_low))
            print("Length of y_data:", len(y_data_low))
            #ScatterAnal(x_data_low, y_data_low, weight_avg, "MIN_price")

            x_data_amount, y_data_amount = GenerateData(source_datas[:, 5], timeslot)  # (sorce_data,timeslot)
            print(x_data_amount.shape)  # (4970, 30)
            print(y_data_amount.shape)  # (4970,)
            res = confirm_data(x_data_amount, y_data_amount, source_datas[:, 5])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합이 잘못됨")
            print("Length of x_data:", len(x_data_amount))
            print("Length of y_data:", len(y_data_amount))
            #ScatterAnal(x_data_amount, y_data_amount, weight_avg, "amount_price")

            #분석결과 5인덱스의 quantity부분은 선형선과 관련부족으로 제외
        else:
            print("데이터 수신 실패")

        #5.최종데이터 생성
        x_data_start #[[1 2 32 4 5 3 55 6544...]]
        x_data_end #[[1 2 32 4 5 3 55 6544...]]
        x_data_high #[[1 2 32 4 5 3 55 6544...]]
        x_data_low #[[1 2 32 4 5 3 55 6544...]]
        print(x_data_start.shape) #(4970, 30)
        print(type(x_data_start)) #<class 'numpy.ndarray'>
         #x_dataset
        x_dataset = []
        for ix in range(len(x_data_start)):
            x_d = []
            for tx in range(timeslot):
                x_d.append(sum([x_data_start[ix][tx],x_data_end[ix][tx],x_data_high[ix][tx],x_data_low[ix][tx]])/4)
            x_dataset.append(x_d)
         #y_dataset
        y_dataset = []
        for ix in range(len(y_data_start)):
            y_dataset.append(sum([y_data_start[ix],y_data_end[ix],y_data_high[ix],y_data_low[ix]])/4)

        x_data = np.array(x_dataset)
        y_data = np.array(y_dataset).reshape((len(y_dataset),-1))


        print(x_data.shape)
        print(y_data.shape)
        print(type(x_data[0])) #<class 'numpy.ndarray'>

        #6.데이터정규화
        scaler = sklearn.preprocessing.MinMaxScaler()
        x_data = scaler.fit_transform(x_data).reshape((len(x_data),timeslot,-1))
        y_data = scaler.fit_transform(y_data)
        print(x_data[0][:10])
        print(y_data[:10])
        rmodel = None
        #LSTM모델
        if not os.path.exists(r"models\{}_{}_rnnmodel.keras".format(coinname,times)):
            rmodel = ConstructorModel(timeslot)
        else:
            rmodel = tf.keras.models.load_model(r"models\{}_{}_rnnmodel.keras".format(coinname,times))
        fit_his = rmodel.fit(x_data,y_data,validation_data=(x_data,y_data),epochs=count_epoch,batch_size=len(x_data)//10)
        rmodel.save(r"models\{}_{}_rnnmodel.keras".format(coinname,times))
        with open(r"models\{}_{}_fit_his".format(coinname,times), "wb") as fp:
            pickle.dump(fit_his, fp)
        #스케일러 저장
        if not os.path.exists(r"models\{}_scaler".format(coinname)):
            with open(r"models\{}_scaler".format(coinname), "wb") as fp:
                pickle.dump(scaler, fp)
        #모델그리기&예측
        loss,acc = EvaluationModel(rmodel,x_data,y_data)
        print("손실도:",loss," :: 정확도:",acc)
        rarr = np.random.randint(0,len(x_data)-2,9)
        rarr = np.append(rarr, [len(x_data) - 1], axis=0)
        test_x = []
        test_y = []
        for i in rarr:
            test_x.append(x_data[i])
            test_y.append(y_data[i])
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        print(test_x.shape) #(10,60,1)
        print(test_y.shape) #(10,60,1)

        pred_y = Today_predict(rmodel, test_x)
        true_value = ConvertValue(scaler,test_y)
        pred_value = ConvertValue(scaler,pred_y)
        for i in range(len(true_value)):
            print(ix+1,".실제값:",round(true_value[i][0],4),"예측값:",round(pred_value[i][0],4))
            rat = (np.abs(pred_value - true_value) / true_value).sum()/len(true_value) *100
        print("{} ({}) {} 실제값과 예측값 오차율: {:.2f}".format(coinname,korname,times,rat,"%"))


names = get_url()
nameArr = [obj["symbol"] for obj in names]
print(",".join(nameArr))
userInput = input("분석 할 화폐목록을 콤마로 구분하여 작성해주세요. (전체선택 시 all을 입력하세요)\n")
if userInput == "all":
    names = names
else:
    userInput = userInput.split(",")
    names=[{"symbol":obj["symbol"],"eng":obj["eng"],"kor":obj["kor"] }for obj in names if obj["symbol"] in userInput]
# print(names)
#변수
timeslot = 60
weight_avg = np.linspace(0,1,timeslot)#평균가중치 부분
count_epoch = input("훈련 횟수를 숫자로 지정하세요\n")
count_epoch = int(count_epoch)
payment = "KRW"
if len(weight_avg) != timeslot:
    print("가중치와 타임슬롯 수량을 동일하게 하세요")

timeArr = ["24h", "12h", "4h", "10m", "3m"] #24시간에 한 번씩, 4시간에 한 번씩
for coinobj in names:
    coinname = coinobj["symbol"]
    korname = coinobj["kor"]
    Train_running(coinname,korname,timeArr,payment)
    break
Run_grahp(userInput, timeArr)