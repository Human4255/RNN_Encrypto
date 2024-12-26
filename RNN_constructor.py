import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, LSTM, Reshape


def ConstructorModel(timeslot):
    rmodel = Sequential()
    rmodel.add(Input(shape=(timeslot,1)))
    rmodel.add(LSTM(128,activation="relu",dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
     #return_sequences=True이면 순회 시 가중치를 모두 추출
     #return_sequences=False이면 마지막 가중치를 모두 추출
    rmodel.add(Dropout(0.2))
    rmodel.add(LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    rmodel.add(Dropout(0.2))
    rmodel.add(LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    rmodel.add(Dropout(0.2))
    rmodel.add(Dense(1,)) #회귀문제로 가중치를 그대로 출력한다.
    rmodel.compile(loss="MSE",optimizer="adam",metrics=["acc"])
    return rmodel