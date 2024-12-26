import matplotlib.pyplot as plt

def Run_grahp(fit_his):
    plt.subplot(1,2,1)
    plt.plot(fit_his.history["acc"],label="ACC")
    plt.plot(fit_his.history["val_acc"],label="VAL_ACC")
    plt.legend()
    plt.title("ACCURACY")
    plt.xticks([]); plt.yticks([])

    plt.subplot(1,2,2)
    plt.plot(fit_his.history["loss"],label="LOSS")
    plt.plot(fit_his.history["val_loss"],label="VAL_LOSS")
    plt.legend()
    plt.title("MSE LOSS")
    plt.xticks([]); plt.yticks([])
    plt.show()

def ConvertValue(scaler,val):
    return scaler.inverse_transform(val)

def Today_predict(rmodel,today_x):
    return rmodel.predict(today_x)

def EvaluationModel(rmodel,xd,yd):
    return rmodel.evaluate(xd,yd)