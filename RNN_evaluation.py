import matplotlib.pyplot as plt
import pickle

def Run_grahp(userInput, timeArr):
    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    for ix, cname in enumerate(userInput):
        for tx,ctime in enumerate(timeArr):
            plt.subplot(len(userInput),len(timeArr),ix+tx+1)
            with open(r"models\{}_{}_fit_his".format(cname,ctime),"rb") as fp:
                fit_his = pickle.load(fp)
                plt.plot(fit_his.history["loss"],label="LOSS")
                plt.plot(fit_his.history["val_loss"],label="VAL_LOSS")
                plt.legend()
                plt.title(f"{cname}_{ctime}MSE LOSS")
                plt.xticks([]); plt.yticks([])
    plt.show()

def ConvertValue(scaler,val):
    return scaler.inverse_transform(val)

def Today_predict(rmodel,today_x):
    return rmodel.predict(today_x)

def EvaluationModel(rmodel,xd,yd):
    return rmodel.evaluate(xd,yd)