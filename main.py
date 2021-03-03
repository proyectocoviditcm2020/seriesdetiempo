import pandas as pd
from TimeSeries import TimeSeries as ts
import eel

eel.init("static")

@eel.expose
def getFile(ruta):
    try:
        DS= pd.read_csv(ruta)
        names = list(DS.columns)
        index = ["{}:{}".format(names.index(i),i) for i in names[1:]]
        eel.receive_colums(index)
    except:
        eel.receive_err_file()

@eel.expose
def sendOptions(ruta,series,perc_train,hm_day_more,epochs,layers,batch):
    DS= pd.read_csv(ruta)
    series = series.split(",")
    series = [eval(i) for i in series]
    # series = [1,2,3,4,5,6]
    # perc_train = 80
    # hm_day_more = 100
    ##Configuracion de la red
    # layers = 1
    # epochs = 150
    # batch = 10
    try:
        Nn = ts(DS,series,perc_train,hm_day_more,layers,epochs,batch)
        Nn.NeuralNetwork()
        eel.receive_check()
        #inferencePredictPlot = np.ones((dataset.shape[0]+inverted.shape[0],1))
        #inferencePredictPlot[:,:] = np.nan
        #inferencePredictPlot[dataset.shape[0], :] = inverted
    except:
        eel.receive_err("An exception occurred")

eel.start("index.html", size=(600, 500))