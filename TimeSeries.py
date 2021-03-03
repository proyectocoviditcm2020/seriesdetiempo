import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
class TimeSeries:
    
    def __init__(self, DataFrame, index:list, percet = 80, days_steps= 30, layer = 1, epochs =150, batch = 10):
        self.dataFrame = DataFrame
        columns = list(DataFrame.columns)
        self.names = [columns[i] for i in index]
        self.percent_train = percet/100
        self.days = days_steps
        self.layers = layer 
        self.epochs = epochs
        self.batch = batch

    def create_setPredict(self, days_list,newValue = None):
        if newValue != None:
            days_list = np.delete(days_list,0)
            days_list = np.append(days_list, newValue[0])
            days_reshape = np.array([days_list])
        else:
            days_reshape = np.array([np.concatenate(days_list)])
        days_reshape =  np.reshape(days_reshape, (days_reshape.shape[0], 1, days_reshape.shape[1]))
        return days_list, days_reshape

    #creat Model 
    def createModel(self):
        model = Sequential()
        model.add(LSTM(50, activation= 'tanh', input_shape = ( 1, self.look_back)))
        for i in range(self.layers):
            model.add(Dense(self.look_back, activation= 'tanh'))
        model.add(Dense(1))
        return model

    #convert an array of values into a dataset matrix
    def create_dataset(self, dataset, look_back=1):
    	dataX, dataY = [], []
    	for i in range(len(dataset)-look_back-1):
    		a = dataset[i:(i+look_back), 0]
    		dataX.append(a)
    		dataY.append(dataset[i + look_back, 0])
    	return np.array(dataX), np.array(dataY)

    #Load the dataset
    def NeuralNetwork(self):

        for name in self.names:

            Data = pd.DataFrame(data=self.dataFrame[name].values, index=self.dataFrame['tiempo'])
            dataset = Data.values
            dataset = dataset.astype('float32')
            #plt.show()

            #normalize the data inputs
            scaler = MinMaxScaler(feature_range=(0,1))
            dataset = scaler.fit_transform(dataset)


            #split into train and test sets
            
            train_size = int(len(dataset)* self.percent_train)
            test_size = int(len(dataset)) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            #reshape into X = t and Y = t+1
            self.look_back = 3 # days on the future
            trainX, trainY = self.create_dataset(train,self.look_back)
            testX, testY = self.create_dataset(test, self.look_back)



            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            nn = self.createModel()
            nn.compile(loss = 'mse', optimizer= 'adam')
            nn.fit(trainX,trainY, epochs= self.epochs, batch_size= self.batch, verbose= 1)


            # make predictions
            trainPredict = nn.predict(trainX)
            testPredict = nn.predict(testX)


            results=[]

            original_days = test[len(test)-self.look_back:,:]
            original_days,create_XPredict = self.create_setPredict(original_days)


            for i in range(self.days):

                R = nn.predict(create_XPredict)
                results.append(R)
                original_days, create_XPredict = self.create_setPredict(original_days, R)

            inverted = scaler.inverse_transform(np.concatenate(results))



            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
            print('Test Score: %.2f RMSE' % (testScore))



            # shift train predictions for plotting
            trainPredictPlot = np.empty_like(dataset)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[self.look_back:len(trainPredict)+self.look_back, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = np.empty_like(dataset)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(trainPredict)+(self.look_back*2)+1:len(dataset)-1, :] = testPredict

            newLen = dataset.shape[0]+inverted.shape[0]
            inferencePredictPlot = np.ones((newLen,1))
            inferencePredictPlot[:,:] = np.nan
            inferencePredictPlot[dataset.shape[0]:newLen, :] = inverted

            # plot baseline and predictions
            plt.plot(scaler.inverse_transform(dataset),'k', label='Original')
            plt.title(name)
            plt.plot(trainPredictPlot,'y', label='Trained_D')
            plt.plot(testPredictPlot, 'c', label='Test_D')
            plt.plot(inferencePredictPlot, 'm', label = "new days")
            plt.xlabel('days')

            plt.show()