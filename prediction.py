# Description: LSTM model of Machine Learning will be used to predict the closing stock price of a corporation (Apple INC.) using the past 60 day stock price.
#libraries
import math
import numpy as np
import pandas as pandas
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

#Import data
df=web.DataReader('AAPL', data_source='yahoo', start='2007-01-01', end='2020-01-01')
print(df.shape)

#Important for visualizing
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

#Just close column
data=df.filter(['Close'])
#numpy array
dataset=data.values
#to train the model on we have to get the number of rows
training_data_len=math.ceil(len(dataset)*.8)
print(training_data_len)

#Scale the data - transformation needed for bulding later model
scaler=MinMaxScaler(feature_range=(0,1)) #scaled between 0 and 1
scaled_data=scaler.fit_transform(dataset)
print(scaled_data)

#Create Training data set, scaled data set
train_data=scaled_data[0:training_data_len , :]
#Split the data into x_train and y_train data sets
x_train= []
y_train= [] #depended one

for i in range(60, len(train_data)):
	x_train.append(train_data[i-60:i,0])
	y_train.append(train_data[i, 0])
	if i<=60: #past 60 days
		print(x_train)
		print(y_train)
		print()
#Convert the x and y train to np arrays
x_train=np.array(x_train)
y_train=np.array(y_train)

#Reshaping data because of LSTM model. Our formed x and y train are twodimensional and LSTM model expects threedimensional.
print(x_train.shape)
print(y_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the LSTM model - model architecture
model= Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Creating now testing data set
#Create a new array containing scaled value from index 1543 to 2007
test_data=scaled_data[training_data_len-60:, :]
#Create the data sets x_test and y_test
x_test= []
y_test= dataset[training_data_len:, :] #all of the values we want our model to predict
for i in range(60, len(test_data)):
	x_test.append(test_data[i-60:i, 0])

#Convert the data to a np
x_test=np.array(x_test)
#Reshape the data
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

#Get the root mean squared error
rmse=np.sqrt(np.mean( predictions-y_test )**2 )
print(rmse)

#Plot the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')

#Show the valid and predicted prices
print(valid)

plt.show()