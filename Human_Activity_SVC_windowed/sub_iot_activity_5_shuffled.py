import microgear.client as client
import logging
import time
import pickle
import pandas as pd
import numpy as np

appid = "ekaratnida"
gearkey = 'jtD9ag08syPtqiK' # key
gearsecret = 'vDEEIuw9Ssj4OvbrBHmM4hZfa' # secret

client.create(gearkey,gearsecret,appid,{'debugmode': True}) # สร้างข้อมูลสำหรับใช้เชื่อมต่อ

client.setalias("ekarat") # microgear name

#create blank dataframe for source and output data
df_source = pd.DataFrame(columns=['time','x1','x2', 'x3'])
df_out = pd.DataFrame(columns=['time', 'activity'])

#select trained model
with open('SVM_Activity_model_window5_new_all_shuffled.pkl', 'rb') as f:
    clf = pickle.load(f)

#create function that form slding window data in batch where w = window_size, o = striding unit
def window(a, w = 5, o = 1, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view

#preprocess raw source data into sliding window data
def preprocess_stream(source):
    df = source 
    df_window = pd.DataFrame(np.concatenate((window(df.time.to_numpy()), window(df.x1.to_numpy()), window(df.x2.to_numpy()), window(df.x3.to_numpy())),axis=1))
    df_window.columns = [i for i in range(len(df_window.columns))]
	
    return df_window


def callback_connect() :
    print ("Now I am connected with netpie")
    
def callback_message(topic, message) :

	#take input message data and chagne data type
    data = message.split(',')
    data[1] = float(data[1])
    data[2] = float(data[2])
    data[3] = float(data[3][:-1])

    #use global parameter
    global df_out
    global df_source
    df_source = df_source.append(pd.Series(data, index=df_source.columns),ignore_index=True)
	
    print(topic, ": ", message, end=' - ')
	
    if len(df_source) >= 5:
        y = preprocess_stream(df_source)
		
        #use bottom 5 rows from windowed df_source when doing prediction
        activity = clf.predict(y.iloc[-1:,-15:])
        time = y.iloc[-1,2]
		
        if activity == 0:
            act = 'Running'
            print(act)
        elif activity == 1:
            act = 'Standing'
            print(act)
        elif activity == 2:
            act = 'Walking'
            print(act)
        
        df_out = df_out.append(pd.Series([time[2:], act], index=df_out.columns),ignore_index=True)
        df_out.to_csv('output.csv', index=False)

def callback_error(msg) :
    print("error", msg)


client.on_connect = callback_connect # display successfully connected message with connected to netpie
client.on_message= callback_message # display received message sent from netpie's publisher
client.on_error = callback_error # display this when error occured
client.subscribe("/bads") # netpie's publisher topic to subscribed
client.connect(True) # เwhen True -- client.on_message= callback_message 
