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

client.setalias("ekarat") # ตั้งชื่้อ


df_source = pd.DataFrame(columns=['time','x1','x2', 'x3'])
df_out = pd.DataFrame(columns=['time', 'activity'])

with open('SVM_Activity_model_window5_new_all.pkl', 'rb') as f:
    clf = pickle.load(f)

# w = window_size
def window(a, w = 5, o = 1, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
    
def preprocess_stream(source):
    df = source ####### ต้องมีการเก็บ time (column แรกที่มาจาก netpie ไว้ refer ตอน predict) แบบมี time
    df_window = pd.DataFrame(np.concatenate((window(df.time.to_numpy()), window(df.x1.to_numpy()), window(df.x2.to_numpy()), window(df.x3.to_numpy())),axis=1))
    df_window.columns = [i for i in range(len(df_window.columns))]
	
    return df_window



def callback_connect() :
    print ("Now I am connected with netpie")
    
def callback_message(topic, message) :

    data = message.split(',')
    data[1] = float(data[1])
    data[2] = float(data[2])
    data[3] = float(data[3][:-1])

    global df_out
    global df_source
    df_source = df_source.append(pd.Series(data, index=df_source.columns),ignore_index=True)
	
    print(topic, ": ", message)
	
    if len(df_source) >= 5:
        y = preprocess_stream(df_source)
		
        #use bottom 5 rows from windowed df_source when doing prediction
        activity = clf.predict(y.iloc[-1:,-15:])
        time = y.iloc[-1,2]
		
        if activity == 0:
            act = 'Running'
            print(time, act)
        elif activity == 1:
            act = 'Standing'
            print(time, act)
        elif activity == 2:
            act = 'Walking'
            print(time, act)
        
        df_out = df_out.append(pd.Series([time[2:], act], index=df_out.columns),ignore_index=True)
        df_out.to_csv('output.csv', index=False)

def callback_error(msg) :
    print("error", msg)


client.on_connect = callback_connect # แสดงข้อความเมื่อเชื่อมต่อกับ netpie สำเร็จ
client.on_message= callback_message # ให้ทำการแสดงข้อความที่ส่งมาให้
client.on_error = callback_error # หากมีข้อผิดพลาดให้แสดง
client.subscribe("/aebzab2") # ชื่อช่องทางส่งข้อมูล ต้องมี / นำหน้า และต้องใช้ช่องทางเดียวกันจึงจะรับส่งข้อมูลระหว่างกันได้
client.connect(True) # เชื่อมต่อ ถ้าใช้ True เป็นการค้างการเชื่อมต่อ client.on_message= callback_message # ให้ทำการแสดงข้อความที่ส่งมาให้
