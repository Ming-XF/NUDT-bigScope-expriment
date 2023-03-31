
from flask import Flask, request
import pre_config
from flask_sqlalchemy import SQLAlchemy

# from predict import Prediction
# import numpy as np

db = SQLAlchemy()
app = Flask(__name__)
app.config.from_object(pre_config) # 加载数据库配置文件
db.init_app(app)

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/predict/<idx>')
def login(idx):
    idx = int(idx)
    print(idx)
    sql = "select * from trace_info where id = " + str(idx)
    print(sql)
    obj = db.session.execute(sql)
    # SQLAlchemy对象转dict
    data = [dict(zip(result.keys(), result)) for result in obj]
    dictTemp =data[0]
    dictTemp['trace_long_lat_dots'] = str(dictTemp['trace_long_lat_dots'], 'utf-8')
    del dictTemp['id']
    
    print(dictTemp)

    # 模型获取dictTemp中的经纬度等信息
    # 模型在此处运行
    #prediction = Prediction()
    #pred = prediction.predict(history, predict_len=50)
    '''
    注意运行前要在config.py中配置区域范围大小（lat_min，lat_max，lon_min，lon_max）
    history: numpy数组，就是历史轨迹点二维数组
    predict_len：需要预测多少个轨迹点
    pred: numpy数组，历史轨迹点加预测的轨迹点二维数组
    '''
    
    # 模型获取dictTemp中的经纬度等信息，操作后，将新的经纬度COG,SOG等覆盖trace_long_lat_dots的值


    listTemp = []
    for value in dictTemp.values():
        value = '\''+str(value)+'\''
        listTemp.append(value)


    STR=','.join('%s' % id for id in listTemp)

    sql2 = "insert into "+"trace_info"+"(trace_name,trace_long_lat_dots,trace_infomations,target_type,target_vehicle_name)"+"values("+STR+")"

    obj2 = db.session.execute(sql2)
    db.session.commit()

    print(obj2.lastrowid)

    sql = "select * from trace_info where id =" + str(obj2.lastrowid)
    obj = db.session.execute(sql)
    # SQLAlchemy对象转dict
    data2 = [dict(zip(result.keys(), result)) for result in obj]
    dictTemp2 = data2[0]
    dictTemp2['trace_long_lat_dots'] = str(dictTemp2['trace_long_lat_dots'], 'utf-8')


    returnResult ={}
    returnResult['data']=dictTemp2
    returnResult['code']=200
    returnResult['msg']="success"


    return returnResult


if __name__ == '__main__':
    app.run("0.0.0.0",5005)