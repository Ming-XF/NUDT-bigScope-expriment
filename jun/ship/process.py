import pandas as pd
import numpy as np
import pickle

import pdb

lat_min = 3
lat_max = 42
lon_min = 107
lon_max = 129


data = pd.read_csv('../ship.csv', sep='\t')
# data['pos_datetime'] = pd.to_datetime(data['pos_datetime'], unit='s')

outdata = []
static = {}
for (index, (mmsi, item)) in enumerate(data.groupby('mmsi')):
    item = item.sort_values(by='pos_datetime')
    
    points = item[['aslatitude', 'longitude', 'speed', 'track', 'pos_datetime']]
    points = points[points['aslatitude'] >= lat_min]
    points = points[points['aslatitude'] <= lat_max]
    points = points[points['longitude'] >= lon_min]
    points = points[points['longitude'] <= lon_max]
    points = points[points['speed'] >= 0]
    points = points[points['speed'] <= 30]
    points = points[points['track'] >= 0]
    points = points[points['track'] <= 360]
    
    points['aslatitude'] = (points['aslatitude'] - lat_min) / (lat_max - lat_min)
    points['longitude'] = (points['longitude'] - lon_min) / (lon_max - lon_min)
    points['speed'] = points['speed'] / 30
    points['track'] = points['track'] / 360
    
    traj = {
        "mmsi": mmsi
    }
    temp = []
    for index, row in points.iterrows():
        lon = row['longitude']
        lat = row['aslatitude']
        speed = row['speed']
        direction = row['track']
        time = row['pos_datetime']
        
        if len(temp) == 0:
            temp.append([lat, lon, speed, direction, time])
        else:
            if abs(temp[len(temp) -1][4] - time) > 0.5 * 60 * 60:
                if len(temp) >= 20:
                    if len(temp) not in static:
                        static[len(temp)] = 1
                    else:
                        static[len(temp)] = static[len(temp)] + 1
                    traj['traj'] = np.array(temp)
                    outdata.append(traj)
                traj = {
                    "mmsi": mmsi
                }
                temp = []
                temp.append([lat, lon, speed, direction, time])
            else:
                temp.append([lat, lon, speed, direction, time])
    
    if len(temp) >= 20:
        if len(temp) not in static:
            static[len(temp)] = 1
        else:
            static[len(temp)] = static[len(temp)] + 1
        traj['traj'] = np.array(temp)
        outdata.append(traj)

static = sorted(static.items(),key=lambda x:x[0],reverse=True)

savedata = []
for item in outdata:
    traj = item['traj']
    mmsi = item['mmsi']
    
    trajs = traj[:int(-1 * (len(traj) % 120))]
    trajs = trajs.reshape((-1, 120, 5))
    for sub in trajs:
        temp = {
            "mmsi": mmsi,
            "traj": sub
        }
        savedata.append(temp)
    temp = {
        "mmsi": mmsi,
        "traj": traj[int(-1 * (len(traj) % 120)):]
    }
    savedata.append(temp)

fp_train = open('ship_time30_train.pkl', 'wb')
fp_test = open('ship_time30_test.pkl', 'wb')
fp_valid = open('ship_time30_valid.pkl', 'wb')
train_data = savedata[:int(0.8 * len(savedata))]
test_data = savedata[int(0.8 * len(savedata)):int(0.9 * len(savedata))]
valid_data = savedata[int(0.9 * len(savedata)):]

print("train_len: {}".format(len(train_data)))
print("test_len: {}".format(len(test_data)))
print("valid_len: {}".format(len(valid_data)))

pickle.dump(train_data, fp_train)
pickle.dump(test_data, fp_test)
pickle.dump(valid_data, fp_valid)