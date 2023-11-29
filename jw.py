import pandas as pd
import numpy as np
import pickle
import os

import pdb


def sample(traj, rate):
    output = []
    for index, item in enumerate(traj):
        if index == 0:
            output.append([item])
        else:
            last = output[-1][-1]
            last_time = last[-1]
            next_time = item[-1]
            
            if abs((next_time - last_time) - rate) < (rate / 3):
                output[-1].append(item)
            else:
                if ((next_time - last_time) - rate) < 0:
                    continue
                else:
                    output.append([item])
    return output
                


# SOG = 'speed'
# COG = 'track'
# LON = 'longitude'
# LAT = 'latitude'
# TIME = 'pos_datetime'
# FL = 'flight_line'
# ID = 'mmsi'
# PATH = "./data_jw/vessel"
# rate = 60*30 #采样间隔
# save_name = 'ship'
# sog_throd = -1

SOG = 'speed'
COG = 'track'
LON = 'longitude'
LAT = 'latitude'
TIME = 'pos_datetime'
FL = 'flight_line'
ID = 'aircraft_icao'
PATH = "./data_jw/aire"
rate = 60 * 2.5
save_name = 'aire'
sog_throd = -1

###########################按照轮船读取轨迹##########################
data = []
for filename in os.listdir(PATH):
    df = pd.read_csv(os.path.join(PATH, filename))
    df = df[[ID, FL, LON, LAT, SOG, COG, TIME]]

    for (index, (fl, item)) in enumerate(df.groupby(FL)):
        name = fl.split("_")[0]
        item = item[[LAT, LON, SOG, COG, TIME]]
        item = item.sort_values(by=TIME, ascending=True)
        temp = item.values
        
        data.append({'mmsi': name, 'traj': temp})
###########################按照轮船读取轨迹#########################

##########################采样,重置异常点###########################
data2 = []
for item in data:
    name = item['mmsi']
    traj = item['traj']
    trajs = sample(traj, rate)
    for item2 in trajs:
        
        if len(item2) >= 20:
            item3 = np.array(item2)
            
            cog = item3[:, 3]
            cog[cog < 0] = 0
            cog[cog > 360] = 360
            item3[:, 3] = cog
            
            sog = item3[:, 2]
            sog[sog < 0] = 0
            item3[:, 2] = sog
            
            lat = item3[:, 0]
            lat[lat < -90] = -90
            lat[lat > 90] = 90
            item3[:, 0] = lat
            
            lon = item3[:, 1]
            lon[lon < -180] = -180
            lon[lon > 180] = 180
            item3[:, 1] = lon
            
            last_lat = item3[:-1, 0]
            next_lat = item3[1:, 0]
            delt_lat = abs(next_lat - last_lat)
            delt_lat = np.mean(delt_lat)
            
            last_lon = item3[:-1, 1]
            next_lon = item3[1:, 1]
            delt_lon = abs(next_lon - last_lon)
            delt_lon = np.mean(delt_lon)
            
            sog = item3[:, 2]
            sog_mean = np.mean(sog)
            if sog_mean > sog_throd:
                data2.append({'mmsi': name, 'traj': item3, 'delt_lat': delt_lat, 'delt_lon': delt_lon})

################################采样###############################

############################获取参数###############################
all_traj = []
delts_lat = []
delts_lon = []
for item in data2:
    all_traj.append(item['traj'])
    delts_lat.append(item['delt_lat'])
    delts_lon.append(item['delt_lon'])
all_traj = np.concatenate(all_traj, axis=0)
lat_min = all_traj[:, 0].min()
lat_max = all_traj[:, 0].max()
lon_min = all_traj[:, 1].min()
lon_max = all_traj[:, 1].max()
sog_min = all_traj[:, 2].min()
sog_max = all_traj[:, 2].max()
cog_min = all_traj[:, 3].min()
cog_max = all_traj[:, 3].max()
delt_lat_mean = np.mean(delts_lat)
delt_lon_mean = np.mean(delts_lon)
###################################################################


########################适配模型#####################################
data3 = []
for item in data2:
    name = item['mmsi']
    traj = item['traj']
    
    traj[:, 0] = (traj[:, 0] - lat_min) / (lat_max - lat_min)
    traj[:, 1] = (traj[:, 1] - lon_min) / (lon_max - lon_min)
    traj[:, 2] = (traj[:, 2] - sog_min) / (sog_max - sog_min)
    traj[:, 3] = (traj[:, 3] - cog_min) / (cog_max - cog_min)
    
    data3.append({'mmsi': name, 'traj': traj})
#####################################################################

part = int(0.1*len(data3))
test = data3[:part]
train = data3[part:]
# train = data3
# test = data3


print('train len : ' + str(len(train)))
print('test len : ' + str(len(test)))
print('lat min : ' + str(lat_min))
print('lat max : ' + str(lat_max))
print('lon min : ' + str(lon_min))
print('lon max : ' + str(lon_max))
print('sog min : ' + str(sog_min))
print('sog max : ' + str(sog_max))
print('cog min : ' + str(cog_min))
print('cog max : ' + str(cog_max))
print('delt lat mean : ' + str(delt_lat_mean))
print('delt lon mean : ' + str(delt_lon_mean))

with open('{}_train.pkl'.format(save_name), 'wb') as f:
    pickle.dump(train, f)
    
with open('{}_test.pkl'.format(save_name), 'wb') as f:
    pickle.dump(test, f)
    
    
# MMSI time lat lon vel cou poi 