import pandas as pd
import numpy as np
import pickle

import pdb

data = pd.read_csv('../ship.csv', sep='\t')
# data['pos_datetime'] = pd.to_datetime(data['pos_datetime'], unit='s')

lat_min = 3
lat_max = 42
lon_min = 107
lon_max = 129

outdata = []
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
    
    points['aslatitude'] = (points['aslatitude'] - lat_min) / lat_max
    points['longitude'] = (points['longitude'] - lon_min) / lon_max
    points['speed'] = points['speed'] / 30
    points['track'] = points['track'] / 360
    
    points = points.values
    
#     temp = points[:, :4][points[:, :4] <= 0]
    
    points = points[:int(-1 * (len(points) % 20))]
    points = points.reshape((-1, 20, 5))
    
    for point in points:
        traj = {
            "mmsi": mmsi,
            "traj": point
        }
        outdata.append(traj)

# data.to_csv('./data.csv', index=False)
fp_train = open('ship_len20_train.pkl', 'wb')
fp_test = open('ship_len20_test.pkl', 'wb')
fp_valid = open('ship_len20_valid.pkl', 'wb')
pickle.dump(outdata[:int(0.8 * len(outdata))], fp_train)
pickle.dump(outdata[int(0.8 * len(outdata)):int(0.9 * len(outdata))], fp_test)
pickle.dump(outdata[int(0.9 * len(outdata)):], fp_valid)