# This script generates autonomy cube in the form a csv file from the provided video and telemetry data.

# import the necessary packages
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import math
from geopy import distance
from datetime import datetime
from datetime import timedelta
import pysrt 
import xarray as xr
import time

file_location = "/18_01_2023_session_6/" # path to original flight data
file_name = "Jan-18th-2023-12-09PM-Flight-Airdata.csv" # file name of telemetry data
flight_record = pd.read_csv(file_location+file_name) # read in telemetry data

# Note on frames: 
# - Airdata exports data for every 100 ms, or 10 per second, or 600 per minute
# - DJI Air2 shoots video 30 frames per second, or 1800 per minute
# - Need to grab every 3rd frame to match Airdata

# read in directories from folder
video_files = os.listdir(file_location)
video_files = video_files[0:4]

# get frames from each video
video_frames = {keys: [] for keys in video_files}
for video_file in video_files:
    frames = os.listdir(file_location + video_file + '/frames')
    frames = frames[3::3] # get every third frame
    video_frames[video_file] = frames

# get start and end times of video from srt file
start_end_times = {keys: [] for keys in video_files}
for video_file in video_files:
    srt = pysrt.open(file_location + video_file + "/" + video_file + ".SRT")

    entry = []
    for line in srt:
        line = line.text.split("\n")
        entry.append(line)
    date = entry[0][1].split(" ")[0]
    start_time = entry[0][1].split(" ")[1].split(",") # remove date and split on comma
    start_time = date + " "+ start_time[0] + "." + start_time[1] # get ms 
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f") # convert to datetime
    last_entry = len(entry)-1
    end_time = entry[last_entry][1].split(" ")[1].split(",")
    end_time = date + " "+ end_time[0] + "." + end_time[1]
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")
    print("Video file: ", video_file, " Start time: ", start_time, " End time: ", end_time)

    # add start and end times to dict
    start_end_times[video_file] = [start_time, end_time]

# convert to correct time, AirData outputs -3 hrs
flight_record[['datetime(utc)']] = flight_record[['datetime(utc)']].apply(pd.to_datetime)
flight_record['datetime(utc)'] = flight_record['datetime(utc)'] + pd.Timedelta(hours=3)

# crop telemetry flight record to match video
flight_record_dict = {keys: [] for keys in video_files}
for video_file in video_files:
    start_time = start_end_times[video_file][0]
    end_time = start_end_times[video_file][1]
    # drop rows that are not in the video
    flight_record_cropped = flight_record[flight_record['datetime(utc)'] >= start_time]
    flight_record_cropped = flight_record_cropped[flight_record_cropped['datetime(utc)'] <= end_time]

    # drop last row(s) to match length of frames
    if len(video_frames[video_file]) != len(flight_record_cropped):
        dif = len(flight_record_cropped) - len(video_frames[video_file])
        flight_record_cropped = flight_record_cropped.iloc[:-dif]
    
    # add in video frames
    flight_record_cropped['frame'] = video_frames[video_file]

    flight_record_cropped = flight_record_cropped.drop_duplicates(subset=['datetime(utc)']) # get 1 second intervals
    flight_record_dict[video_file] = flight_record_cropped
    print("Video file: ", video_file, " Length of flight record: ", len(flight_record_dict[video_file]), 
          " Length of video: ", str(timedelta(seconds=(len(flight_record_dict[video_file])))))
    
    # get degree ranges for each of the cardinal directions 
directions = np.arange(0, 360, 45)
heading_ranges = []
for i in directions:
    heading_ranges.append((i-22.5, i+22.5))
#reassign tuple value for 45 degrees left of north
heading_ranges[0] = (heading_ranges[0][0]+360, heading_ranges[0][1])
heading_ranges

# convert gimbal heading to direction of drone
# get general direction drone is facing 
def get_direction(degree):
    # check if value is within tuple range
    if (degree >= heading_ranges[0][0] and degree <= 360) or (degree >= 0 and degree <= heading_ranges[0][1]):
        return "north"
    if (degree >= heading_ranges[1][0] and degree <= heading_ranges[1][1]):
        return "northeast"
    if (degree >= heading_ranges[2][0] and degree <= heading_ranges[2][1]):
        return "east"
    if (degree >= heading_ranges[3][0] and degree <= heading_ranges[3][1]):
        return "southeast"
    if (degree >= heading_ranges[4][0] and degree <= heading_ranges[4][1]):
        return "south"
    if (degree >= heading_ranges[5][0] and degree <= heading_ranges[5][1]):
        return "southwest"
    if (degree >= heading_ranges[6][0] and degree <= heading_ranges[6][1]):
        return "west"
    if (degree >= heading_ranges[7][0] and degree <= heading_ranges[7][1]):
        return "northwest"
    else:
        return "error"
    
# save each flight as a csv file
ac_dict = {keys: [] for keys in video_files}
cube_dict = {keys: [] for keys in video_files}
path_dict = {keys: [] for keys in video_files}

for video in flight_record_dict:
#for video_file in ['DJI_0063']:
    #print("video file: ", video_file)
    ac = flight_record_dict[video][['datetime(utc)', 'latitude', 'longitude', 'height_above_takeoff(feet)', 'gimbal_heading(degrees)', 'frame']]
    ac['height_above_takeoff(meters)'] = ac['height_above_takeoff(feet)'] * 0.3048 # convert to meters
    ac = ac.drop(columns=['height_above_takeoff(feet)']) # drop columns that are not needed
    ac['heading_direction'] = ac['gimbal_heading(degrees)'].apply(get_direction)
    ac.to_csv(file_location + video_file + "/" + video_file + "_flight_record.csv", index=False)
    ac_dict[video_file] = ac

    ac = ac.reset_index(drop=True)

    # CREATE hypercube of each video
    flight = flight_record_dict[video_file]
    fence = (flight['latitude'].min()-0.001, flight['longitude'].min()-0.001, flight['latitude'].max()+0.001, flight['longitude'].max()+0.001)
    avg_lat = (fence[0] + fence[2])/2
    avg_lon = (fence[1] + fence[3])/2

    # divide fence area into 3 m by 3 m squares
    squares_latitude = np.arange(fence[0], fence[2], 0.00003)
    squares_longitude = np.arange(fence[1], fence[3], 0.00003)
    squares = []
    for i in range(len(squares_latitude)-1):
        for j in range(len(squares_longitude)-1):
            squares.append((squares_latitude[i], squares_longitude[j], squares_latitude[i+1], squares_longitude[j+1]))
    print("Number of latitude squares: ", len(squares_latitude), "Number of longitude squares: ", len(squares_longitude))

    # create cube
    latitudes = squares_latitude # coordinate variables
    longitudes = squares_longitude # coordinate variables
    # altitude = np.array[10:30:10]
    time = np.arange(0, int(len(ac['datetime(utc)'])), 1) # coordinate variables
    heading = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'] # coordinate variables
    frames = "frame_XXXXX.jpg" # data variable

    # create empty cube
    cube = xr.DataArray(frames, coords=[latitudes, longitudes, time, heading], dims=['latitude', 'longitude', 'time', 'heading'])
    cube_dict[video_file] = cube
    # update cube with frames
    # iterate through dataframe
    # save OG path 
    path = []
    for idx, row in ac.iterrows():
        t = idx
        #print("time: ", t)
        lat, long, head = row['latitude'], row['longitude'], row['heading_direction']
        print("lat: ", lat, "long: ", long)
        ilat = list(cube.latitude.values).index(cube.sel(latitude=lat, method='nearest').latitude)
        ilon = list(cube.longitude.values).index(cube.sel(longitude=long, method='nearest').longitude)
        print("lat index: ", ilat, "long index: ", ilon)
        itime= list(cube.time.values).index(cube.sel(time=t).time)
        iheading = list(cube.heading.values).index(cube.sel(heading=head).heading)
        #print("long: ", ilon, "lat: ", ilat, "time: ", itime, "heading: ",iheading)
        cube[ilat, ilon, itime, iheading] = str(row['frame'])
        #print(cube[ilon, ilat, itime, iheading])
        #print(row['frame'])
        path.append([ilat, ilon, itime, iheading])
    path_dict[video_file] = path # save path to dictionary
    # save cube as netcdf file
    cube.to_netcdf(file_location + video_file + "/" + video_file + "_cube.nc")
    
