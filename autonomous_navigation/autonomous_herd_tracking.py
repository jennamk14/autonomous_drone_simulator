# This script autonomous tracks a herd of zebras using a drone. 
# The drone uses a policy to navigate autonomously using outputs from YOLO.

# Libraries
import pandas as pd
from ultralytics import YOLO
import os
import xarray as xr
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# load object detection model
model = YOLO("yolov8n.pt")

# define autonomous navigation functions
def next_action(results):
    """
    Autonomous navigation policy for drone to track herd of zebras
    Input: results from YOLO model run on video frame
    Output: direction of drone movement in x, y, z direction
    """
    centroid_camera = (results[0].orig_shape[1]/2, results[0].orig_shape[0]/2)
    px = pd.DataFrame((results[0].boxes.boxes).numpy(), columns = ('x1', 'y1','x2', 'y2', 'confidence', 'class'))

    # get x, y, w, h for results and convert to dataframe
    pxywh = pd.DataFrame((results[0].boxes.xywh).numpy(), columns = ('x', 'y','w', 'h'))
    px = px.join(pxywh)

    bbox_sizes = []

    # get centroid of herd
    centroid_herd = (px['x'].mean(), px['y'].mean())

    image_shape_h, image_shape_w = results[0].orig_shape
    x_center_range = image_shape_w/2 - image_shape_w/8, image_shape_w/2 + image_shape_w/8
    y_center_range = image_shape_h/2 - image_shape_h/8, image_shape_h/2 + image_shape_h/8

    # calculate differencee between centroid of herd and camera
    dif_x = centroid_herd[0] - centroid_camera[0]
    dif_y = centroid_herd[1] - centroid_camera[1]

    # get the middle 75% of the image, i.e. 12.5% on each side
    left_range = results[0].orig_shape[1]/8
    right_range = results[0].orig_shape[1] - left_range
    top_range = results[0].orig_shape[0]/8
    bottom_range = results[0].orig_shape[0] - top_range

    # get range of x values for herd
    x_min_herd, x_max_herd = px['x1'].min(), px['x2'].max()
    y_min_herd, y_max_herd = px['y1'].min(), px['y2'].max()

    # Calculate next move for drone in x, y, z direction

    # navigation policy: move x, y, z, yaw until herd is in center of camera frame, keep checking every 1 sec to adjust
    # continuous adjustments allows us to avoid complex calculations and avoid overshooting
    # check if value not in range of tuple
    direction_x = "No movement in x-axis"
    direction_y = "No movement in y-axis"
    direction_z = "No movement in z-axis"
    
    if (centroid_herd[0] < x_center_range[0]) | (centroid_herd[0] > x_center_range[1]):
        if dif_x > 0:
            #print("y-axis: Move right")
            direction_x = "Move right"
        elif dif_x < 0:
            #print("y-axis: Move left")
            direction_x = "Move left"
        else:
            #print("y-axis: No movement in y-axis")
            direction_x = "No movement in x-axis"
    else:
        #print("y-axis: No movement in y-axis")
        direction_x = "No movement in x-axis"

    # if no movement left or right, move forward or backward
    if direction_x == "No movement in x-axis":
        # check to see if herd is in center 75% of camera frame
        if (x_min_herd >= left_range) | (x_max_herd <= right_range):
            #print("x-axis: Move forward")
            direction_y = "Move forward"
        elif (x_min_herd <= left_range) | (x_max_herd >= right_range):
            #print("x-axis: Move backward")
            direction_y = "Move backward"
    else:
        #print("No movement in x-axis")
        direction_y = "No movement in y-axis"

    # note: y-axis in image is actually z-axis in drone; y-axis in image is inverted (0,0 is top left corner)
    if (centroid_herd[1] < y_center_range[0]) | (centroid_herd[1] > y_center_range[1]):
        if (dif_y >= 0.0) & (y_min_herd >= bottom_range):
            #print("z-axis: Move down")
            direction_z = "Move down"
        elif (dif_y <= 0.0) & (y_max_herd <= bottom_range):
            #print("z-axis: Move up")
            direction_z = "Move up"
        else:
            #print("No movement in z-axis")
            direction_z = "No movement in z-axis"
    else:
        #print("No movement in z-axis")
        direction_z = "No movement in z-axis"
    return  direction_x, direction_y, direction_z, bbox_sizes

def update_index(cube, ilat, ilon, itime, iheading, direction_y, direction_x, direction_counts):
    """
    Updates the index of the drone in the autonomy cube based on the direction of movement
    """

    # get current heading
    heading = cube.isel(latitude=ilat, longitude=ilon, time=itime, heading=iheading).heading.values

    # direction counter 
    if heading in ['northwest', 'north', 'northeast']:
        if direction_x != "No movment in -axis":
            if direction_x == "Move right":
                direction_counts["Move right"] = direction_counts["Move right"] + 1
                if direction_counts["Move right"] > 5:
                    ilat = ilat + 1
                    direction_counts["Move right"] = 0 # reset counter
            elif direction_x == "Move left":
                if direction_counts["Move left"] > 5:
                    ilat = ilat - 1
                    direction_counts["Move left"] = 0
        if direction_y != "No movment in y-axis":
            if direction_y == "Move forward":
                if direction_counts["Move forward"] > 10:
                    ilon = ilon + 1
                    direction_counts["Move forward"] = 0
            elif direction_y == "Move backward":
                if direction_counts["Move backward"] > 10:
                    ilon = ilon - 1
                    direction_counts["Move backward"] = 0

    if heading in ['northeast', 'east', 'southeast']:
        if direction_y != "No movment in y-axis":
            if direction_y == "Move right":
                ilon = ilon + 1
            elif direction_y == "Move left":
                ilon = ilon - 1
        if direction_x != "No movment in x-axis":
            if direction_x == "Move forward":
                ilat = ilat + 1
            elif direction_x == "Move backward":
                ilat = ilat - 1

    if heading in ['southwest', 'south', 'southeast']:
        if direction_y != "No movment in y-axis":
            if direction_y == "Move right":
                ilat = ilat + 1
            elif direction_y == "Move left":
                ilat = ilat - 1
        if direction_x != "No movment in x-axis":
            if direction_x == "Move forward":
                ilon = ilon - 1
            elif direction_x == "Move backward":
                ilon = ilon + 1

    if heading in ['southwest', 'west', 'northwest']:
        if direction_y != "No movment in y-axis":
            if direction_y == "Move right":
                ilon = ilon + 1
            elif direction_y == "Move left":
                ilon = ilon - 1
        if direction_x != "No movment in x-axis":
            if direction_x == "Move forward":
                ilat = ilat - 1
            elif direction_x == "Move backward":
                ilat = ilat + 1        

    # increase time 1 step
    itime = itime 

    iheading = iheading

    return ilat, ilon, itime, iheading

def get_starting_location(autonomy_cube):
    """
    Returns the starting location of the drone
    """
    return autonomy_cube.iloc[0]["latitude"], autonomy_cube.iloc[0]["longitude"], autonomy_cube.iloc[0]["height_above_takeoff(meters)"], autonomy_cube.iloc[0]["heading_direction"]

def get_starting_index(cube, autonomy_cube):
    """
    Returns the starting index of the drone
    """
    lat, lon, alt, heading = get_starting_location(autonomy_cube)
    time = 0
    ilat = list(cube.latitude.values).index(cube.sel(latitude=lat, method='nearest').latitude)
    ilon = list(cube.longitude.values).index(cube.sel(longitude=lon, method='nearest').longitude)
    itime= list(cube.time.values).index(cube.sel(time=time).time)
    iheading = list(cube.heading.values).index(cube.sel(heading=heading).heading)
    return ilat, ilon, itime, iheading

def get_location(autonomy_cube, index):
    """
    Returns the location of the drone
    """
    return autonomy_cube.iloc[index]["latitude"], autonomy_cube.iloc[index]["longitude"], autonomy_cube.iloc[index]["height_above_takeoff(meters)"], autonomy_cube.iloc[index]["heading_direction"]

def get_index(cube, autonomy_cube, index):
    """
    Returns the index of the drone
    """
    lat, lon, alt, heading = get_location(autonomy_cube, index)
    time = index
    ilat = list(cube.latitude.values).index(cube.sel(latitude=lat, method='nearest').latitude)
    ilon = list(cube.longitude.values).index(cube.sel(longitude=lon, method='nearest').longitude)
    itime= list(cube.time.values).index(cube.sel(time=time).time)
    iheading = list(cube.heading.values).index(cube.sel(heading=heading).heading)
    return ilat, ilon, itime, iheading

def autonomous_navigation(image, t, cube, autonomy_cube, direction_counts):
    """
    Autonomous navigation policy for drone to track herd of zebras
    Input: image from drone camera
    Output: direction of drone movement in x, y, z direction
    """
    # crop image
    im = Image.open(image)
    width, height = im.size
    width_third = int(width/3)
    height_quarter= int(height/4)

    image = cv2.imread(image)
    f = image[int(height_quarter):int(height_quarter*3), int(width_third):int(width_third*2)]

    # run object detection model to detect zebras
    results = model.predict(f)

    # get current index
    ilat, ilon, itime, iheading = get_index(cube, autonomy_cube, t)
    clt, cil, cit, cih  =  ilat, ilon, itime, iheading 
    
    #print("Current index: ", ilat, ilon, itime, iheading)
    current_loc = [ilat, ilon, itime, iheading]

    # if zebras (or horses) are detected in the frame
    if (22 in results[0].boxes.cls or 23 in results[0].boxes.cls):
        direction_x, direction_y, direction_z, bboxes = next_action(results)
        #print("direction_y: ", direction_y, "   direction_x: ", direction_x, "  direction_z: ", direction_z)

        # are the bboxes at least 100 px by 100 px?
        for i in range(len(results[0].boxes.xywh)):
            # check if bbox is less than 100 px by 100 px since this is minimum required to infer behavior
            if results[0].boxes.xywh[i][2]*results[0].boxes.xywh[i][3] < 100*100:
                print("bbox too small")

        ilat, ilon, itime, iheading = update_index(cube, ilat, ilon, itime, iheading, direction_y, direction_x, direction_counts)
        # print("Updated index: ", ilat, ilon, itime, iheading)
            
        lat, log, alt, heading = get_index(cube, autonomy_cube, t+1)
        # print("Next index: ", lat, log, alt, heading)
            
        # get_location
        latitude, longitude, altitude, direction = get_location(autonomy_cube, t+1)
        # print("Next location: ", latitude, longitude, altitude, direction)

        # keep altitude at 18 meters +/- 8 meters (10 to 26 meters) for behavior inference
        # Ref: Integrating Biological Data into Autonomous Remote Sensing Systems for In Situ Imageomics: A Case Study for Kenyan Animal Behavior Sensing with Unmanned Aerial Vehicles (UAVs)
        if altitude > 26:
            # move down
            altitude = 26
        elif altitude < 10:
            # move up
            altitude = 10
    else:
        # if no zebras detected, move forward
        direction_x = "No movement in x-axis"
        direction_y = "Move forward"
        direction_z = "No movement in z-axis"
        ilat, ilon, itime, iheading = update_index(cube, ilat, ilon, itime, iheading, direction_y, direction_x, direction_counts)
        # print("Updated index: ", ilat, ilon, itime, iheading)
    
    # Check if the drone is at the correct location and heading
    if clt == ilat and cil == ilon and cit == itime and cih == iheading:
        prediction = True
    else:
        prediction = False
    if clt == lat and cil == log and cit == itime and cih == heading:
        correct = True
    else:
        correct = False

    return prediction, correct, current_loc, [lat, log, alt, heading]