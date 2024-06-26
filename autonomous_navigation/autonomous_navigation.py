# Main script for simulating autonomous drone navigation.
# The script uses the autonomous_herd_tracking.py script to simulate the autonomous navigation policy.
# First, the autonomy cube data is loaded, and iterates through the data to test the autonomous navigation policy.
# Next, the script calculates the accuracy and f1 score of the autonomous navigation policy.

# Libraries
import xarray as xr
import pandas as pd
import cv2
import autonomous_herd_tracking as aht # Replace with your own path

# Load the autonomy cube data to test the autonomous navigation policy on
# Update the file location and video file name to test the autonomous navigation policy on a different video
file_location = 'data/'
video_file = 'DJI_0063'
frame_location = '/data/18_01_2023_session_6/' + video_file + '/frames/'
autonomy_cube = pd.read_csv(file_location + video_file + "/" + video_file + "_flight_record.csv")
cube = xr.open_dataset(file_location + video_file + "/" + video_file + "_cube.nc")

# Count number of times autonomous navigation model makes correst prediction
count_correct = 0
count_wrong = 0
direction_counts = {'Move forward': 0, 'Move backward': 0, 'Move left': 0, 'Move right': 0}

# Select range of data to test the autonomous navigation policy on
duration = 20

# create a dataframe to store the results
df1 = pd.DataFrame(columns=['pred', 'correct'])
results = pd.DataFrame(columns=['# of actions that differ', '% of frames with differing actions'])
count_correct = 0
count_wrong = 0

for t in range(duration):
    itime = t
    time = t
    frame = autonomy_cube.iloc[itime]['frame']
    #image = cv2.imread(frame)

    # get the prediction and correct action from the selected autonomous navigation policy
    prediction, correct, current_loc, [lat, log, alt, heading] = aht.autonomous_navigation(frame, t, cube, autonomy_cube, direction_counts)

    df1.loc[len(df1)] = [prediction, correct]

    ilat, ilon, itime, iheading = current_loc

    if ilat == lat and ilon == log and itime == time and iheading == heading:
       # print("Correct location.")
        count_correct += 1
    else:    
        # Uncomment for debugging
        # print("Incorrect location.")
        # print("Correct location: ", lat, log, time, heading)
        # print("Current location: ", ilat, ilon, itime, iheading)
        count_wrong += 1

# calculate accuracy of autonomous navigation policy
accuracy = count_correct / (count_correct + count_wrong)
results.loc[len(results)] = [count_wrong, accuracy]

# calculate f1 score of autonomous navigation policy
tp = len(df1[(df1['pred']==True) & (df1['correct']==True)])
tn = len(df1[(df1['pred']==False) & (df1['correct']==False)])
fp = len(df1[(df1['pred']==True) & (df1['correct']==False)])
fn = len(df1[(df1['pred']==False) & (df1['correct']==True)])

precision =  tp/(tp+fp)
recall = tp/(tp+fn)
print("precision: ", tp/(tp+fp))
print("recall: ", tp/(tp+fn))

f1 = 2 *(precision*recall)/(precision + recall)
print("f1: ", f1)

# store stats to dataframe
stats = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
stats.loc[len(stats)] = [accuracy, precision, recall, f1]
stats.to_csv('results.csv')