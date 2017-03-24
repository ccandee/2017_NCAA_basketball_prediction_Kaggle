import sys
import numpy as np
import pandas as pd
import matplotlib

data = {}
csv = ['/RegularSeasonCompactResults.csv',
'/RegularSeasonDetailedResults.csv',
'/Seasons.csv',
'/Teams.csv',
'/TourneyCompactResults.csv',
'/TourneyDetailedResults.csv',
'/TourneySeeds.csv',
'/TourneySlots.csv',
'/sample_submission.csv',
      '/SampleSubmission.csv']
path = "../data/raw"
for file in csv:
    data[file[1:-4]] = pd.read_csv(path+file)
    print('Reading '+file)


to_submit_res= pd.read_csv("../data/result/to_submit_result.csv", header= None)
data["SampleSubmission"]["Pred"] = to_submit_res 
data["SampleSubmission"].to_csv('../data/result/final_res.csv', index=False)
