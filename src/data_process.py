import sys
import numpy as np
import pandas as pd
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.cross_validation import train_test_split

#Prepare match detail for each seed starting from 2003, with n years' tourney, m years' regular match detail, then calculate the average

n = int(sys.argv[1])
m = int(sys.argv[2])

split_year = int(sys.argv[3]) if sys.argv[3] else 2018

data = {}
csv = ['/RegularSeasonCompactResults.csv',
'/RegularSeasonDetailedResults.csv',
'/Seasons.csv',
'/Teams.csv',
'/TourneyCompactResults.csv',
'/TourneyDetailedResults.csv',
'/TourneySeeds.csv',
'/TourneySlots.csv',
# '/sample_submission.csv',
'/SampleSubmission.csv']
path = "../data/raw"
for file in csv:
    data[file[1:-4]] = pd.read_csv(path+file)
    print('Reading '+file)

def get_regular_win_rate(team, season): 
    d = data["RegularSeasonCompactResults"][data["RegularSeasonCompactResults"]["Season"] == season]
    team_season_filter = (d["Wteam"] == team) | (d["Lteam"] == team) 
    team_win_filter = (d["Wteam"] == team) 
    return float(d[team_win_filter].shape[0])/d[team_season_filter].shape[0]

def get_match_number(team, season):
    d = data["TourneyCompactResults"][data["TourneyCompactResults"]["Season"] == season]
    team_season_filter = (d["Wteam"] == team) | (d["Lteam"] == team) 
    return d[team_season_filter].shape[0]

def get_team_detail(tp, team, season_list):
    if tp == "regular":
        d = data["RegularSeasonDetailedResults"]
    elif tp == "tourney":
        d = data["TourneyDetailedResults"]
    team_win_filter = (d["Wteam"] == team) & (d["Season"]>= season_list[0]) & (d["Season"]<= season_list[-1])
    team_lose_filter = (d["Lteam"] == team) & (d["Season"]>= season_list[0]) & (d["Season"]<= season_list[-1])
    win_detail_col = ["Wteam", "Wscore", "Wfgm", "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta",
                     "Wor", "Wdr", "Wast", "Wto", "Wstl", "Wblk", "Wpf"]
    lose_detail_col = ["Lteam", "Lscore", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta",
                      "Lor", "Ldr", "Last", "Lto", "Lstl", "Lblk", "Lpf"]
    d1 = pd.DataFrame()
    combined1 = d[team_win_filter][win_detail_col]
    combined2 = d[team_lose_filter][lose_detail_col]
    d1[win_detail_col] = combined2[lose_detail_col]
    combined = combined1.append(d1)
    temp1 = 0
    temp2 = 0
    for year in season_list:
        temp1 += get_regular_win_rate(team, year)
        temp2 += get_match_number(team, year)
    combined["win_rate"] = temp1/len(season_list)
    combined["game_number"] = temp2/len(season_list)
    return combined

def seed_to_int(seed):
    """Get just the digits from the seeding. Return as int"""
    s_int = int(seed[1:3])
    return s_int

season_filter = data["TourneySeeds"]["Season"] >= 2003+n
seed_team = data["TourneySeeds"][season_filter]
team_detail = pd.DataFrame()
for i in range(seed_team.shape[0]):
    team = seed_team.iloc[i]["Team"]
    year = seed_team.iloc[i]["Season"]
    a = get_team_detail("tourney", team, range(year-n, year))
    b = get_team_detail("regular", team, range(year+1-m, year+1))
    c = a.append(b)
    d = c.apply(np.mean)
    d["Season"] = year
    team_detail = team_detail.append(d, ignore_index=True)
# print(team_detail.tail())

team_season_detail = team_detail

print("Preparing training and testing data...")

def prepare_train_test():
    win_detail_col = ['Wast', 'Wblk', 'Wdr', 'Wfga', 'Wfga3', 'Wfgm', 'Wfgm3', 'Wfta',
                  'Wftm', 'Wor', 'Wpf', 'Wscore', 'Wstl', 'Wto', 'game_number', 'win_rate']
    lose_detail_col = ['Last', 'Lblk', 'Ldr', 'Lfga', 'Lfga3', 'Lfgm', 'Lfgm3', 'Lfta',
                  'Lftm', 'Lor', 'Lpf', 'Lscore', 'Lstl', 'Lto', 'Lgame_number', 'Lwin_rate']
    all_col = win_detail_col + lose_detail_col
    train_data = pd.DataFrame(columns = all_col)
    test_data = pd.DataFrame(columns = all_col)
    res = pd.DataFrame(columns = ["result", "predict"])
    data_filter = data["TourneyDetailedResults"]["Season"]>= 2003+n
    d = data["TourneyDetailedResults"][data_filter]
    
    for i in range(d.shape[0]):
        current_season = d.iloc[i]["Season"]
        w_filter = (team_detail["Season"] == current_season) & (team_detail["Wteam"]
                                                                     == d.iloc[i]["Wteam"])
        l_filter = (team_detail["Season"] == current_season) & (team_detail["Wteam"]
                                                                     == d.iloc[i]["Lteam"])
        
        wteam_detail = team_detail[w_filter][win_detail_col]
        lteam_detail = team_detail[l_filter][win_detail_col]

        val1 = np.append(wteam_detail.values, lteam_detail.values)
        row1 = pd.Series(val1, index = all_col)
        row1["label"] = 1
        
        val2 = np.append(lteam_detail.values, wteam_detail.values,)
        row2 = pd.Series(val2, index = all_col)
        row2["label"] = 0
        
        if current_season <= split_year:
            if random.random()>=0.5:
                train_data = train_data.append(row1, ignore_index=True)
            else:
                train_data = train_data.append(row2, ignore_index=True)
        else:
            if random.random()>=0.5:
                test_data = test_data.append(row1, ignore_index=True)
            else:
                test_data = test_data.append(row2, ignore_index=True)
    return (train_data, test_data)
(train_data, test_data) = prepare_train_test() 


train_data = shuffle(train_data)
def stand(col):
    m = min(col)
    n = max(col)
    col = (col-m)/(n-m)
    return col
train_data = train_data.apply(stand, axis = 0)
test_data = test_data.apply(stand, axis = 0)

print("Saving training and testing data")
train_data.to_csv('../data/processed/train_data.csv', index=False)
test_data.to_csv('../data/processed/test_data.csv', index=False)


def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))

def prepare_data():
    win_detail_col = ['Wast', 'Wblk', 'Wdr', 'Wfga', 'Wfga3', 'Wfgm', 'Wfgm3', 'Wfta',
                  'Wftm', 'Wor', 'Wpf', 'Wscore', 'Wstl', 'Wto', 'game_number', 'win_rate']
    lose_detail_col = ['Last', 'Lblk', 'Ldr', 'Lfga', 'Lfga3', 'Lfgm', 'Lfgm3', 'Lfta',
                  'Lftm', 'Lor', 'Lpf', 'Lscore', 'Lstl', 'Lto', 'Lgame_number', 'Lwin_rate']
    all_col = win_detail_col + lose_detail_col
    test_data = pd.DataFrame(columns = all_col)

    for ii, row in data["SampleSubmission"].iterrows():

        year, t1, t2 = get_year_t1_t2(row.Id)

        current_season = year
        w_filter = (team_detail["Season"] == current_season) & (team_detail["Wteam"]
                                                                     == t1)
        l_filter = (team_detail["Season"] == current_season) & (team_detail["Wteam"]
                                                                     == t2)
        wteam_detail = team_detail[w_filter][win_detail_col]
        lteam_detail = team_detail[l_filter][win_detail_col]
        
        val1 = np.append(wteam_detail.values, lteam_detail.values)
        row1 = pd.Series(val1, index = all_col)
        
        test_data = test_data.append(row1, ignore_index=True)
    return test_data
to_submit = prepare_data()


to_submit = to_submit.apply(stand, axis = 0)
to_submit.to_csv('../data/processed/to_submit.csv', index=False)
