# 2017_NCAA_basketball_prediction_Kaggle
Predict 2017 NCAA basketball result

Note: This is an on-going competation. The final result will be revealed in mid-April.

# Development environment
* Python: 2.7
* Torch: 7
* pip: sys, numpy, pandas, random, matplotlib, seaborn
* luarocks: nn, csvgo, io, optim, torch

# Data
Place the data files  [data files](https://www.kaggle.com/c/march-machine-learning-mania-2017/data) into a subfolder ./data/raw and unzip

# Algorithm design
* Processing data: neural network has great capacity of expressing data, so all the detailed features (in TourneyDetailedResults.csv and RegularSeasonDetailedResults.csv) are used to depict a team. While teams preform differenly every year, so an average is taken for counting n year's tourney matches results and m year's regular matches results. Two additional features "regular season win rate" and "tourney match number" are added for better represent a team.
* When the training and testing data are all set, a fully connected neural network is used to train the data.
* The result is shown in res.png which shows the heapmap of probability a row team wins the column team. The teams are sorted by seed number. We can find some intersing fact in the picture. 

![Alt text](..data/result/res.png "Result")

<p align="center">
  <img src="..data/result/res.png" width="350"/>
</p>

# Local Testing
$ python data_process.py 1 2 2018

$ th train.lua

