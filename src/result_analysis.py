import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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

to_submit_res= pd.read_csv("../data/result/to_submit_result.csv", header= None)
data["SampleSubmission"]["Pred"] = to_submit_res 
data["SampleSubmission"].to_csv('../data/result/final_res.csv', index=False)

seed_filter = data['TourneySeeds']['Season'] == 2017
seed_team = data['TourneySeeds'][seed_filter]['Team']
labels_id = seed_team.values
xlabels = []
for i in labels_id:
	name =  data['Teams'][data['Teams']["Team_Id"] == i]['Team_Name'].values[0]
	xlabels.append(name)
ylabels = xlabels[::-1]
res_matrix = pd.DataFrame(np.random.randn(68, 68), columns=seed_team.values, index = seed_team.values)
for ii, row in res_matrix.iterrows():
	for jj, value in row.iteritems():
		if int(ii)<int(jj):
			match_id = '2017_'+str(ii)+'_'+str(jj)
			f = data["SampleSubmission"]['Id'] == match_id
			res_matrix[ii][jj] = 1- data["SampleSubmission"][f]['Pred']
		elif int(ii) == int(jj):
			res_matrix[ii][jj] = None
		else:
			match_id = '2017_'+str(jj)+'_'+str(ii)
			f = data["SampleSubmission"]['Id'] == match_id
			res_matrix[ii][jj] = data["SampleSubmission"][f]['Pred']

sns.set(font_scale=1.5)
sns.set_style({"savefig.dpi": 100})
ax = sns.heatmap(res_matrix,  cmap=plt.cm.Greens, linewidths=.1)
ax.xaxis.tick_top()
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
plt.xticks(rotation=80)
plt.yticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(40, 30)
fig.savefig("../data/result/res.png")



