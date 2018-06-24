import os
import numpy as np
import pandas as pd
from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

##############
# Load event #
##############

event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('../input/train_1', event_prefix))

mem_bytes = (hits.memory_usage(index=True).sum() 
             + cells.memory_usage(index=True).sum() 
             + particles.memory_usage(index=True).sum() 
             + truth.memory_usage(index=True).sum())
print('{} memory usage {:.2f} MB'.format(event_prefix, mem_bytes / 2**20))

#######################
# Plot Hits Per Track #
#######################

n_track_hits = [sum(particles['nhits']==i)*i for i in range(20)]

plt.plot(range(20), [i/len(hits) for i in n_track_hits])
plt.title("NHits Distribution")
plt.xlabel("NHits in Track")
plt.ylabel("Fraction of Total Hits")
plt.show()

plt.plot(range(20), np.cumsum(n_track_hits)/len(hits))
plt.title("Cumulative NHits Distribution")
plt.xlabel("NHits in Track")
plt.ylabel("Fraction of Total Hits")
plt.show()

print(len(truth.loc[truth.particle_id == 0]))
print(len(truth.loc[truth.particle_id != 0]))
print(len(hits))

###################
# dR Between Hits #
###################

# only look at particles with the longest tracks
long_particles = particles.loc[particles.nhits >= 15]
long_truth = truth.loc[truth.particle_id.isin(long_particles['particle_id'].values)]
long_hits = hits.loc[hits.hit_id.isin(long_truth.hit_id)]

distance_same = []
distance_different = []
for x1, y1, z1, truth1 in zip(long_hits['x'], long_hits['y'], long_hits['z'], long_truth['particle_id']):
    for x2, y2, z2, truth2 in zip(long_hits['x'], long_hits['y'], long_hits['z'], long_truth['particle_id']):
        if (truth1==truth2):
            distance_same.append((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        else:
            distance_different.append((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

plt.hist(distance_same, bins=np.arange(0, 10, 0.1))
plt.show()
plt.hist(distance_different, bins=np.arange(0, 10, 0.1))
plt.show()

###############
# Hit Modules #
###############

modules = hits.groupby(truth['particle_id'])['module_id']
n_hits = list(modules.agg(['count'])['count'].values)
unique_modules = modules.unique()
n_unique_modules = [len(i) for i in unique_modules]
sns.jointplot(x='Hits in Track', y='Unique Modules in Track', data=pd.DataFrame({"Hits in Track":n_hits[1:], "Unique Modules in Track":n_unique_modules[1:]}))
sns.heatmap(data=pd.DataFrame({"Hits in Track":n_hits[1:], "Unique Modules in Track":n_unique_modules[1:]}))

modules = long_hits.groupby(long_truth['particle_id'])['module_id']
n_hits = list(modules.agg(['count'])['count'].values)
unique_modules = modules.unique()
n_unique_modules = [len(i) for i in unique_modules]
sns.jointplot(np.array(n_hits[1:]), np.array(n_unique_modules[1:]), kind='kde')
