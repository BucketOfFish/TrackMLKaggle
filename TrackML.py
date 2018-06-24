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

#####################
# Polar coordinates #
#####################

x = hits.x.values
y = hits.y.values
z = hits.z.values

r = np.sqrt(x**2 + y**2)
theta = np.arctan(r/z)%np.pi
hits['eta'] = -np.log(np.tan(theta/2))
hits['phi'] = np.arctan(y/x)
hits['R'] = r

###############
# Plot tracks #
###############

tracks = truth.particle_id.unique()[1::10]
fig = plt.figure(figsize=(100,35))
ax = fig.add_subplot(121)#, projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['eta', 'phi', 'R']]
    if(t.phi.max()-t.phi.min()) < 1:
        ax.plot(t.phi, t.eta, ".-")
#ax.set_xlabel("R")
#ax.set_ylabel("eta")
#ax.set_zlabel("phi")
plt.show()

#####################
# Examining C shift #
#####################

# tracks = truth.loc[abs(hits.eta)<1].particle_id.unique()[1::200]
# def plot_tracks(C):
    # fig = plt.figure(figsize=(20,7))
    # ax = fig.add_subplot(121)#, projection='3d')
    # for track in tracks:
        # hit_ids = truth[truth['particle_id'] == track]['hit_id']
        # t = hits[hits['hit_id'].isin(hit_ids)][['eta', 'phi', 'R']]
        # if(t.phi.max()-t.phi.min()) < 1:
            # ax.plot(t.phi-C*t.R, t.eta, ".-")
    # #ax.set_xlabel("R")
    # #ax.set_ylabel("eta")
    # #ax.set_zlabel("phi")
    # plt.show()
    # print(C)

# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
# interact(plot_tracks, C=(-0.002,0.002,0.00001))

##############
# Clustering #
##############

from sklearn.cluster import DBSCAN
from sklearn import metrics

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = -1
remaining_hits = hits
n_clusters_found = 0

def find_clusters(min_points, max_radius, shift, phi_wraparound=False, plot_intermediate=False):
    global clustering, remaining_hits, n_clusters_found
    remaining_hits['phiCR'] = remaining_hits['phi'] - shift*remaining_hits['R']
    X = remaining_hits[['eta', 'phiCR']]

    eps = max_radius
    min_samp = min_points
    db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
    labels = db.labels_
    labels = [i+n_clusters_found if i!=-1 else -1 for i in labels]
    if max(labels)>-1:
        n_clusters_found = max(labels)+1
    remaining_hits['track_id'] = labels
    clustering.update(remaining_hits['track_id'])
    remaining_hits = remaining_hits[remaining_hits.track_id==-1]

    # plot currently found clusters
    if (plot_intermediate):
        hits['phiCR'] = hits['phi'] - shift*hits['R']
        fig = plt.figure(figsize=(20,7))
        ax = fig.add_subplot(111)
        clusters = np.unique(clustering['track_id'])
        for cluster in clusters:
            cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
            t = hits[hits['hit_id'].isin(cluster_hit_ids)][['eta', 'phiCR']]
            if cluster != -1:
                ax.plot(t.phiCR, t.eta, '.-', ms=10)
        plt.show()

    # print score
    score = score_event(truth, clustering)
    print('track-ml custom metric score:', round(score, 4), '- %d hits remaining to match' % len(remaining_hits), '- %d clusters found' % n_clusters_found)

pd.options.mode.chained_assignment = None # default='warn'
for min_points in range(15, 6, -1):
    print("Finding tracks with at least %d hits" % min_points)
    for shift in np.arange(-0.0025, 0.0025, 0.00025):
        find_clusters(min_points=min_points, max_radius=0.01, shift=shift, plot_intermediate=False)
        find_clusters(min_points=min_points, max_radius=0.01, shift=shift, phi_wraparound=True, plot_intermediate=False)

##################
# Plot remaining #
##################

def plot_predicted_and_true():
    hits['phiCR'] = hits['phi'] - shift*hits['R']
    
    tracks = truth.particle_id.unique()[1::20]
    fig = plt.figure(figsize=(20,7))
    ax = fig.add_subplot(111)
    tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
    clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
    for cluster in clusters:
        cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
        plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
        t = hits[hits['hit_id'].isin(plot_hit_ids)][['eta', 'phi']]
        if cluster == -1:
            ax.plot(t.phi, t.eta, '.', ms=10, color='black')
        else:
            ax.plot(t.phi, t.eta, '.-', ms=10)
    ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)
    plt.show()
    
    fig = plt.figure(figsize=(20,7))
    ax = fig.add_subplot(111)
    for track in tracks:
        hit_ids = truth[truth['particle_id'] == track]['hit_id']
        t = hits[hits['hit_id'].isin(hit_ids)][['eta', 'phi']]
        ax.plot(t.phi, t.eta, '.-', ms=10)
    ax.set_title("True Tracks (Line Plot)", y=-.15, size=20)
    plt.show()

plot_predicted_and_true()
