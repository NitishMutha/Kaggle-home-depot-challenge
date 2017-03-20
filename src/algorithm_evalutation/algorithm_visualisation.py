# Course - COMPGI15: Information Retrival & Data Mining
# Developers: Russel Daries, Rafiel Faruq, Nitish Mutha, Alaister Moull
# Purpose: Plot training and validation curves for various algorithms

# Import Nesscary Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

IMAGE_DIR = '../../data/output/images/'
filename = 'ranknet'
datafile_1 = "rb_all_features"
datafile_2 = 'test_results_2'
datafile_3 = 'test_results_3'
datafile_4 = 'test_results_4'

x_data_label = "#iter   "
x_axis_label = 'Epochs'
y_axis_label = 'NDCG@10'


# Read in nesscary files
train_data_1 = pd.read_excel('../../data/output/results/'+datafile_1+'.csv')
train_data_2 = pd.read_excel('../../data/output/results/'+datafile_2+'.csv')
train_data_3 = pd.read_excel('../../data/output/results/'+datafile_3+'.csv')
train_data_4 = pd.read_excel('../../data/output/results/'+datafile_4+'.csv')

print(train_data_1.keys())

# Function to plot training curves
def plot_metrics(panda_dataframe_1,panda_dataframe_2,panda_dataframe_3,panda_dataframe_4,x_data_label,x_label,y_label,filename):

    plt.figure()
    plt.plot(panda_dataframe_1[x_data_label],panda_dataframe_1[" NDCG@10-T "], label='Train-1', linewidth=2.0)
    plt.plot(panda_dataframe_1[x_data_label],panda_dataframe_1[" NDCG@10-V "], label='Valid-1', linewidth=2.0)

    plt.plot(panda_dataframe_2[x_data_label],panda_dataframe_2[" NDCG@10-T "], label='Train-2', linewidth=2.0)
    plt.plot(panda_dataframe_2[x_data_label],panda_dataframe_2[" NDCG@10-V "], label='Valid-2', linewidth=2.0)

    plt.plot(panda_dataframe_3[x_data_label],panda_dataframe_3[" NDCG@10-T "], label='Train-3', linewidth=2.0)
    plt.plot(panda_dataframe_2[x_data_label],panda_dataframe_2[" NDCG@10-V "], label='Valid-3', linewidth=2.0)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(x_label,fontsize = 18)
    plt.ylabel(y_label,fontsize = 18)

    plt.plot(panda_dataframe_4[x_data_label],panda_dataframe_4[" NDCG@10-T "], label = 'Train-F', linewidth=2.0)
    plt.plot(panda_dataframe_4[x_data_label],panda_dataframe_4[" NDCG@10-V "], label='Valid-F', linewidth=2.0)

    plt.legend()
    plt.grid(True)
    plt.savefig(filename + '_1.eps', bbox_inches='tight', format='eps',dpi=100)
    plt.close()

# Plotting call for graph
plot_metrics(train_data_1,train_data_2,train_data_3,train_data_4,x_data_label,x_axis_label,y_axis_label,IMAGE_DIR+filename)
