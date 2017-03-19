# Course - COMPGI15: Information Retrival & Data Mining
# Developers: Russel Daries, Rafiel Faruq, Nitish Mutha, Alaister Moull
# Purpose: Plot training and validation curves for various algorithms

# Import Nesscary Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_DIR = 'Data/Output/Images/'
filename = 'RankNet'
datafile = 'test_results'
x_axis_label = 'Epochs'
y_axis_label = 'RMSE'
# Read in nesscary files
train_data = pd.read_excel('Data/Output/Results/'+datafile+'.csv')

# Function to plot training curves
def plot_metrics(panda_dataframe,x_label,y_label,filename):

    plt.figure()
    plt.plot(panda_dataframe["Epochs"],panda_dataframe[" ERR@10-T  "],'ro-', label='Train', linewidth=2.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(x_label,fontsize = 18)
    plt.ylabel(y_label,fontsize = 18)
    plt.plot(panda_dataframe["Epochs"],panda_dataframe[" ERR@10-V  "],'bD-', label = 'Valid', linewidth=2.0)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename + '_metrics.eps', bbox_inches='tight', format='eps',dpi=100)
    plt.close()

# Plotting call for graph
plot_metrics(train_data,x_axis_label,y_axis_label,IMAGE_DIR+filename)
