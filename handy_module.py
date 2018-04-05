"""
Created on:   Mon Apr  2 11:33:33 2018
@author:      Alessandro Nesti
"""



import matplotlib
import matplotlib.pyplot as plt
import os

def std_format_fig():
    # To plot pretty figures
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12



def save_fig(fig_id, tight_layout=True, path = "."):

    path = os.path.join(path, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)