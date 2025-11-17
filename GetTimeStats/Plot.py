import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_time_stats_NodeSize(file_name):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)

    grayscale = df[df['Dataset']=="MNIST"]
    rgb = df[df['Dataset']=="CIFAR10"]

    grayscale = grayscale.groupby(['Nodes'])['Time'].mean().reset_index()
    rgb = rgb.groupby(['Nodes'])['Time'].mean().reset_index()

    plt.figure(figsize=(6, 4))
    plt.plot(rgb['Nodes'], rgb['Time'], marker='o', label='RGB')
    plt.plot(grayscale['Nodes'], grayscale['Time'], marker='s', label='Grayscale')

    plt.xlabel('# Nodes')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Figures/TimeStats_NodeSize.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

def plot_time_stats_SampleSize(file_name):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)

    grayscale = df[df['Dataset']=="MNIST"]
    rgb = df[df['Dataset']=="CIFAR10"]

    grayscale = grayscale.groupby(['Sample_Size'])['Time'].mean().reset_index()
    rgb = rgb.groupby(['Sample_Size'])['Time'].mean().reset_index()


    plt.figure(figsize=(6, 4))
    plt.plot(rgb['Sample_Size'], rgb['Time'], marker='o', label='RGB')
    plt.plot(grayscale['Sample_Size'], grayscale['Time'], marker='s', label='Grayscale')

    plt.xlabel('# Samples')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Figures/TimeStats_SampleSize.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

def plot_time_stats_LayerSize(file_name):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)
    df['ExtraLayers'] = df['ExtraLayers']+2
    print(df)
    # grayscale = df[df['Dataset']=="MNIST"]
    rgb = df[df['Dataset']=="CIFAR10"]

    # grayscale = grayscale.groupby(['Sample_Size'])['Time'].mean().reset_index()
    rgb = rgb.groupby(['ExtraLayers'])['Time'].mean().reset_index()


    plt.figure(figsize=(6, 4))
    plt.plot(rgb['ExtraLayers'], rgb['Time'], marker='o', label='RGB')
    # plt.plot(grayscale['ExtraLayers'], grayscale['Time'], marker='s', label='Grayscale')

    plt.xlabel('# Layers')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Figures/TimeStats_LayerSize.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    # plot_time_stats_NodeSize("Stats/TimeStats_NodeSize.csv")
    # plot_time_stats_SampleSize("Stats/TimeStats_SampleSize.csv")
    plot_time_stats_LayerSize("Stats/TimeStats_LayerSize.csv")