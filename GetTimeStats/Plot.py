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

    plt.xlabel('# Nodes', fontsize=14)
    plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.legend(fontsize=14)
    # plt.xticks(rotation=45)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("Figures/TimeStats_NodeSize.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

def plot_time_stats_SampleSize(file_name, method):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)

    grayscale = df[df['Dataset']=="MNIST"]
    rgb = df[df['Dataset']=="CIFAR10"]

    grayscale = grayscale.groupby(['Sample_Size'])['Time'].mean().reset_index()
    rgb = rgb.groupby(['Sample_Size'])['Time'].mean().reset_index()


    plt.figure(figsize=(6, 4))
    plt.plot(rgb['Sample_Size'], rgb['Time'], marker='o', label='RGB')
    plt.plot(grayscale['Sample_Size'], grayscale['Time'], marker='s', label='Grayscale')

    plt.xlabel('# Samples', fontsize=14)
    plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.legend(fontsize=14)
    # plt.xticks(rotation=45)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Figures/TimeStats_SampleSize_{method}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

def plot_time_stats_LayerSize(file_name):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)
    df['ExtraLayers'] = df['ExtraLayers']+2
    # print(df)
    # grayscale = df[df['Dataset']=="MNIST"]
    rgb = df[df['Dataset']=="CIFAR10"]

    # grayscale = grayscale.groupby(['Sample_Size'])['Time'].mean().reset_index()
    rgb = rgb.groupby(['ExtraLayers'])['Time'].mean().reset_index()


    plt.figure(figsize=(6, 4))
    plt.plot(rgb['ExtraLayers'], rgb['Time'], marker='o', label='RGB')
    # plt.plot(grayscale['ExtraLayers'], grayscale['Time'], marker='s', label='Grayscale')

    plt.xlabel('# Layers', fontsize=14)
    plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.legend(fontsize=14)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.yticks(np.arange(0, 100, 10), fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig("Figures/TimeStats_LayerSize.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    plot_time_stats_NodeSize("Stats/TimeStats_NodeSize.csv")
    plot_time_stats_SampleSize("Stats/TimeStats_SampleSize_RAB.csv", "RAB")
    plot_time_stats_SampleSize("Stats/TimeStats_SampleSize_RAF.csv", "RAF")
    plot_time_stats_LayerSize("Stats/TimeStats_LayerSize.csv")
