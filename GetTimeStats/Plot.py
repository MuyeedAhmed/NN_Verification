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
    plt.plot(rgb['Nodes'], rgb['Time'], marker='o', label='CIFAR10')
    plt.plot(grayscale['Nodes'], grayscale['Time'], marker='s', label='MNIST')

    plt.xlabel('# Nodes', fontsize=16)
    plt.ylabel('Average Time (seconds)', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures/TimeStats_NodeSize.pdf", format='pdf', bbox_inches='tight')


def plot_time_stats_Classes(file_name):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)

    grayscale = df[df['Dataset']=="MNIST"]
    rgb = df[df['Dataset']=="CIFAR10"]

    grayscale = grayscale.groupby(['Classes'])['Time'].mean().reset_index()
    rgb = rgb.groupby(['Classes'])['Time'].mean().reset_index()

    plt.figure(figsize=(6, 4))
    plt.plot(rgb['Classes'], rgb['Time'], marker='o', label='CIFAR10')
    if grayscale.shape[0]>0:
        plt.plot(grayscale['Classes'], grayscale['Time'], marker='s', label='')

    plt.xlabel('# Classes', fontsize=16)
    plt.ylabel('Average Time (seconds)', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures/TimeStats_Classes.pdf", format='pdf', bbox_inches='tight')



def plot_time_stats_SampleSize(file_name, method):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)

    grayscale = df[df['Dataset']=="MNIST"]
    rgb = df[df['Dataset']=="CIFAR10"]

    grayscale = grayscale.groupby(['Sample_Size'])['Time'].mean().reset_index()
    rgb = rgb.groupby(['Sample_Size'])['Time'].mean().reset_index()


    plt.figure(figsize=(6, 4))
    plt.plot(rgb['Sample_Size'], rgb['Time'], marker='o', label='CIFAR10')
    if grayscale.shape[0]>0:
        plt.plot(grayscale['Sample_Size'], grayscale['Time'], marker='s', label='MNIST')
    # plt.text(
    #     0.5, 0.95, "(a) STC",
    #     transform=plt.gca().transAxes,
    #     fontsize=18,
    #     va='top', ha='center'
    # )
    plt.xlabel('# Samples', fontsize=16)
    plt.ylabel('Average Time (seconds)', fontsize=16)
    plt.legend(fontsize=16)
    if method == "RAB":
        plt.xticks(
            ticks=[2000, 4000, 6000, 8000, 10000],
            labels=["2k", "4k", "6k", "8k", "10k"],
            fontsize=14
        )
    else:
        plt.xticks(
            ticks=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
            labels=["2k", "4k", "6k", "8k", "10k", "12k", "14k", "16k", "18k", "20k"],
            fontsize=14
        )
    plt.yticks(fontsize=14)
    if method == "RAB":
        plt.title("(a) STC", fontsize=18)
    else:
        plt.title("(b) CmC", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"Figures/{file_name.split('/')[1].split('.')[0]}_{method}.pdf", format='pdf', bbox_inches='tight')

def plotyFit(file_name, method):
    df = pd.read_csv(file_name)
    df['Time'] = df['Time'].astype(float)
    print(df)
    x = df['Sample_Size'].values
    y = df['Time'].values

    logx = np.log(x)
    logy = np.log(y)

    p, c = np.polyfit(logx, logy, 1)
    a = np.exp(c)

    print("Best exponent p =", p)
    print("Coefficient a =", a)

    x_dense = np.linspace(min(x), max(x), 500)
    y_pred = a * x_dense**p

    plt.figure(figsize=(8,5))
    plt.scatter(x, y, label="Actual Data", color="blue")
    plt.plot(x_dense, y_pred, label="Regression", linewidth=2, color="red")

    plt.xlabel("N")
    plt.ylabel("Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()




def TimeVSample_1st_Best(file_name1, file_name2, method):
    df1 = pd.read_csv(file_name1)
    df2 = pd.read_csv(file_name2)
    df1['Time'] = df1['Time'].astype(float)
    df2['Time'] = df2['Time'].astype(float)


    df1 = df1.groupby(['Sample_Size'])['Time'].mean().reset_index()
    df2 = df2.groupby(['Sample_Size'])['Time'].mean().reset_index()


    plt.figure(figsize=(6, 4))
    plt.plot(df2['Sample_Size'], df2['Time'], marker='o', label='Best Solution')
    plt.plot(df1['Sample_Size'], df1['Time'], marker='o', label='1st Solution')

    plt.xlabel('# Samples', fontsize=16)
    plt.ylabel('Average Time (seconds)', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(
        ticks=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
        labels=["2k", "4k", "6k", "8k", "10k", "12k", "14k", "16k", "18k", "20k"],
        fontsize=14
    )
    plt.yticks(fontsize=14)
    # plt.title("(b) CmC", fontsize=18)

    plt.tight_layout()
    plt.savefig(f"Figures/FirstSolutionVSBest.pdf", format='pdf', bbox_inches='tight')


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
    plt.plot(rgb['ExtraLayers'], rgb['Time'], marker='o', label='CIFAR10')
    # plt.plot(grayscale['ExtraLayers'], grayscale['Time'], marker='s', label='Grayscale')

    plt.xlabel('# Layers', fontsize=16)
    plt.ylabel('Average Time (seconds)', fontsize=16)
    plt.legend(fontsize=16)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.yticks(np.arange(0, 100, 10), fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig("Figures/TimeStats_LayerSize.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

def plot_GlobalMisclassified(file_name):
    df = pd.read_csv(file_name)
    # df = df.sort_values(by="n")

    plt.figure(figsize=(8,4))

    for run_id, group in df.groupby("RunID"):
        plt.plot(
            group["n"],
            group["GlobalMisclassified"],
            marker='o',
            label=f"Run {run_id}"
        )

    plt.xlabel("Subset Sizes", fontsize=16)
    plt.ylabel("Total Misclassified Samples", fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(
        ticks=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
        labels=["2k", "4k", "6k", "8k", "10k", "12k", "14k", "16k", "18k", "20k"],
        fontsize=14
    )
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures/GlobalFlips.pdf", format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    # plot_time_stats_NodeSize("Stats/TimeStats_NodeSize.csv")
    # plot_time_stats_SampleSize("Stats/TimeStats_SampleSize_RAB.csv", "RAB")
    # plot_time_stats_SampleSize("Stats/TimeStats_S_Thelma_20k.csv", "RAF")

    # # plot_time_stats_SampleSize("Stats/TimeStats_SampleSize_RAF.csv", "RAF")
    # plot_time_stats_LayerSize("Stats/TimeStats_LayerSize.csv")


    # # plot_time_stats_SampleSize("Stats/TimeStats_S_Louise.csv", "RAF")
    # TimeVSample_1st_Best("Stats/TimeStats_S_Thelma_20k_1s.csv", "Stats/TimeStats_S_Thelma_20k.csv", "RAF") ## Remove MNIST From stats
    # # plot_time_stats_SampleSize("Stats/TimeStats_S_Thelma_Small.csv", "RAF")

    # plot_GlobalMisclassified("Stats/GlobalFlips.csv")

    # plot_time_stats_Classes("Stats/TimeStats_Classes.csv")
    plotyFit("Stats/TimeStats_S_Thelma_20k_1s.csv", "RAF")
    plotyFit("Stats/TimeStats_S_Thelma_20k.csv", "RAF")