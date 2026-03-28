import pandas as pd
import matplotlib.pyplot as plt

def plot_trend(df):
    train_df = df[(df['Run'] == "5") & (df['Phase'] == "Train")].copy()

    train_df['Epoch'] = pd.to_numeric(train_df['Epoch'])
    train_df['Train_acc'] = pd.to_numeric(train_df['Train_acc'])
    train_df['Val_acc'] = pd.to_numeric(train_df['Val_acc'])
    train_df = train_df.sort_values('Epoch')

    epochs = train_df['Epoch'].values
    train_acc = train_df['Train_acc'].values
    val_acc = train_df['Val_acc'].values


    gurobi_df = df[(df['Run'] == "5") & (df['Phase'] == "GurobiEdit")].copy()
    gurobi_df['Epoch'] = pd.to_numeric(gurobi_df['Epoch'])
    gurobi_df['Train_acc'] = pd.to_numeric(gurobi_df['Train_acc'])
    gurobi_df['Val_acc'] = pd.to_numeric(gurobi_df['Val_acc'])
    gurobi_df = gurobi_df.sort_values('Epoch')
    gurobi_df['Epoch'] = gurobi_df['Epoch'] + 11
    g_epochs = gurobi_df['Epoch'].values
    g_train_acc = gurobi_df['Train_acc'].values
    g_val_acc = gurobi_df['Val_acc'].values



    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='.')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='.')
    plt.plot(g_epochs, g_train_acc, label='Gurobi Train Accuracy', marker='.')
    plt.plot(g_epochs, g_val_acc, label='Gurobi Validation Accuracy', marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.ylim(97, 100)
    plt.grid(True)

    plt.show()


def plot_loss(df):
    train_df = df[(df['Run'] == "5") & (df['Phase'] == "Train")].copy()

    train_df['Epoch'] = pd.to_numeric(train_df['Epoch'])
    train_df['Train_loss'] = pd.to_numeric(train_df['Train_loss'])
    train_df['Val_loss'] = pd.to_numeric(train_df['Val_loss'])
    train_df = train_df.sort_values('Epoch')

    epochs = train_df['Epoch'].values
    train_loss = train_df['Train_loss'].values
    val_loss = train_df['Val_loss'].values


    gurobi_df = df[(df['Run'] == "5") & (df['Phase'] == "GurobiEdit")].copy()
    gurobi_df['Epoch'] = pd.to_numeric(gurobi_df['Epoch'])
    gurobi_df['Train_loss'] = pd.to_numeric(gurobi_df['Train_loss'])
    gurobi_df['Val_loss'] = pd.to_numeric(gurobi_df['Val_loss'])
    gurobi_df = gurobi_df.sort_values('Epoch')
    gurobi_df['Epoch'] = gurobi_df['Epoch'] + 11
    g_epochs = gurobi_df['Epoch'].values
    g_train_loss = gurobi_df['Train_loss'].values
    g_val_loss = gurobi_df['Val_loss'].values



    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='.')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='.')
    plt.plot(g_epochs, g_train_loss, label='Gurobi Train Loss', marker='.')
    plt.plot(g_epochs, g_val_loss, label='Gurobi Validation Loss', marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.ylim(0, 0.13)
    plt.grid(True)

    plt.show()


df = pd.read_csv("NNRunLog/MNIST_WithTestAccForPlot.csv")
plot_loss(df)
