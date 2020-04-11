from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pickle 
import torch 
import IPython 
import pdb



def tsne_F(matrix_representation,matrix_labels,kind=63):
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=500)
    tsne_results = tsne.fit_transform(matrix_representation)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df = pd.DataFrame(matrix_representation.numpy(), index=list(range(matrix_representation.numpy().shape[0])), columns=["x"for i in range(matrix_representation.numpy().shape[1])], dtype=None, copy=False)
    # df['tsne-one'] = tsne_results[:,0]
    # df['tsne-two'] = tsne_results[:,1] 
    # df['tsne-three'] = tsne_results[:,2]
    # df['y'] = matrix_labels.numpy()
    # ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    # ax.scatter(
    # xs=df["tsne-one"], 
    # ys=df["tsne-two"], 
    # zs=df["tsne-three"], 
    # c=df["y"], 
    # cmap='cubehelix'
    # )
    # ax.set_xlabel('tsne-one')
    # ax.set_ylabel('tsne-two')
    # ax.set_zlabel('tsne-three')
    # plt.savefig("test_tsne.png")
    
    # for i in range(kind):
    #     pseudo_label = matrix_labels.numpy()
    #     one_label = np.zeros_like(pseudo_label)
    #     one_label[pseudo_label==i] = 1
    #     df["y"] = one_label
    #     df["tsne_1_dim"] =tsne_results[:,0]
    #     df["tsne_2_dim"] =tsne_results[:,1]
    #     plt.figure(figsize=(19,17))
    #     sns.scatterplot(
    #         x="tsne_1_dim", y="tsne_2_dim",
    #         hue="y",
    #         palette=sns.color_palette("RdBu_r", 2),
    #         data=df,
    #         legend="full",
    #         alpha=1
    #     )
    #     plt.savefig(f"tsne/number_{i}_cluster.png")
    #     plt.clf()


def pca_F(matrix_representation,matrix_labels,kind=63):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(matrix_representation)
    df = pd.DataFrame(matrix_representation.numpy(), index=list(range(matrix_representation.numpy().shape[0])), columns=["x"for i in range(matrix_representation.numpy().shape[1])], dtype=None, copy=False)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    df['y'] = matrix_labels.numpy()
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
    xs=df["pca-one"], 
    ys=df["pca-two"], 
    zs=df["pca-three"], 
    c=df["y"], 
    cmap='cubehelix'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.savefig("test_pca.png")

def pca_reduce_50_tsne_2d(matrix_representation,matrix_labels,kind=63):
    pca = PCA(n_components=50)
    df_old = pd.DataFrame(matrix_representation.numpy(), index=list(range(matrix_representation.numpy().shape[0])), columns=["x"for i in range(matrix_representation.numpy().shape[1])], dtype=None, copy=False)
    pca_result = pca.fit_transform(df_old)
    df_new = pd.DataFrame(pca_result, index=list(range(pca_result.shape[0])), columns=["x"for i in range(pca_result.shape[1])], dtype=None, copy=False)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    # df_new
    # pseudo_label = matrix_labels.numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    tsne_results = tsne.fit_transform(df_new)
    df_new["y"] = matrix_labels.numpy()
    df_new["tsne_1_dim"] =tsne_results[:,0]
    df_new["tsne_2_dim"] =tsne_results[:,1]
    plt.figure(figsize=(19,17))
    sns.scatterplot(
        x="tsne_1_dim", y="tsne_2_dim",
        hue="y",
        palette=sns.color_palette("RdBu_r", 63),
        data=df_new,
        legend="full",
        alpha=1
    )
    plt.savefig(f"tsne_pca50_train.png")
    plt.clf()

    # for i in range(kind):
    #     pseudo_label = matrix_labels.numpy()
    #     one_label = np.zeros_like(pseudo_label)
    #     one_label[pseudo_label==i] = 1
    #     df_new["y"] = one_label
    #     df_new["tsne_1_dim"] =tsne_results[:,0]
    #     df_new["tsne_2_dim"] =tsne_results[:,1]
    #     plt.figure(figsize=(19,17))
    #     sns.scatterplot(
    #         x="tsne_1_dim", y="tsne_2_dim",
    #         hue="y",
    #         palette=sns.color_palette("RdBu_r", 2),
    #         data=df_new,
    #         legend="full",
    #         alpha=1
    #     )
    #     plt.savefig(f"visualize_embedding/tsne_pca50/number_{i}_cluster.png")
    #     plt.clf()

    #     one_label = np.zeros_like(pseudo_label)
    #     one_label[pseudo_label==i] = 1
    #     df["y"] = one_label
    #     df["tsne_1_dim"] =tsne_results[:,0]
    #     df["tsne_2_dim"] =tsne_results[:,1]
    #     plt.figure(figsize=(19,17))
    #     sns.scatterplot(
    #         x="tsne_1_dim", y="tsne_2_dim",
    #         hue="y",
    #         palette=sns.color_palette("RdBu_r", 2),
    #         data=df,
    #         legend="full",
    #         alpha=1
    #     )
    #     plt.savefig(f"tsne/number_{i}_cluster.png")
    #     plt.clf()


if __name__ == "__main__":
    speaker_representation = pickle.load(open("speaker_representation_train.p","rb"))

    tuple_of_list = list(zip(*speaker_representation))

    matrix_representation = torch.cat(tuple_of_list[0],dim=0)
    matrix_labels         = torch.cat(tuple_of_list[1],dim=0).unsqueeze(-1)

    time_start = time.time()
    pca_reduce_50_tsne_2d(matrix_representation,matrix_labels)
    # tsne_F(matrix_representation,matrix_labels)
    



