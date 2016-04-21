from imports import *

def augmented_dendrogram(*args, **kwargs):
    ddata = hier.dendrogram(*args, **kwargs)
    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')
    return ddata


def pairwise_distance(cells_df):
    # max_radius = 50
    # df = cells_df[['x', 'y', 'intensity', 'area']].copy()
    df = cells_df[['x', 'y']].copy()
    # labels = np.array(df.index)
    dist_mat = dist.squareform(dist.pdist(df.values, 'mahalanobis'))
    return dist_mat
    # print dist_mat[1, :]
    # exit()
    # link_mat = hier.linkage(dist_mat, method='ward')
    # plt.figure(figsize=(25, 10))
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('sample index')
    # plt.ylabel('distance')
    # hier.dendrogram(
    #     link_mat,
    #     leaf_rotation=90.,  # rotates the x axis labels
    #     leaf_font_size=8.,  # font size for the x axis labels
    #     labels=labels,
    #     # show_contracted=True,
    #     # truncate_mode='lastp',
    #     # p=60,
    # )
    # K = len(labels)/4
    # max_d = 3
    # indices = hier.fcluster(link_mat, max_d, criterion='distance')
    # for i in np.unique(indices):
    #     print np.sort(labels[indices==i])
    #
    #
    # #
    # # print labels[x["leaves"]]
    # # print x["dcoord"]
    # # print x["icoord"]
    # # print x.keys()
    # plt.show()
    # exit()
    # cluster_idx = hier.fcluster(link_mat, max_radius,
    #                             criterion='distance')
    # print cluster_idx
    # exit()
    # den = hier.dendrogram(link_mat, labels=df.index, abv_threshold_color='#AAAAAA')
    # plt.xticks(rotation=90)
    # no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
    # sns.despine(**no_spine);
    # print ddata
    # print cluster_idx
    #

