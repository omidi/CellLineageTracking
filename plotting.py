from imports import *
from toolbox import *

def plot_for_n_time(images, cells, max_time_range):
    ncols, nrows = 4, 4
    fig, axes = plt.subplots(4, 4, figsize=(4*ncols, 4*nrows))
    for t in xrange(1, 16+1):
        img = convertTo8bit(images[t], 0, 65535)
        index = t - 1
        i, j = index // ncols, index % ncols
        axes[i, j].imshow(img,
                          interpolation='nearest', cmap='gist_heat')
        detected_cells = cells[cells['t'] == t].copy()
        labels = np.array(detected_cells.index)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        x_lim = axes[i, j].get_xlim()
        y_lim = axes[i, j].get_ylim()
        axes[i, j].scatter(detected_cells['y'], detected_cells['x'], edgecolors='w',
                            s=90, marker='o', facecolor='none')
        # for label, x, y in zip(labels, detected_cells['y'], detected_cells['x']):
        #     axes[i, j].annotate(
        #         label, fontsize=9,
        #         xy = (x, y), xytext = (-3, 3),
        #         textcoords = 'offset points', ha = 'right', va = 'bottom', color='yellow')

        axes[i, j].set_xlim(x_lim)
        axes[i, j].set_ylim(y_lim)
        axes[i, j].set_title("t=%d" % t)
    ## Remove empty plots
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    plt.show()


def plot_for_image(images, cells, t=-1):
    img = convertTo8bit(images[t], 0, 65535)
    plt.imshow(img, interpolation='nearest', cmap='gist_heat')
    detected_cells = cells[cells['t'] == t].copy()
    labels = np.array(detected_cells.index)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(detected_cells['y'], detected_cells['x'], edgecolors='w',
                            s=90, marker='o', facecolor='none')
    ax = plt.gca()
    for label, x, y, r in zip(labels, detected_cells['x'], detected_cells['y'], detected_cells['area']):
        # e = plt.Circle((y,x), r/20)
        # ax.add_artist(e)
        # e.set_clip_box(ax.bbox)
        # e.set_edgecolor( 'w' )
        # e.set_facecolor( 'none' )  # "none" not None
        # e.set_alpha( .5 )
        plt.annotate(
            str(int(r)), fontsize=7,
            xy = (y, x), xytext = (-3, 3),
            textcoords = 'offset points', ha = 'right', va = 'bottom', color='yellow')
    plt.title("t=%d" % t)
    plt.axis("off")
    plt.savefig('detected_cells/time_%d.png' % t, bbox_inches='tight')
    plt.close()
    return 0
