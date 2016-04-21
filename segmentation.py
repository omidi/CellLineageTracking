from imports import *
from toolbox import *

def binary_segmentation(img):
    x, thresh = cv2.threshold(img, 20, 255,  \
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    cells = cv2.morphologyEx(thresh,cv2.MORPH_ELLIPSE,kernel, iterations = 2)
    return cells

def watershed_segmentation(img, t):
    img_median = median(img, disk(2))
    dil = cv2.dilate(img_median, disk(4), iterations=2)
    thresh = np.zeros_like(img_median, dtype=np.uint8)
    thresh[img_median > (dil * 0.5)] = 1
    thresh_erosion = erosion(thresh, disk(2))
    img2 = img.copy()
    img2[thresh_erosion==0] = 0
    # _, thresh = cv2.threshold(img, 20, 255,  \
    #                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = erosion(thresh, disk(1))
    # sure_fg = erosion(thresh, disk(1))
    # mask = np.zeros_like(sure_fg, dtype=np.uint8)
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)[-2]
    # for c in cnts:
    #     ((x, y), radius) = cv2.minEnclosingCircle(c)
    #     if radius < 100:
    #         cv2.circle(mask, (int(x),int(y)), int(radius)+1, 1, -1)
    #
    # mask = mask>0

    # markers = rank.gradient(img, disk(3))
    # markers[thresh==0] = 255
    # markers = np.invert(markers)
    markers = img2.copy()
    local_maxi = peak_local_max(markers,
                        indices=False,
                        min_distance=5,
                        )
    # markers = ndi.label(local_maxi, disk(1))[0]
    markers = measure.label(local_maxi, neighbors=4)
    markers2 = markers.copy()
    markers2[markers>0] = 255
    #
    # local_maxi = peak_local_max(markers,
    #                         indices=False,
    #                         footprint=disk(2),
    #                         threshold_abs=10,
    #                         labels=labels,
    #                         )
    # markers = ndi.label(local_maxi, disk(1))[0]
    # local gradient (disk(1) is used to keep edges thin)
    # borders = rank.gradient(img, disk(1))
    # x = markers.copy()
    # x[x>0] = 255
    # show_image(x)
    # show_image(borders)
    # exit()
    # borders = filters.sobel(img)
    img_enh = filters.rank.enhance_contrast(img_median, disk(1))
    borders = filters.sobel(img_enh)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.imshow(img_enh)
    ax0.set_title("Original image")

    ax1.imshow(median(thresh, disk(2)), cmap=plt.cm.gray)
    ax1.set_title("Threshold")

    ax2.imshow(markers2, cmap="inferno")
    ax2.set_title("Markers")

    ax3.imshow(borders, cmap=plt.cm.gray)
    ax3.set_title("Borders")

    for ax in axes.ravel():
        ax.axis('off')

    fig.tight_layout()
    plt.savefig('/Users/omidi/image_analysis/intermediate_results/time_%d.png' % t, bbox_inches='tight')
    plt.close()
    # process the watershed
    labels = watershed(borders, markers)
    return labels


def extract_cells_properties(img, labels, img16bit, t):
    max_area = 250  # maximum area for the cell, to avoid detecting large cluster of cells
    min_area =  20  # to avoid detecting tiny dots
    columns = ('l', 'x', 'y', 't', 'intensity', 'area', 'orientation',
               'solidity', 'major_axis_len', 'minor_axis_len', 'diameter',
               # 'zoom',
               'eccentricity',
               # 'bbox', 'coord',
               )
    regions = regionprops(labels, intensity_image=img16bit)
    properties = []
    indices = []
    # ploting
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img16bit, cmap='gray')
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    ells = []
    num = 1
    for region in regions:
        # filter out regions that are either too small or too big
        if (min_area < region.area < max_area):
            properties.append([     str(region.label),
                                    region.weighted_centroid[0],
                                    region.weighted_centroid[1],
                                    t,
                                    region.mean_intensity * region.area,
                                    region.area,
                                    region.orientation,
                                    region.solidity,
                                    region.major_axis_length,
                                    region.minor_axis_length,
                                    region.equivalent_diameter,
                                    # region.filled_image,
                                    region.eccentricity,
                                    # region.bbox,
                                    # region.coords,
                                ])
            num += 1
            y0, x0 = region.centroid
            orientation = region.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * region.major_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * region.major_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * region.minor_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * region.minor_axis_length
            # ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            # ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            # ax.plot(x0, y0, '.g', markersize=15)

            r = np.min([region.major_axis_length, region.minor_axis_length])
            ells.append(Ellipse(xy=(x0,y0), angle=orientation*360,
                                width=r, height=r))
            ax.annotate(
                '%d-%d' % (t, region.label), fontsize=10,
                xy = (x0, y0), xytext = (region.minor_axis_length, region.major_axis_length ),
                textcoords = 'offset points', ha = 'right', va = 'bottom', color='yellow')


            # in order to have the indices unique,
            # I used time.label for each cell
            indices.append('%d-%d' % (t, region.label))
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(.6)
        e.set_facecolor(rnd.rand(3))
    # plt.show()
    plt.savefig('/Users/omidi/image_analysis/detected_cells/time_%d.png' % t, bbox_inches='tight')
    plt.close()

    return pd.DataFrame(properties, index=indices, columns=columns)