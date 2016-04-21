from imports import *
from toolbox import *
import pandas as pd
from preprossing import *
from segmentation import *
from plotting import *
from tracking import *


# movie_dir = "/Users/omidi/Dropbox/Communication/Gene.Expression.Inheritance/Matlab differences/Movies/8.2/zStackedYFP"
movie_dir = "/Users/omidi/Dropbox/Communication/Gene.Expression.Inheritance/Matlab differences/Movies/8.2/img"

images = deque(maxlen=500)
cells = deque(maxlen=500)


# # from slic import *
# fname = os.path.join(movie_dir, "fluc_%d.png" % 73)
# img = cv2.imread(fname, -1)
# img8bit = convertTo8bit(img, 0, 65535)
# img8bit = bluring(img8bit)
# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6,6))
# cl1 = clahe.apply(img8bit)
# img = rescale_intensity(cl1, in_range=(50,200))
# dil = cv2.dilate(img, disk(5), iterations=2)
# thresh = np.zeros_like(img, dtype=np.uint8)
# thresh[img > (dil * 0.5)] = 1
# show_image(thresh)
# exit()
# grd = rank.gradient(img, disk(2), mask=thresh)
# exit()
# # local_maxi = peak_local_max(distance, indices=False, min_distance=10,
# #                          labels=img)
# # gradient = rank.gradient(img, disk(2))
# # markers = ndi.label(local_maxi)[0]
# markers = rank.gradient(img, disk(1))
# # print markers[218:244, 188:204]
# # print markers[185:288, 200:217]
# markers[thresh==0] = 255
# markers = np.invert(markers)
# # print markers[185:288, 200:217]
# # print markers[218:244, 188:204]
# local_maxi = peak_local_max(markers,
#                         indices=False,
#                         footprint=disk(3),
#                         threshold_abs=15,
#                         )
# markers = ndi.label(local_maxi, disk(1))[0]
# markers = measure.label(local_maxi, neighbors=8)
# # print markers[185:288, 200:217]
# # print dil[218:244, 188:204]
# # print markers[69:93, 465:503]
#
# exit()
# print markers[265:288, 200:217]
# markers[markers==-1] = 0
# # print markers[267:290, 198:214]
# # print markers[220:242, 186:201]
# exit()
# # print markers[164:188, 362:382]
#
# # print markers[270:286, 199:214]
# # print markers[171:183, 366:384]
# # print num_of_cells
# borders = filters.scharr(img)
# exit()
# show_image(markers)
#
#
# # markers = ndi.label(markers)[0]
# # local gradient (disk(1) is used to keep edges thin)
# # gradient = rank.gradient(img, disk(1))
# borders = filters.scharr(img)
# # process the watershed
# labels = watershed(borders, markers)
# show_image(borders)
# exit()
#
# # fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True, sharey=True,
# #                          subplot_kw={'adjustable': 'box-forced'})
# # ax0, ax1, ax2 = axes
# # ax0.imshow(cl1, cmap=plt.cm.gray, interpolation='nearest')
# # ax0.set_title('Overlapping objects')
# # ax1.imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
# # ax1.set_title('Distances')
# # ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
# # ax2.set_title('Separated objects')
# # plt.show()
# # exit()

im = io.imread("/Users/omidi/Dropbox/Communication/Gene.Expression.Inheritance/Matlab differences/Movies/Movies/48h_4.2_240715 2.tif",
          plugin='tifffile')

# print im.shape
# exit()
# all_fnames = [f for f in os.listdir(movie_dir) if re.search("^fluc_\\d+\.png$", f)]

# for i in xrange(1, len(all_fnames)+1):
max_time_point = im.shape[0]
max_time_point = 100
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6,6))
for i in xrange(1, max_time_point):
    # fname = os.path.join(movie_dir, "fluc_%d.png" % i)
    # img = cv2.imread(fname, -1)
    img = im[i]
    images.append(img)
    img8bit = convertTo8bit(img, 0, 65535)
    # img8bit = bluring(img8bit)
    img8bit = clahe.apply(img8bit)
    img8bit = rescale_intensity(img8bit, in_range=(70,200))
    print i
    labels = watershed_segmentation(img8bit, i)
    img_for_showing = clahe.apply(convertTo8bit(img, 0, 65535))
    img_for_showing = rescale_intensity(img_for_showing, in_range=(70,255))
    img_for_showing = median(img_for_showing, disk(1))
    cells.appendleft(extract_cells_properties(img8bit, labels,
                                              img_for_showing, i))
    cells = pd.concat(cells)
    cells.to_csv("~/image_analysis/output_%d.csv" % i)
    del(cells)
    cells = deque(maxlen=500)
# cells = pd.concat(cells)
# print cells
exit()
for t in range(65, max_time_point):
    print t
    plot_for_image(images, cells, t)
# plot_for_n_time(images, cells, 16)
dist_matrix = pairwise_distance(cells)
lab2index = dict([(name, i) for i, name in enumerate(np.array(cells.index))])
max_distance = .2
graph = gv.Digraph(format='svg')
labels = np.array(cells.index)
edges = set()
depth = 5
for level in range(1,max_time_point-2-depth):
    print level
    cells = [l for l in labels if re.search('^%d\\.\d+$' % level, l)]
    # print cells
    for cell in cells:
        current = cell
        query_node = cell
        graph.node(current, color="white", style="filled",
                   fillcolor='#8fbdecff', fontname='Helvetica')
        for t in range(level+1, level+depth+1):
            second_layer = [lab2index[l] for l in labels if re.search('^%d\\.\d+$' % t, l)]
            second_labels = labels[second_layer]
            dist_vector = dist_matrix[lab2index[query_node], second_layer]
            index_lowest = np.argmin(dist_vector)
            if dist_vector[index_lowest] < max_distance:
                c = second_labels[index_lowest]
                graph.node(c, color="white", style="filled",
                           fillcolor='#8fbdecff', fontname='Helvetica')
                edges.add( (current, c, 1) )                # g2.edge(current, c, arrowhead='open',
                #         color="#1d64adff")
                # print g2
                # exit()
                current = c
                query_node = current
            else:
                u = "%d.%s.?" % (t, current.split(".")[1])
                graph.node(u, color="white", style="filled",
                           fillcolor='#e1a593ff',
                           shape='hexagon', fontname='Helvetica')
                edges.add( (current, u, 0) )
                # g2.edge(current, u, color="#dd6534ff",
                #         arrowhead="open", style="dashed")
                current = u

for edge in edges:
    if edge[2]:
        graph.edge(edge[0], edge[1], arrowhead='open',
                   color="#1d64adff")
    else:
        graph.edge(edge[0], edge[1], color="#dd6534ff",
                   arrowhead="open", style="dashed")

filename = graph.render(filename='graph')

# print second_layer[dist_matrix[lab2index['1.93'], second_layer] < max_distance]
exit()
# img = color_enhancement(img)
# img2 = local_adaptive_thresholding(img)
# img2 = watershed_segmentation(img)
show_img(img)
# slic_segmentation(img2, numSegments=40, compactness=5)
exit()
# cv2.circle(img, (200, 200), 25, (65535,65535,65535), 3)
# show_img(img)
cnts, intes, areas, contours, radius = find_cells(img)

cells = deque(maxlen=len(cnts))
for i in xrange(len(cnts)):
    cells.appendleft(Cell(cnts[i], intes[i], areas[i], radius[i], 1, contours[i]))

print len(cells)

for img_index in xrange(1, len(all_fnames)):
    img = images[img_index]
    cnts, intes, areas, contours, radius = find_cells(img)
    print img_index, len(cnts)
    for cell in cells:
        dists = [distance(cell.last_position(), c) for c in cnts]
        try:
            min_index, min_distance = min(enumerate(dists), key=operator.itemgetter(1))
        except:
            min_distance = 1000
        if min_distance <= 25:
            cell.add_position(cnts[min_index])
            cell.add_intensity(intes[min_index])
            cell.add_area(areas[min_index])
            cell.add_radius(radius[min_index])
            cell.update_contours(contours[min_index])
            cell.add_flag(v=1)
            # removing them from the discovered cells
            del cnts[min_index]
            del intes[min_index]
            del areas[min_index]
            del contours[min_index]
            del radius[min_index]
        else:
            cell.add_position(cell.last_position())  # use the last recorded position
            cell.add_radius(cell.last_radius())
            mask = np.zeros(img.shape, np.uint16)
            cv2.circle(mask, cell.last_position(), int(cell.last_radius())+1,
                       (65535, 65535, 65535), -1)
            total_area = np.sum(mask) / 65535.
            cell.add_area(total_area)  # use the last recorded area
            mask = np.bitwise_and(img, mask)
            cell.add_intensity(np.sum(mask))
            cell.add_flag(v=0)

    new_cells = deque(maxlen=1000)
    for c, i, ar, cont, rad in zip(cnts, intes, areas, contours, radius):
        new_cells.appendleft(Cell(c, i, ar, rad, img_index, cont))
    for nc in new_cells:
         cells.appendleft(nc)

# fig, ax = plt.subplots()
# plt.interactive(True)
# plt.set_cmap("gist_heat")
#
# for i in xrange(len(images)):
#     img = images[i]
#     for x in xrange(len(cells)):
#         cell = cells[x]
#         cv2.circle(img, cell.return_positions()[i], 7, (255, 255, 255), 1)
#     # show_img(img)
#     im = ax.imshow(img, interpolation="bicubic")
#     im.set_data(img)
#     fig.show()
#     fig.canvas.draw()

print len(cells)
exit()
img = images[0]
print len(cells)
for x in range(len(cells)):
    cell = cells[x]
    vals = cell.return_positions()
    for v in xrange(len(vals)-1):
        try:
            # print distance(vals[v], vals[v+1])
            cv2.line(img, vals[v], vals[v+1], (65535, 65535, 65535), 1)
        except IndexError:
            continue
    # cv2.circle(img, cells[i].last_position(), 10, (255, 255, 255), 1)
show_img(img)


for c, index in zip(cells, range(1, len(cells)+1)):
    areas = c.return_areas()
    areas.reverse()
    intens = c.return_intensities()
    intens.reverse()
    pos = c.return_positions()
    pos.reverse()
    flag = c.return_flags()
    flag.reverse()
    start = c.return_start_time()
    fname = "res/%d_cell.dat" % index
    with open(fname, "w") as outf:
        for a, i, p, f in zip(areas, intens, pos, flag):
            outf.write("%d\t%d\t%d\t%d\t%d\t%d\n" % (p[0], p[1], a, i, f, start) )

# fig, ax = plt.subplots()
# plt.interactive(True)
# plt.set_cmap("gist_heat")
# im = ax.imshow(img, interpolation="bicubic")
# im.set_data(img)
# fig.show()
# plt.colorbar(im)
# fig.canvas.draw()
# _, thresh = cv2.threshold(img, 20, 255,
#         cv2.THRESH_BINARY)
#
# frame = img
# frame = imutils.resize(frame, width=600)
# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
# kernel = np.ones((3,3),np.uint8)
# mask = cv2.morphologyEx(thresh,cv2.MORPH_ELLIPSE,kernel, iterations=2)
# mask = cv2.erode(mask, None, iterations=2)
# mask = cv2.dilate(mask, None, iterations=2)
#
# # kernel = np.ones((3,3),np.uint8)
# # opening = cv2.morphologyEx(thresh,cv2.MORPH_ELLIPSE,kernel, iterations = 3)
# #
# # sure_bg = cv2.dilate(opening, kernel, iterations=3)
# # dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
# # ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
# # sure_fg = np.uint8(sure_fg)
# # marker = cv2.subtract(sure_bg, sure_fg)
# # marker32 = np.int32(marker)
# # # cv2.watershed(img,marker32)
# # m = cv2.convertScaleAbs(marker32)
# # ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # res = cv2.bitwise_and(img,img,mask = thresh)
# # # contours, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# # # cv2.drawContours(res, contours, -1, (0,255,0), 3)
# #
#
# cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)[-2]
# pts = deque(maxlen=1000)
# rad = deque(maxlen=1000)
# intens = deque(maxlen=1000)
# polygs = deque(maxlen=1000)
# cells = deque(maxlen=len(cnts))
# for c in cnts:
#     if cv2.contourArea(c)>150:   # the threshold need to be adjusted
#         continue
#     ((x, y), radius) = cv2.minEnclosingCircle(c)
#     M = cv2.moments(c)
#     polygs.appendleft(c)
#     rad.appendleft(radius)
#     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#     cells.appendleft( Cell(center, M["m00"]) )
#     intens.appendleft(M["m00"])
#     pts.appendleft(center)