from yolo import yolov3
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from shapely.geometry import box, polygon
from sklearn.cluster import DBSCAN
from watchdog.observers import Observer
import queue
import _pickle as cPickle
from time import sleep
import threading
from argparse import ArgumentParser
import utils.utils as utils


class ParkingLotDetector:
    class DEFAULTS:
        min_cluster_size = 35
        eps = 0.36
        buffer_location = '.'
        output_location = './output'
        show = False

    # region Constructors and Destructors
    def __init__(self, folder, min_cluster_size=DEFAULTS.min_cluster_size, buffer_location=DEFAULTS.buffer_location,
                 output_location=DEFAULTS.output_location, show=DEFAULTS.show):
        self._min_cluster_size = min_cluster_size
        self._buffer_location = buffer_location
        self._output_location = output_location
        self._yoloV3 = yolov3.Yolov3()
        self._detect_thread_running = False
        self._working_folder = folder
        self._show = show
        self._tracker=utils.FileCreatedTracker(working_folder=folder, callback=self.load_cars,
                                               args_array=True)

    def __del__(self):
        self.stop_tracking()

    # endregion

    # region Track folder
    def track_live(self):
        self.load_all_cars()
        self._tracker.start()

    def stop_tracking(self):
        self._tracker.stop()

    # endregion

    # region Load cars of images
    def load_all_cars(self):
        _list = [os.path.join(self._working_folder, file) for file in os.listdir(self._working_folder) if
                 os.path.isfile(os.path.join(self._working_folder, file))]
        self.load_cars(_list)

    def load_cars(self, file_list):
        files = []
        data = self._load_data()
        for file in file_list:
            if file not in data.keys():
                files.append(file)
        if len(files) > 0:
            res = self._yoloV3.yolo(files, return_mask=False, classes=(2,))
        for i in range(len(files)):
            data[files[i]] = res[i]
        print(f'Processed {len(files)} file(s)')
        self._dump_data(data)

    # endregion

    def detect_parking_lots(self, eps=DEFAULTS.eps):
        self.load_all_cars()
        data = self._load_data()
        centers = []
        elements = []
        for i in data.keys():
            if data[i] is None:
                continue
            for j in range(len(data[i])):
                x1, y1, x2, y2, conf, cls_conf, cls_pred = data[i][j]
                centers.append((int((x1 + x2) / 2), (int((y1 + y2) / 2))))
                elements.append(data[i][j])

        # TRYOUT CUSTOM CLUSTERING
        clusters = self._my_cluster(elements, eps=eps)
        clusters, centers, elements = self._remove_clusters_below(self._min_cluster_size, clusters, centers, elements)

        # Creating plot
        fig, ax = plt.subplots(1)

        parking_lots = list()

        # Intersecting:
        for cluster in np.unique(clusters):
            intersect = self.get_intersection2((np.array(elements)[np.array(clusters) == cluster]).tolist())
            if type(intersect) == polygon.Polygon:
                x, y = intersect.exterior.xy
                parking_lots.append((max(x), max(y), min(x), min(y)))
                ax.plot(x, y)

        ax.imshow(Image.open(data.popitem()[0]))
        self.plot_clusters(ax, centers, clusters)
        ax.axis("equal")
        title = "number of clusters: %d" % (len(set(clusters)))
        plt.title(title)
        if self._show:
            plt.show()
        plt.savefig(self._get_data_path(output=True, file_extension='png'))
        cPickle.dump(parking_lots, open(self._get_data_path(output=True), 'wb+'))

    # region Load and dump car data

    def _load_data(self):
        path = self._get_data_path(buffer=True)
        data = dict()
        if os.path.exists(path):
            read = open(path, 'rb')
            data = cPickle.load(read)
        return data

    def _dump_data(self, data):
        path = self._get_data_path(buffer=True)
        ouf = open(path, 'wb')
        cPickle.dump(data, ouf)

    # endregion

    # region Utils

    def _get_data_path(self, buffer=False, output=False, file_extension=None):
        if buffer:
            folder = self._buffer_location
            _file_extension = "buffer"
        elif output:
            folder = self._output_location
            _file_extension = "out"
        else:
            raise Exception("Either buffer of output have to be defined.")
        if file_extension:
            _file_extension=file_extension
        return os.path.join(folder, f'{self._working_folder}.{_file_extension}')

    # endregion

    # region Class methods

    @classmethod
    def plot_clusters(cls, ax, centers, clusters):
        ax.scatter(*np.transpose(centers), c=clusters, marker='o', cmap='rainbow')

    @classmethod
    def _remove_clusters_below(cls, min_cluster_size, clusters, centers, elements):
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_count = zip(unique, counts)
        for cc in cluster_count:
            if cc[1] < min_cluster_size:
                clusters[clusters == cc[0]] = -1
        centers = [cent for cent, clus in zip(centers, clusters) if clus != -1]
        elements = [elem for clust, elem in zip(clusters, elements) if clust != -1]
        clusters = [elem for elem in clusters if elem != -1]
        return (clusters, centers, elements)

    # def load(folder, k=None):
    #     list = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
    #     print(len(list))
    #     if k is None:
    #         images = list  # random.choices(list, k=k)
    #     else:
    #         images = random.choices(list, k=k)
    #
    #     images = [os.path.join(folder, im) for im in images]
    #
    #     path = f'{folder}_export.npy'
    #     if os.path.isfile(path):
    #         res = np.load(path, allow_pickle=True)
    #     else:
    #         res = yv3.yolo(images, return_mask=False, classes=(2,))
    #         np.save(path, np.array(res))
    #     return res

    @classmethod
    def _my_cluster(cls, detections, eps=DEFAULTS.eps):
        differences = np.full((len(detections), len(detections)), -1.0)
        centers = []
        for i1 in range(0, len(detections)):
            centers.append(
                (int((detections[i1][0] + detections[i1][2]) / 2), (int((detections[i1][1] + detections[i1][3]) / 2))))
            for i2 in range(0, len(detections)):
                if i1 == i2:
                    differences[i2, i1] = 1
                    continue
                if differences[i2, i1] != -1:
                    differences[i1, i2] = differences[i2, i1]
                differences[i1, i2] = 1 - cls.bb_intersection_over_union(detections[i1], detections[i2])
                # if differences[i1][i2] != 0:
                #    print(differences[i1, i2])
        dbs = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        dbs.fit_predict(differences)
        labels = dbs.labels_

        return labels

    @classmethod
    def get_intersection(cls, boxes):
        first = True
        last = None
        for box_ in boxes:
            temp = box(box_[0], box_[1], box_[2], box_[3])
            if first:
                first = False
                last = temp
            else:
                temp2 = last.intersection(temp)
                last = temp2
        return last

    @classmethod
    def get_intersection2(cls, boxes):
        mask = np.zeros((int(max(np.array(boxes)[:, 2])), int(max(np.array(boxes)[:, 3]))))
        for box_ in boxes:
            mask[int(box_[0]):int(box_[2]), int(box_[1]):int(box_[3])] += 1
        max_val = len(boxes)  # max(map(max, mask))
        mask[mask < max_val * (0.75)] = None
        mask_np = np.array(mask)
        if np.where(~np.isnan(mask_np))[0].sum() == 0:
            return None
        xmax, ymax = np.max(np.where(~np.isnan(mask_np)), 1)
        xmin, ymin = np.min(np.where(~np.isnan(mask_np)), 1)
        return box(xmin, ymin, xmax, ymax)

    @classmethod
    def bb_intersection_over_union(cls, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        if interArea == 0:
            return 0

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = float(interArea) / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    # endregion


def _main():
    parser = ArgumentParser('Detect parking lots in parking lot images')
    parser.add_argument('folder', type=str, help='Folder containing the images')
    parser.add_argument("--mode", type=str, choices=['live', 'detect'],
                        help="Runnning mode")
    parser.add_argument("--output_location", default=ParkingLotDetector.DEFAULTS.output_location, type=str,
                        help="Location of all outputs")
    parser.add_argument("--buffer_location", default=ParkingLotDetector.DEFAULTS.buffer_location, type=str,
                        help="Location for buffer locations")
    parser.add_argument("--min_cluster_size", default=ParkingLotDetector.DEFAULTS.min_cluster_size, type=int,
                        help="Minimum Cluster size")
    parser.add_argument("--clustering_eps", default=ParkingLotDetector.DEFAULTS.eps, type=float,
                        help="EPS value for the clustering")
    args = parser.parse_args()

    pld = ParkingLotDetector(args.folder, min_cluster_size=args.min_cluster_size, buffer_location=args.buffer_location,
                             output_location=args.output_location, konker_username=args.username,
                             konker_password=args.password)
    if args.mode == 'live':
        pld.track_live()

        while True:
            sleep(5)
    else:
        pld.load_all_cars()
        pld.detect_parking_lots(eps=args.clustering_eps)

    # pld.detect_parking_lots()
    # # folder = 'sparking2'
    # folder = 'camera3'  # 'smartparking' # 'sparking2
    # # folder ='smartparking'
    # min_cluster_size = 35
    #
    # thresh = 10
    # res = []
    # centers = []
    # elements = []
    # show = False

    # res = load(folder)
    # images = os.listdir(folder)
    # #print(res.size)
    # # centers = np.load('centers')
    # print(len(res))
    # for i in range(len(res)):
    #
    #     if show:
    #         fig, ax = plt.subplots(1)
    #         ax.imshow(Image.open(os.path.join(folder, images[i])))
    #     for j in range(len(res[i])):
    #         x1, y1, x2, y2, conf, cls_conf, cls_pred = res[i][j]
    #
    #         # Display the image
    #         if show:
    #             rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    #             ax.add_patch(rect)
    #         centers.append((int((x1 + x2) / 2), (int((y1 + y2) / 2))))
    #         elements.append(res[i][j])
    #     if show:
    #         plt.show()
    #
    # print(f'Cars detected: {len(elements)}')
    #
    # # TRYOUT CUSTOM CLUSTERING
    # clusters = _my_cluster(elements)
    #
    # # Clustering
    # #centers = np.array(centers)
    # #clusters = hcluster.fclusterdata(centers, thresh, criterion="distance")
    #
    # fig, ax = plt.subplots(1)
    # ax.imshow(Image.open(os.path.join(folder, images[0])))
    # plot_clusters(ax,centers,clusters)
    # plt.show()
    #
    # # remove all small clusters
    # unique, counts = np.unique(clusters, return_counts=True)
    # cluster_count = zip(unique, counts)
    # for cc in cluster_count:
    #     if cc[1] < min_cluster_size:
    #         clusters[clusters == cc[0]] = -1
    # centers = [cent for cent, clus in zip(centers, clusters) if clus != -1]
    # elements = [elem for clust, elem in zip(clusters, elements) if clust != -1]
    # clusters = [elem for elem in clusters if elem != -1]
    #
    # fig, ax = plt.subplots(1)
    #
    # # intersecting:
    # for cluster in np.unique(clusters):
    #     intersect = get_intersection2((np.array(elements)[np.array(clusters) == cluster]).tolist())
    #     # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    #     if type(intersect) == polygon.Polygon:
    #         x, y = intersect.exterior.xy
    #         ax.plot(x,y)
    # # plotting
    #
    # ax.imshow(Image.open(os.path.join(folder,images[0])))
    # plot_clusters(ax,centers,clusters)
    # ax.axis("equal")
    # title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
    # plt.title(title)
    #
    #     # ax.add_patch(rect)
    # plt.show()
    # # plt.imshow(, alpha= 0.5)
    # # plt.show()


if __name__ == '__main__':
    _main()
