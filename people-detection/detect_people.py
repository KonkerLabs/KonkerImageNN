from yolo import yolov3
import os
import matplotlib.pyplot as plt
from PIL import Image
from time import sleep
from argparse import ArgumentParser
from matplotlib.patches import Rectangle
import utils.utils as utils
import json


class PeopleDetector:
    class DEFAULTS:
        output_location = './output'
        show = False
        img_size = 1920
        send_to_konker = False
        output_count = False
        output_count_file = 'output_counts.json'

    # region Constructors and Destructors
    def __init__(self, folder, output_location=DEFAULTS.output_location, show=DEFAULTS.show, img_size=DEFAULTS.img_size,
                 send_to_konker=DEFAULTS.send_to_konker, output_count=DEFAULTS.output_count,
                 konker_username=utils.KonkerSender.DEFAULTS.username,
                 konker_password=utils.KonkerSender.DEFAULTS.password):
        self._output_location = output_location
        self._yoloV3 = yolov3.Yolov3(img_size=img_size)
        self._working_folder = folder
        self._show = show
        self._tracker = utils.FileCreatedTracker(self._working_folder, self.load_people, args_array=True)
        self._send_to_konker = send_to_konker
        self._output_count = output_count
        if self._send_to_konker:
            self._konker_sender = utils.KonkerSender(username=konker_username, password=konker_password)

    # endregion
    #
    # # region Track folder
    # def track_live(self):
    #     self._event_handler = utils.FileCreatedHandler(created_callback=self.add_to_queue)
    #     self._detect_thread_running = True
    #     self.load_all_people()
    #     t1 = threading.Thread(target=self.run_detect_queue)
    #     t1.start()
    #     self._observer = Observer()
    #     self._observer.schedule(self._event_handler, self._working_folder, recursive=True)
    #     self._observer.start()
    #
    # def stop_tracking(self):
    #     if self._observer:
    #         self._observer.stop()
    #
    # # endregion
    # region Track folder

    def track_live(self):
        self._tracker.start()

    def stop_tracking(self):
        self._tracker.stop()

    # endregion

    # region Load people of images
    def load_all_people(self):
        _list = [os.path.join(self._working_folder, file) for file in os.listdir(self._working_folder) if
                 os.path.isfile(os.path.join(self._working_folder, file))]
        return self.load_people(_list)

    def load_people(self, file_list):
        files = []
        data = dict()
        if self._output_count:
            outputs = dict()
            output_file = os.path.join(self._output_location, self.DEFAULTS.output_count_file)
            if os.path.exists(output_file):
                outputs = json.load(open(output_file, "r"))
        for file in file_list:
            if file not in data.keys():
                files.append(file)
        if len(files) > 0:
            res = self._yoloV3.yolo(files, return_mask=False, classes=(0,))
            for i in range(len(files)):
                data[files[i]] = res[i]
                if self._output_count:
                    timestamp = int(os.path.getmtime(files[i]) * 1000)
                    count = len(res[i]) if res[i] is not None else 0
                    outputs[files[i]] = (timestamp, count)
                    json.dump(outputs, open(output_file, "w+"))
                    if self._send_to_konker:
                        self._konker_sender.send("people_count",
                                                 {
                                                     '_ts': timestamp+10,
                                                     'people': count
                                                 })
                else:
                    self.save_image(files[i], res[i], show=self._show)
        print(f'Processed {len(files)} file(s)')
        return data

    # endregion

    # # region File processing queue
    #
    # def add_to_queue(self, path):
    #     self._file_queue.put(path)
    #
    # def run_detect_queue(self):
    #     print('Waiting for new images...')
    #     while self._detect_thread_running:
    #         file_list = list()
    #         file_list.append(self._file_queue.get())
    #         sleep(3)
    #         if self._file_queue.qsize() > 0:
    #             for i in range(self._file_queue.qsize()):
    #                 file_list.append(self._file_queue.get())
    #         if len(file_list) > 0:
    #             self.load_people(file_list)
    #
    # # endregion
    def save_image(self, img_path, data, show=False):
        im = Image.open(img_path)
        plt.imshow(im)
        if data is not None:
        # Get the current reference
            ax = plt.gca()
            for rect in data:
                # Create a Rectangle patch
                rect = Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
        if show:
            plt.show()
        plt.savefig(os.path.join(self._output_location, os.path.basename(img_path)))

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


def _main():
    parser = ArgumentParser('Detect parking lots in parking lot images')
    parser.add_argument('folder', type=str, help='Folder containing the images')
    parser.add_argument("--mode", type=str, choices=['live', 'all'],
                        help="Runnning mode")
    parser.add_argument("--output_mode", type=str, choices=['count', 'images'], default="images",
                        help="Output mode")
    parser.add_argument("--output_location", default=PeopleDetector.DEFAULTS.output_location, type=str,
                        help="Location of all outputs")
    parser.add_argument("--img-size", default=PeopleDetector.DEFAULTS.img_size, type=int,
                        help="Size of the images to process")
    parser.add_argument("--send_to_konker", type=bool, default=False,
                        help="If set to true the people count is sent to konker")
    utils.KonkerSender.add_credential_options(parser)

    args = parser.parse_args()

    if args.send_to_konker and args.output_mode != 'count':
        print("If data should be sent to konker the output mode has to be 'count'.")
        parser.print_usage()
        return

    pld = PeopleDetector(args.folder, output_location=args.output_location, img_size=args.img_size,
                         send_to_konker=args.send_to_konker, output_count=(args.output_mode == 'count'))
    if args.mode == 'live':
        pld.track_live()

        while True:
            sleep(5)
    else:
        pld.load_all_people()

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
