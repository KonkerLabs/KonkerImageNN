from yolo import yolov3
import os
import matplotlib.pyplot as plt
from PIL import Image
from time import sleep
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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


def _main():
    parser = ArgumentParser('Detect parking lots in parking lot images', formatter_class=ArgumentDefaultsHelpFormatter)
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


if __name__ == '__main__':
    _main()
