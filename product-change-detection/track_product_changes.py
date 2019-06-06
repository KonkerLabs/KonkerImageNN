from image_changes import Utils, DifferenceDetection
from argparse import ArgumentParser
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import queue
import time
import os
import requests
from requests.auth import HTTPBasicAuth
import json
import numpy as np

def __main():
    # [ color_cnt, oversizing_horiz, oversizing_vert, num_classes, scale_factor,
    # color diff_threshold, pre_scale_size, change_threshold]

    parser = ArgumentParser('Detecting Product changes')
    parser.add_argument('folder', type=str,
                        help='Folder where new images appear')
    parser.add_argument("--scale_factor", default=0.2, type=float,
                        help="Scale factor")
    parser.add_argument("--change_threshold", default=0.32, type=float,
                        help="Changes below that threshold are not considered")
    parser.add_argument("--oversizing_vertical", default=15, type=int,
                        help="Oversizing vertically")
    parser.add_argument("--oversizing_horizontal", default=10, type=int,
                        help="Oversizing horizontal")
    parser.add_argument("--color_diff_threshold", default=60, type=float,
                        help="Color differences below that threshold are not considered")
    parser.add_argument("--num_classes", default=30, type=int,
                        help="Number of classes used for the change detection")
    parser.add_argument("--color_cnt", default=7, type=int,
                        help="Number of dominant colors to compare")
    parser.add_argument("--pre_scale", default=(384, 384), type=tuple,
                        help="Scaling before the change detection")
    parser.add_argument("--prefix", default='changes', type=str,
                        help="File prefix")
    parser.add_argument("--username", default="t0vbkqelr9hi", type=str,
                        help="Konker username to send change notifications to")
    parser.add_argument("--password", default="RBU31lv8T1Yk", type=str,
                        help="Konker password to send change notifications to")
    args = parser.parse_args()
    det_proc = DetectionProcessor(args)
    tracker = Tracker(args.folder, callback=det_proc.callback)

    tracker.track_live()
    while True:
        time.sleep(5)


class DetectionProcessor:
    def __init__(self, args):
        self.args = args
        self.dif_det = DifferenceDetection(file_prefix=args.prefix, caching=True)

    def callback(self, f1, f2):
        result = self.dif_det.calculate_differences(f1, f2, self.args.scale_factor, self.args.change_threshold,
                                                    self.args.color_cnt,
                                                    oversizing_x=self.args.oversizing_horizontal,
                                                    oversizing_y=self.args.oversizing_vertical,
                                                    overwrite=False,
                                                    color_diff_threshold=self.args.color_diff_threshold,
                                                    num_classes=self.args.num_classes,
                                                    pre_scale_size=self.args.pre_scale)
        if result:

            if result[0].sum() > 0:
                print('Changes detected')
                self.send_detection_results((np.round(result[0].numpy(), decimals=2)*100).astype(int).tolist())
            else:
                print('No changes detected')

    def send_detection_results(self, changes):
        publication_url = f"https://data.demo.konkerlabs.net:443/pub/{self.args.username}/change_detection"
        data = json.dumps({'_ts': int(time.time() * 1000), 'detected_changes': changes})
        result = requests.post(publication_url, data, auth=HTTPBasicAuth(self.args.username, self.args.password))
        if result.status_code == 200:
            return True
        else:
            raise Exception(f'Sending data to konker failed {result.status_code}: {result.reason}')


class MyHandler(FileSystemEventHandler):

    def __init__(self, created_callback=None):
        self._created_callback = created_callback

    def on_created(self, event):
        if callable(self._created_callback):
            self._created_callback(event.src_path)
        print(f'New file detected: {event.src_path}')


class Tracker:

    def __init__(self, working_folder, callback):
        self._file_queue = queue.Queue()
        self._event_handler = None
        self._observer = None
        self._working_folder = working_folder
        self._detect_thread_running = False
        self._callback = callback

    def __del__(self):
        self.stop_tracking()

    def track_live(self):
        self._event_handler = MyHandler(created_callback=self.add_to_queue)
        self._detect_thread_running = True
        t1 = threading.Thread(target=self.run_detect_queue)
        t1.start()
        self._observer = Observer()
        self._observer.schedule(self._event_handler, self._working_folder, recursive=True)

        self._observer.start()

    def stop_tracking(self):
        if self._observer:
            self._observer.stop()

    def run_detect_queue(self):
        print('Waiting for new images...')
        while self._detect_thread_running:
            file1 = self._file_queue.get()
            file2 = self.get_file_before(file1)

            if file2 is not None:
                print(f'Processing {file1} and {file2}')
                self._callback(file1, file2)
            else:
                print(f'No image before {file1} found.')

    def add_to_queue(self, path):
        self._file_queue.put(path)

    def get_file_before(self, path):
        files = [os.path.join(self._working_folder, file) for file in os.listdir(self._working_folder) if
                 os.path.isfile(os.path.join(self._working_folder, file))]
        files.sort(key=lambda x: os.path.getmtime(path) - os.path.getmtime(x))
        for f in files:
            if os.path.getmtime(path) - os.path.getmtime(f) > 0:
                return f
        return None


if __name__ == '__main__':
    __main()
