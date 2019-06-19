from watchdog.events import FileSystemEventHandler
import requests
from requests.auth import HTTPBasicAuth
import json
import time
from watchdog.observers import Observer
import threading
import queue
from time import sleep


class FileCreatedHandler(FileSystemEventHandler):

    def __init__(self, created_callback=None):
        self._created_callback = created_callback

    def on_created(self, event):
        if callable(self._created_callback):
            self._created_callback(event.src_path)
        print(f'New file found: {event.src_path}')


class KonkerSender:
    class DEFAULTS:
        username = "t1e7udp3v5rq"
        password = "uSzHe6hZ3KyP"

    def __init__(self, username=DEFAULTS.username, password=DEFAULTS.password):
        self.username = username
        self.password = password
        self.publication_url = f"https://data.demo.konkerlabs.net:443/pub/{self.username}"

    def send(self, channel, data):
        result = requests.post(f"{self.publication_url}/{channel}", json.dumps(data), auth=HTTPBasicAuth(self.username, self.password))
        if result.status_code == 200:
            return True
        else:
            raise Exception(f'Sending data to konker failed {result.status_code}: {result.reason} [{result.content}]')

    def send_detection_results(self, changes):
        data = {'_ts': int(time.time() * 1000), 'detected_changes': changes}
        return self.send("change_detection", data)
    
    @classmethod
    def add_credential_options(cls, parser):
        parser.add_argument("--konker_username", default=cls.DEFAULTS.username, type=str,
                            help="Konker username to send change notifications to")
        parser.add_argument("--konker_password", default=cls.DEFAULTS.password, type=str,
                            help="Konker password to send change notifications to")


class FileCreatedTracker:

    def __init__(self, working_folder, callback, args_array=False):
        self._file_queue = queue.Queue()
        self._event_handler = None
        self._observer = None
        self._working_folder = working_folder
        self._detect_thread_running = False
        self._callback = callback
        self.args_array = args_array

    def __del__(self):
        self.stop()

    def start(self):
        self._event_handler = FileCreatedHandler(created_callback=self.add_to_queue)
        self._detect_thread_running = True
        t1 = threading.Thread(target=self.run_detect_queue)
        t1.start()
        self._observer = Observer()
        self._observer.schedule(self._event_handler, self._working_folder, recursive=True)

        self._observer.start()

    def stop(self):
        if self._observer:
            self._observer.stop()

    def run_detect_queue(self):
        print('Waiting for new images...')
        while self._detect_thread_running:
            file = self._file_queue.get()
            if self.args_array:
                file_list = list()
                file_list.append(file)
                sleep(3)
                if self._file_queue.qsize() > 0:
                    for i in range(self._file_queue.qsize()):
                        file_list.append(self._file_queue.get())
                if len(file_list) > 0:
                    self._callback(file_list)
            else:
                self._callback(file)

    def add_to_queue(self, path):
        self._file_queue.put(path)

