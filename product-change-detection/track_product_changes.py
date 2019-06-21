from image_changes import DifferenceDetection
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
import os
import numpy as np
import utils.utils as utils


def __main():
    # [ color_cnt, oversizing_horiz, oversizing_vert, num_classes, scale_factor,
    # color diff_threshold, pre_scale_size, change_threshold]

    parser = ArgumentParser('Detecting Product changes', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', type=str,
                        help='Folder where new images appear')
    parser.add_argument("--scale_factor", default=DifferenceDetection.DEFAULTS.scale_factor, type=float,
                        help="Scale factor")
    parser.add_argument("--change_threshold", default=DifferenceDetection.DEFAULTS.change_threshold, type=float,
                        help="Changes below that threshold are not considered")
    parser.add_argument("--oversizing_vertical", default=DifferenceDetection.DEFAULTS.oversizing_horiz, type=int,
                        help="Oversizing vertically")
    parser.add_argument("--oversizing_horizontal", default=DifferenceDetection.DEFAULTS.oversizing_horiz, type=int,
                        help="Oversizing horizontal")
    parser.add_argument("--color_diff_threshold", default=DifferenceDetection.DEFAULTS.color_diff_threshold, type=float,
                        help="Color differences below that threshold are not considered")
    parser.add_argument("--num_classes", default=DifferenceDetection.DEFAULTS.num_classes, type=int,
                        help="Number of classes used for the change detection")
    parser.add_argument("--color_cnt", default=DifferenceDetection.DEFAULTS.color_cnt, type=int,
                        help="Number of dominant colors to compare")
    parser.add_argument("--pre_scale", default=DifferenceDetection.DEFAULTS.pre_scale_size, type=tuple,
                        help="Scaling before the change detection")
    parser.add_argument("--prefix", default=DifferenceDetection.DEFAULTS.file_prefix, type=str,
                        help="Output file prefix")
    parser.add_argument("--mask_path", default=DifferenceDetection.DEFAULTS.mask_path, type=str,
                        help="Black/White mask to remove not needed parts of the image (black: not relevant, white: relevant)")
    parser.add_argument("--areas_path", default=DifferenceDetection.DEFAULTS.areas_path, type=str,
                        help="Json file with containing a single multi dimensional array with 4 coordinates for the shelf areas for every element.")
    utils.KonkerSender.add_credential_options(parser)
    args = parser.parse_args()
    det_proc = DetectionProcessor(args)
    tracker = utils.FileCreatedTracker(args.folder, callback=det_proc.callback)

    tracker.start()
    while True:
        time.sleep(5)


class DetectionProcessor:
    def __init__(self, args):
        self.args = args
        self.dif_det = DifferenceDetection(file_prefix=args.prefix, caching=True, mask_path=args.mask_path,
                                           areas_path=args.areas_path)
        self.sender = utils.KonkerSender(self.args.username, self.args.password)

    def callback(self, f1):
        f2 = self.get_file_before(f1)

        if f2 is not None:
            print(f'Processing {f1} and {f2}')
        else:
            print(f'No image before {f1} found.')
            return None

        result = self.dif_det.calculate_differences(f1, f2, self.args.scale_factor, self.args.change_threshold,
                                                    self.args.color_cnt,
                                                    oversizing_horizontal=self.args.oversizing_horizontal,
                                                    oversizing_vertical=self.args.oversizing_vertical,
                                                    overwrite=False,
                                                    color_diff_threshold=self.args.color_diff_threshold,
                                                    num_classes=self.args.num_classes,
                                                    pre_scale_size=self.args.pre_scale)
        if result:

            if result[0].sum() > 0:
                print('Changes detected')
                self.sender.send_detection_results((np.round(result[0].numpy(), decimals=2) * 100).astype(int).tolist())
            else:
                print('No changes detected')

    def get_file_before(self, path):
        files = [os.path.join(self.args.folder, file) for file in os.listdir(self.args.folder) if
                 os.path.isfile(os.path.join(self.args.folder, file))]
        files.sort(key=lambda x: os.path.getmtime(path) - os.path.getmtime(x))
        for f in files:
            if os.path.getmtime(path) - os.path.getmtime(f) > 0:
                return f
        return None


if __name__ == '__main__':
    __main()
