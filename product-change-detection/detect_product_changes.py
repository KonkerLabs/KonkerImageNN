from image_changes import Utils, DifferenceDetection
from argparse import ArgumentParser


def __main():
    # [ color_cnt, oversizing_horiz, oversizing_vert, num_classes, scale_factor,
    # color diff_threshold, pre_scale_size, change_threshold]

    parser = ArgumentParser('Detecting Product changes')
    parser.add_argument('images', type=str, nargs=2,
                        help='Pathes to images')
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
    args = parser.parse_args()

    dif_det = DifferenceDetection(file_prefix=args.prefix)
    dif_det.calculate_differences(args.images[0], args.images[1], args.scale_factor, args.change_threshold, args.color_cnt,
                                  oversizing_x=args.oversizing_horizontal, oversizing_y=args.oversizing_vertical,
                                  overwrite=False,
                                  color_diff_threshold=args.color_diff_threshold, num_classes=args.num_classes,
                                  pre_scale_size=args.pre_scale)


if __name__ == '__main__':
    __main()
