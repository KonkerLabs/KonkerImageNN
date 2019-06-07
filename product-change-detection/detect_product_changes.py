from image_changes import Utils, DifferenceDetection
from argparse import ArgumentParser


def __main():
    # [ color_cnt, oversizing_horiz, oversizing_vert, num_classes, scale_factor,
    # color diff_threshold, pre_scale_size, change_threshold]

    parser = ArgumentParser('Detecting Product changes')
    parser.add_argument('images', type=str, nargs=2,
                        help='Pathes to images')
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
    args = parser.parse_args()

    dif_det = DifferenceDetection(file_prefix=args.prefix, export_image=True, mask_path=args.mask_path,
                                  areas_path=args.areas_path)
    dif_det.calculate_differences(args.images[0], args.images[1], args.scale_factor, args.change_threshold,
                                  args.color_cnt,
                                  oversizing_horizontal=args.oversizing_horizontal,
                                  oversizing_vertical=args.oversizing_vertical,
                                  overwrite=False,
                                  color_diff_threshold=args.color_diff_threshold, num_classes=args.num_classes,
                                  pre_scale_size=args.pre_scale)


if __name__ == '__main__':
    __main()
