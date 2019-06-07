# KonkerImageNN
## Parkinglot detection
To detect parking lots the `parkinglot-detection.py` can be exectuted in two different modes:
- `live`: Tracks the given folder and detects car positions out of all existing and newly incoming images and saves those detections.
- `detect`: Uses the detected car positions and detects potential parking lots.

The ouput location is used to save the image with the detected parking lots and a file with the detected parking lots to reload.
The buffer location containes the buffer files with the detected cars and should be consistent if first `live` mode is used and then `detect` mode.
```
usage: Detect parking lots in parking lot images [-h] [--mode {live,detect}]
                                                 [--output_location OUTPUT_LOCATION]
                                                 [--buffer_location BUFFER_LOCATION]
                                                 [--min_cluster_size MIN_CLUSTER_SIZE]
                                                 [--clustering_eps CLUSTERING_EPS]
                                                 folder

positional arguments:
  folder                Folder containing the images

optional arguments:
  -h, --help            show this help message and exit
  --mode {live,detect}  Runnning mode
  --output_location OUTPUT_LOCATION
                        Location of all outputs
  --buffer_location BUFFER_LOCATION
                        Location for buffer locations
  --min_cluster_size MIN_CLUSTER_SIZE
                        Minimum Cluster size
  --clustering_eps CLUSTERING_EPS
                        EPS value for the clustering

```

## Product change detection
### `detect_product_changes.py` 
Can be used to manually detect product changes between two different images.

```
usage: Detecting Product changes [-h] [--scale_factor SCALE_FACTOR]
                                 [--change_threshold CHANGE_THRESHOLD]
                                 [--oversizing_vertical OVERSIZING_VERTICAL]
                                 [--oversizing_horizontal OVERSIZING_HORIZONTAL]
                                 [--color_diff_threshold COLOR_DIFF_THRESHOLD]
                                 [--num_classes NUM_CLASSES]
                                 [--color_cnt COLOR_CNT]
                                 [--pre_scale PRE_SCALE] [--prefix PREFIX]
                                 [--mask_path MASK_PATH]
                                 images images

positional arguments:
  images                Pathes to images

optional arguments:
  -h, --help            show this help message and exit
  --scale_factor SCALE_FACTOR
                        Scale factor
  --change_threshold CHANGE_THRESHOLD
                        Changes below that threshold are not considered
  --oversizing_vertical OVERSIZING_VERTICAL
                        Oversizing vertically
  --oversizing_horizontal OVERSIZING_HORIZONTAL
                        Oversizing horizontal
  --color_diff_threshold COLOR_DIFF_THRESHOLD
                        Color differences below that threshold are not
                        considered
  --num_classes NUM_CLASSES
                        Number of classes used for the change detection
  --color_cnt COLOR_CNT
                        Number of dominant colors to compare
  --pre_scale PRE_SCALE
                        Scaling before the change detection
  --prefix PREFIX       File prefix
  --mask_path MASK_PATH
                        Black/White mask to remove not needed parts of the
                        image (black: not relevant, white: relevant)

```
Implementations of the parking-lot detection and the product change detection
