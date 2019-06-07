from torchvision import transforms
from PIL import Image
from yolo import yolov3
from unet_models import *
from sklearn.cluster import MiniBatchKMeans  # KMeans,
import numpy as np
from multiprocessing import Pool, cpu_count

from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import os
import operator
import sys
from matplotlib.path import Path
import json
from tqdm import tqdm
import time
import logging
from threading import Thread

area = []


class DifferenceDetection:
    _cache = dict()
    _logger = logging.getLogger('DifferenceDetection')
    _model = None
    _num_classes = 0

    class DEFAULTS:
        file_prefix = 'DifferenceDetection'
        output_dir = './output'
        cuda = True
        caching = False
        export_image = False
        oversizing_horiz = 10  # x
        oversizing_verti = 15  # y
        scale_factor = 0.2
        change_threshold = 0.32
        color_diff_threshold = 60
        num_classes = 30
        color_cnt = 7
        pre_scale_size = (384, 384)
        pre_scale_factor = 1
        mask_path = 'conf/shelf_mask.png'
        areas_path = 'conf/areas.json'

    def __init__(self, file_prefix=DEFAULTS.file_prefix, output_dir=DEFAULTS.output_dir, cuda=DEFAULTS.cuda,
                 caching=DEFAULTS.caching, export_image=DEFAULTS.export_image, mask_path=DEFAULTS.mask_path, areas_path=DEFAULTS.areas_path, ):
        Utils.init_logging()
        self._file_prefix = file_prefix
        self._output_dir = output_dir
        self._cuda = cuda
        self._yolo = yolov3.Yolov3()
        self._caching_enabled = caching
        self._export_image = export_image
        self._areas_path = areas_path
        self._mask_path = mask_path
        Utils.set_areas_file(areas_path)
        if self._cuda:
            torch.cuda.empty_cache()

    @classmethod
    def prepare_tensor(cls, data1, data2, num_classes, threshold=0):
        sum_tens = torch.zeros(data1[0].size())
        for cnt in range(0, num_classes):
            temp_tens = (data1[cnt] - data2[cnt]).abs()
            temp_tens[temp_tens < threshold] = 0
            sum_tens += temp_tens

        # sum_tens = sum_tens / torch.max(sum_tens)
        return sum_tens

    '''
    def prepare_tensor_test(self, data1, data2, num_classes):
        thres = 0
        sum_tens = torch.zeros(data1[:, :, 0].size())
        for cnt in range(0, num_classes):
            temp_tens = (data1[:, :, cnt] - data2[:, :, cnt]).abs()
            temp_tens[temp_tens < thres] = 0
            sum_tens += temp_tens

        sum_tens = sum_tens / torch.max(sum_tens)
        return sum_tens

    def trygetdif(self, img1, img2):
        pretrained = torchfcn.models.FCN8s.download()

        model = torchfcn.models.FCN8s(n_class=21)
        model_data = torch.load(pretrained)

        try:
            model.load_state_dict(model_data)
        except Exception:
            model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        img1_ = Image.open(img1)
        scale_fac = 1
        img1 = transforms.Resize((int(img1_.size[1] / scale_fac), int(img1_.size[0] / scale_fac)))(img1_)
        img1 = Utils.TO_TENSOR(img1)
        img2_ = Image.open(img2)
        img2 = transforms.Resize((int(img2_.size[1] / scale_fac), int(img2_.size[0] / scale_fac)))(img2_)
        img2 = Utils.TO_TENSOR(img2)
        # img1 = torch.from_numpy(img1).float()
        # img2 = torch.from_numpy(img2).float()
        # img1 = self.transform(img1)
        # img2 = self.transform(img2)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        # with torch.no_grad():
        model.eval()
        res1 = model(img1)[0]
        res2 = model(img2)[0]

        # print(res1.sum())
        # print(res1.size())
        # print(res2.size())
        print(res1.size())
        print(res2.size())
        # res1 = torch.from_numpy(self.untransform(res1) .copy())
        print(res1.size())
        # res2 = torch.from_numpy(self.untransform(res2).copy())
        res = self.prepare_tensor(res1, res2, 21)
        print(scale_factor)
        # res = Utils.tens_scale_2d(res, scale_factor=scale_factor)
        # print(res.size())
        res = res / res.max()
        # res[res < 0.37] = 0
        plt.imshow(img1_)
        plt.imshow(img2_, alpha=0.5)
        plt.imshow(Utils.TO_PIL(res).resize(img1_.size, resample=Image.NEAREST), cmap='hot', alpha=0.5)
        # plt.imshow(Utils.TO_PIL(res1), cmap='hot', alpha=0.5)
        # plt.imshow(Utils.TO_PIL(res2), cmap='hot', alpha=0.5)

        plt.axis('off')
        plt.show()
        # plt.savefig('test1234', bbox_inches='tight', dpi=200)
        plt.close()
        print('test')


    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= np.array([104.00698793, 116.66876762, 122.67891434])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img):
        img = img.detach().numpy()
        img = img.transpose(1, 2, 0)
        img += np.array([104.00698793, 116.66876762, 122.67891434])
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img'''

    def calculate_differences(self, img_path1, img_path2, scale_factor, change_threshold, color_cnt,
                              oversizing_horizontal=DEFAULTS.oversizing_horiz,
                              oversizing_vertical=DEFAULTS.oversizing_verti, pre_scale_factor=DEFAULTS.pre_scale_factor, overwrite=False, color_diff_threshold=DEFAULTS.color_diff_threshold,
                              out_color_diff=True, out_diff=False, num_classes=DEFAULTS.num_classes, pre_scale_size=DEFAULTS.pre_scale_size,
                               evaluation_mode=False):
        # identifier = f'{img_path1}_{img_path2}{self._file_prefix}_scale_fac={scale_factor}/_change_thres='\
        #              f'{change_threshold}/_colors={color_cnt}_oversizing_x={oversizing_x}_oversizing_y='\
        #              f'{oversizing_y}_num_classes={num_classes}_color_diff_threshold={color_diff_threshold}_'\
        #              f'pre_scale={pre_scale_factor if pre_scale_factor != 1 else pre_scale_size}'
        Utils.timing()
        if evaluation_mode:
            identifier = f'{self._file_prefix}_scale_fac={scale_factor}_change_thres={change_threshold}_' \
                f'colors={color_cnt}_oversizing_x={oversizing_horizontal}_oversizing_y={oversizing_vertical}_num_classes={num_classes}_' \
                f'color_diff_threshold={color_diff_threshold}_pre_scaltede=' \
                f'{pre_scale_factor if pre_scale_factor != 1 else pre_scale_size}/{img_path1.replace("/","")}_'\
                f'{img_path2.replace("/", "")}'
        else:
            identifier = f'{img_path1.replace("/", "_")}_{img_path2.replace("/", "_")}'

            identifier = identifier.replace('(', '')
            identifier = identifier.replace(')', '')
            identifier = identifier.replace(' ', '')

        self._logger.info(f'Calculating differences for: {identifier}')

        filename_out = f'{self._output_dir}/{identifier}_out.png'
        filename_change = f'{self._output_dir}/{identifier}_out_change.png'

        if self._export_image:
        # Check if it was already executed
            if not (overwrite or Utils.DEBUG) and os.path.isfile(filename_out):
                self._logger.info(f'{identifier} already ran!')
                return None
            else:
                if not os.path.isdir(filename_out[:filename_out.rfind('/')]):
                    os.makedirs(filename_out[:filename_out.rfind('/')])
                os.system(f'touch {filename_out}')


        # Load images from cache or disk

        if self._caching_enabled and img_path1 in self._cache.keys():
            img1_ = self._cache[img_path1]
        else:
            img1_ = Image.open(img_path1)
            self._cache[img_path1] = img1_

        if self._caching_enabled and img_path2 in self._cache.keys():
            img2_ = self._cache[img_path2]
        else:
            img2_ = Image.open(img_path2)
            self._cache[img_path2] = img2_

        if self._caching_enabled and 'mask' in self._cache.keys():
            _mask = self._cache['mask']
        else:
            _mask = Image.open(self._mask_path)
            _mask = _mask.convert('1')
            self._cache['mask'] = _mask

        # Check image size
        if img1_.size != img2_.size:
            raise Exception('Images are not the same size')

        img_size_full = Utils.TO_TENSOR(img1_).size()[1:]

        img1 = img1_.copy()
        img2 = img2_.copy()

        # Scale Image
        if pre_scale_factor != 1:
            sz1, sz2 = img1.size
            img1 = img1.resize(size=(int(sz1 * pre_scale_factor), int(sz2 * pre_scale_factor)))
            img2 = img2.resize(size=(int(sz1 * pre_scale_factor), int(sz2 * pre_scale_factor)))
            _mask = _mask.resize(size=(int(sz1 * pre_scale_factor), int(sz2 * pre_scale_factor)))
        if pre_scale_size:
            img1 = img1.resize(size=pre_scale_size)
            img2 = img2.resize(size=pre_scale_size)
            _mask = _mask.resize(size=pre_scale_size)

        # Convert everything to tensors
        img1 = Utils.TO_TENSOR(img1)
        img2 = Utils.TO_TENSOR(img2)
        _mask = Utils.TO_TENSOR(_mask)

        img1 = img1 * _mask
        img2 = img2 * _mask

        change_id = f'{pre_scale_factor}_{pre_scale_size}_{img_path1}_{img_path2}'

        Utils.timing(name='Prepare')

        # Classification
        if change_id in self._cache.keys() and 'yolo' in self._cache[change_id] and 'diff' in self._cache[change_id]:
            yolo = self._cache[change_id]['yolo']
            diff = self._cache[change_id]['diff'].clone()
            Utils.timing(name='Loading from Cache')
        else:

            # diff = differences.getDiffTens(img_path1, img_path2, cuda=self._cuda,
            #                                pretrained_dir='./DiffDetection/pretrained',
            #                                num_classes=num_classes, model="baseline-resnet101-upernet", epoch=50,
            #                                fc_dim=2048, arch_encoder='resnet101', arch_decoder='upernet').cpu()
            if not self._model or self._num_classes != num_classes:
                self._model = Utils.get_model(num_classes)
                self._model.share_memory()
            with torch.no_grad():
                img1__ = self._model(img1.clone().unsqueeze(0))[0]
                img2__ = self._model(img2.clone().unsqueeze(0))[0]
                diff = self.prepare_tensor(img1__, img2__, num_classes)
            Utils.timing(name='Differences')
            yolo = self._yolo.yolo([img_path1, img_path2])
            Utils.timing(name='Yolo')
            if self._caching_enabled:
                self._cache[change_id] = dict()
                self._cache[change_id]['yolo'] = yolo
                self._cache[change_id]['diff'] = diff.clone()
        diff = diff * 7
        # print(f'{diff.max()} {diff.min()}')

        # Filter People and other objects from the changes.
        diff_without_objects = diff * Utils.tens_scale_2d(yolo[0] * yolo[1], size=diff.size())

        # Upscale the differences and apply the threshold
        dif_scaled = Utils.tens_scale_2d(diff_without_objects.clone(), scale_factor=scale_factor,
                                         mode='nearest')  # , mode='area'
        dif_scaled[dif_scaled < change_threshold] = 0
        diff_without_objects[diff_without_objects < change_threshold] = 0

        result = self.check_coverage(Utils.tens_scale_2d(diff_without_objects.clone(), size=img_size_full))
        thres, _ = Utils.get_coverage_thresholds()[-1]
        changed = False
        for row in result:
            for diff, _ in row:
                if diff >= thres:
                    changed = True
                    break
            if changed:
                break
        if changed:
            color_comp = DifferencesColorComparator(color_cnt=color_cnt,
                                                    color_mode=DifferencesColorComparator.DeltaECIEMode)
            img1_tens = Utils.TO_TENSOR(img1_)
            img2_tens = Utils.TO_TENSOR(img2_)
            # img1_tens = Utils.tens_scale_2d(img1_tens, scale_factor=0.5)
            # img2_tens = Utils.tens_scale_2d(img2_tens, scale_factor=0.5)
            color_differences = color_comp.compare_colors_of_differences(dif_scaled, img1_tens,
                                                                         img2_tens, color_cnt,
                                                                         oversizing_x=oversizing_horizontal,
                                                                         oversizing_y=oversizing_vertical,
                                                                         change_threshold=change_threshold)
            Utils.timing(name='Color Differences')
            # color_differences = Utils.tens_scale_2d(color_differences, scale_factor=1, mode='bilinear')
            color_differences[color_differences < color_diff_threshold] = 0
            # Utils.remove_non_coherent_values(color_differences)
            color_differences = Utils.tens_scale_2d(color_differences, size=img_size_full)
            ten = Utils.TO_PIL(dif_scaled).resize(img1_.size, resample=Image.NEAREST,
                                                  box=None)  # diff_without_objects[0]   dif_scaled
            mask, result, values = self.check_areas(color_differences)
            product_changed = Utils.TO_PIL(mask)
        else:
            self._logger.info(f"Nothing changed between {img_path1} and {img_path2}")
            return torch.zeros((len(result), len(result[0]))), torch.zeros((len(result), len(result[0])))
        Utils.timing(name='Finished')
        if self._export_image:
            if out_color_diff:
                Utils.output_image(img1_, img2_, product_changed, filename_out)
            if out_diff:
                Utils.output_image(img1_, img2_, ten, filename_change)
        self._logger.info(f'Finished calculating differences for: {identifier}')
        return result, values


    @classmethod
    def check_areas(cls, color_diffs):
        full_mask = torch.zeros_like(color_diffs)
        arrs = cls.check_coverage(color_diffs)
        result_arr = torch.zeros(len(arrs), len(arrs[0]))
        diffs = torch.zeros(len(arrs), len(arrs[0]))
        for i in range(len(arrs)):
            for j in range(len(arrs[i])):
                diff, mask = arrs[i][j]
                diffs[i][j] = diff
                for thres, factor in Utils.get_coverage_thresholds():
                    if diff > thres:  # 50 55
                        full_mask = full_mask + (mask / factor)
                        result_arr[i][j] = round(1.0 / factor, 2)
                        break
        full_mask[0][0] = 1
        return full_mask, result_arr, diffs


    @classmethod
    def check_coverage(cls, tens):
        areas = Utils.get_areas()
        arr2 = torch.zeros(len(areas), len(areas[0])).tolist()
        size = tens.size()
        for i in range(len(areas)):
            for j in range(len(areas[i])):
                ar = areas[i][j]
                if ar is None:
                    continue
                color_differences = tens.clone()
                x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
                x, y = x.flatten(), y.flatten()
                path1 = Path(ar)
                points = np.vstack((x, y)).T
                grid = path1.contains_points(points)
                mask = torch.from_numpy(grid.reshape((size[0], size[1])).astype(float)).float()
                color_differences = color_differences * mask
                _sum = mask.sum().item()
                cnt = color_differences[color_differences != 0].numel()
                diff = cnt / _sum
                arr2[i][j] = (diff, mask)
        return arr2


class Utils:
    TO_PIL = transforms.ToPILImage()
    TO_TENSOR = transforms.ToTensor()
    _time = None
    _logger = None
    _logging_initialized = False
    DEBUG = False
    _areas_path = None

    @classmethod
    def set_areas_file(cls, areas_path):
        cls._areas_path =areas_path

    @classmethod
    def get_areas(cls):
        new_areas = json.load(open(cls._areas_path))
        return new_areas

    @classmethod
    def get_coverage_thresholds(cls):
        arr = [(0.34, 1), (0.24, 2), (0.144, 3)]
        return arr

    @classmethod
    def remove_non_coherent_values(cls, arr, threshold=1):
        help_arr = torch.zeros_like(arr)
        sz1, sz2 = help_arr.size()
        cls.set_recursion_limit()
        for indx1 in range(0, sz1):
            for indx2 in range(0, sz2):
                if arr[indx1, indx2] != 0 and help_arr[indx1, indx2] == 0:
                    temp_arr = cls.get_connected_area_size(arr, indx1, indx2)
                    help_arr += temp_arr
                    if temp_arr.sum() <= threshold:
                        arr[temp_arr == 1] = 0

    @classmethod
    def get_connected_area_size(cls, arr, indx1, indx2, help_arr=None):
        if help_arr is None:
            help_arr = torch.zeros_like(arr)
        if arr[indx1, indx2] == 0:
            return help_arr
        else:
            help_arr[indx1, indx2] = 1
            if indx1 - 1 >= 0 and help_arr[indx1 - 1, indx2] == 0 and arr[indx1 - 1, indx2] != 0:
                cls.get_connected_area_size(arr, indx1 - 1, indx2, help_arr=help_arr)
            if indx2 - 1 >= 0 and help_arr[indx1, indx2 - 1] == 0 and arr[indx1, indx2 - 1] != 0:
                cls.get_connected_area_size(arr, indx1, indx2 - 1, help_arr=help_arr)
            if indx1 + 1 < arr.size()[0] and help_arr[indx1 + 1, indx2] == 0 and arr[indx1 + 1, indx2] != 0:
                cls.get_connected_area_size(arr, indx1 + 1, indx2, help_arr=help_arr)
            if indx2 + 1 < arr.size()[1] and help_arr[indx1, indx2 + 1] == 0 and arr[indx1, indx2 + 1] != 0:
                cls.get_connected_area_size(arr, indx1, indx2 + 1, help_arr=help_arr)
        return help_arr

    @classmethod
    def tens_scale_2d(cls, tens, scale_factor=None, size=None, mode='nearest'):
        tens = tens.unsqueeze_(0).unsqueeze_(0)
        if scale_factor:
            tens = F.interpolate(input=tens, scale_factor=scale_factor, mode=mode)
        elif size:
            tens = F.interpolate(input=tens, size=size, mode=mode)
        else:
            raise Exception('Either scale_factor or size has to be set.')
        return tens[0][0]

    @classmethod
    def set_recursion_limit(cls, new_limit=2000):
        sys.setrecursionlimit(new_limit)

    @classmethod
    def get_model(cls, num_classes):
        torch.manual_seed(0)
        return AlbuNet(pretrained=True, num_classes=num_classes)

    @classmethod
    def output_image(cls, img1_, img2_, overlay_, file_path):
        '''
        fig = plt.figure(frameon=False)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(img1_)
        ax.imshow(img2_, alpha=0.5)
        ax.imshow(overlay_, cmap='hot', alpha=0.5)
        '''
        result = Image.blend(img1_.convert("RGBA"), img2_.convert("RGBA"), alpha=0.5)
        overlay = Utils.set_color_transparent(overlay_, [0, 0, 0])

        result = Image.alpha_composite(result, overlay)

        if Utils.DEBUG:

            def onclick(event):
                global area
                area.append((int(event.xdata), int(event.ydata)))
                if (len(area)) == 4:
                    print(f'{area}')
                    area = []

            import matplotlib as mpl
            if not Utils.DEBUG:
                mpl.use('Agg')
            import matplotlib.pyplot as plt
            plt.imshow(result)
            plt.show()
        result.save(file_path)  # , bbox_inches='tight', dpi=200, pad_inches=0)
        # plt.close()

    @classmethod
    def set_color_transparent(cls, img, color):
        img = img.convert("RGBA")
        datas = img.getdata()
        from matplotlib.cm import get_cmap
        cmap = get_cmap('autumn')
        new_data = []
        for item in datas:
            if item[0] == color[0] and item[1] == color[1] and item[2] == color[2]:
                new_data.append((255, 255, 255, 0))
            else:
                rgba = cmap(item[0] / 255.0)
                item = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 120)
                new_data.append(item)

        img.putdata(new_data)
        return img

    @classmethod
    def evaluate(cls, images, labels, settings=None, evaluation_file='eval_sc2.txt', file_prefix='evaluate',
                 scale_factor=None, color_diff_threshold=None, oversizing_vert=None, oversizing_horiz=None,
                 num_classes=None, color_cnt=None, pre_scale_size=None, change_threshold=None):
        Utils.init_logging()
        logger = logging.getLogger('Evaluate')
        dif_det = DifferenceDetection(file_prefix=file_prefix, caching=True)
        if settings is None and None in [oversizing_vert, oversizing_horiz, num_classes, color_diff_threshold,
                                         color_cnt, scale_factor, change_threshold, pre_scale_size]:
            raise Exception(
                f'Either set the settings parameter or the (oversizing_vert, oversizing_horiz, num_classes, '
                f'color_diff_threshold, color_cnt, scale_factor, change_threshold, pre_scale_size) parameters.')
        scale_factor = cls._make_list(scale_factor)
        color_diff_threshold = cls._make_list(color_diff_threshold)
        oversizing_horiz = cls._make_list(oversizing_horiz)
        oversizing_vert = cls._make_list(oversizing_vert)
        num_classes = cls._make_list(num_classes)
        color_cnt = cls._make_list(color_cnt)
        pre_scale_size = cls._make_list(pre_scale_size)
        change_threshold = cls._make_list(change_threshold)
        if None not in [oversizing_vert, oversizing_horiz, num_classes, color_diff_threshold,
                        color_cnt, scale_factor]:
            if settings is None:
                settings = []
            for sf in scale_factor:
                for ct in change_threshold:
                    for cc in color_cnt:
                        for oh in oversizing_horiz:
                            for ov in oversizing_vert:
                                for nc in num_classes:
                                    for cdt in color_diff_threshold:
                                        for pss in pre_scale_size:
                                            # [ color_cnt, oversizing_horiz, oversizing_vert, num_classes, scale_factor,
                                            # color diff_threshold, pre_scale_size, change_threshold]
                                            settings.append([cc, oh, ov, nc, sf, cdt, pss, ct])

        arr = []
        if os.path.isfile(evaluation_file):
            with open(evaluation_file, 'r') as f:
                arr = json.load(f)
        pbar = tqdm(total=len(settings))
        for setting in arr:
            setting['setting'][6] = tuple(setting['setting'][6])

        for setting in settings:
            score = None
            elem = None
            results = None
            values = None
            pbar.update(1)
            if len(arr) > 0:
                pbar.set_description(
                    f'{round(min(arr, key=operator.itemgetter("score"))["score"], 2)}/'
                    f'{round(min(arr, key=operator.itemgetter("score2"))["score2"], 2)}/'
                    f'{round(score.sum().item(), 2) if score and score != -1 else "nan"}')

            c_cnt = setting[0]
            oversizing_x = setting[1]
            oversizing_y = setting[2]
            num_cla = setting[3]
            scale_fac = setting[4]
            color_diff_thres = setting[5]
            pre_scale = setting[6]
            change_thres = setting[7]

            score = torch.zeros(1)
            run = True
            for elem in arr:
                if elem['setting'] == setting:
                    run = False
                    break
            if run:

                values = []
                results = []
                for i in range(0, len(images) - 1):
                    result, vals = dif_det.calculate_differences(images[i], images[i + 1], scale_fac, change_thres,
                                                                 c_cnt,
                                                                 oversizing_horizontal=oversizing_x,
                                                                 oversizing_vertical=oversizing_y,
                                                                 num_classes=num_cla,
                                                                 color_diff_threshold=color_diff_thres,
                                                                 pre_scale_size=pre_scale, overwrite=True,
                                                                 evaluation_mode=False)
                    if result is None:
                        continue
                    if score.sum() == -1:
                        score += 1
                    values.append(vals.tolist())
                    results.append(result.tolist())
            elif elem:
                values = elem['values']
                results = elem['result']

            if values is not None and results is not None and score is not None and len(values) > 0:
                zero_val = max(np.array(values[0]).max(), np.array(values[2]).max())
                score = 0
                score2 = 0
                for index in range(len(values)):
                    temp = np.array(values[index])
                    temp2 = np.array(labels[index])

                    temp[temp > zero_val] = 1
                    temp[temp <= zero_val] = 0
                    temp2[temp2 > 0] = 1

                    temp[temp2 == -1] = 0
                    temp2[temp2 == -1] = 0
                    temp = temp - temp2

                    temp[temp == -1] = 1
                    score += temp.sum()
                    # print(labels[index])
                    # print(temp)
                    # print(temp[np.array(labels[index]) == 0.0])
                    score2 += (temp * labels[index]).sum() + temp[np.array(labels[index]) == 0].sum() * 0.2

                temp = dict()
                temp['setting'] = setting
                temp['score'] = score
                temp['score2'] = score2
                temp['result'] = results
                temp['values'] = values

                for elem in arr:  # remove all old outputs
                    if elem['setting'] == setting:
                        arr.remove(elem)
                arr.append(temp)

                if score <= 4:
                    logger.info(f'{setting}: resulted in a {score} score.')
                if score2 <= 2:
                    logger.info(f'{setting}: resulted in a {score2} score2.')
                    print(f'{setting}: resulted in a {score2} score2.')

                with open(evaluation_file, 'w+') as file:
                    file.write(json.dumps(arr))
        logger.info(f'Best result: {(min(arr, key=operator.itemgetter("score")))["setting"]}')

    @classmethod
    def _make_list(cls, var):
        if not isinstance(var, list):
            var = [var]
        return var

    @classmethod
    def brg2rgb(cls, im):
        b, g, r = im.split()
        return Image.merge("RGB", (r, g, b))

    @classmethod
    def timing(cls, name=None):
        if not cls._logger:
            Utils.init_logging()
            cls._logger = logging.getLogger('Timing')
        if cls._time and name:
            cls._logger.info(f'{name}: {round(time.time() - cls._time)}s')
            # print(f'{name}: {round(time.time() - cls._time)}s')
        cls._time = time.time()

    @classmethod
    def init_logging(cls, log_file='diff_detection.log'):
        if not cls._logging_initialized:
            cls._logging_initialized = True
            logging.basicConfig(filename=log_file, level=logging.INFO,
                                format='%(asctime)s %(levelname)-8s %(name)-8s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')


class DifferencesColorComparator:
    HSLMode = 0
    DeltaECIEMode = 1

    SumValueMode = 10
    MaxValueMode = 11

    _last_colors = None
    _color_mode = 0
    _value_mode = 11
    _weighted = False
    _color_cnt = 3
    _logger = logging.getLogger('DifferencesColorComparator')

    def __init__(self, color_mode=DeltaECIEMode, value_mode=11, weighted=False, color_cnt=3):
        Utils.init_logging()
        self._color_mode = color_mode
        self._value_mode = value_mode
        self._weighted = weighted
        self._color_cnt = color_cnt

    def start_pooled(self, parameters, callback=None, processes=cpu_count(), wait_finished=False):
        pool = Pool(processes)  # processes=1)  #
        ret = pool.map_async(self._compare_sections, parameters, callback=callback)
        pool.close()
        if not wait_finished:
            pool.join()
        return ret

    def _compare_sections(self, args):
        img1 = args[0]
        img2 = args[1]
        x = args[2]
        y = args[3]
        rx1 = args[4]
        rx2 = args[5]
        ry1 = args[6]
        ry2 = args[7]

        pt1 = Utils.TO_PIL(img1[:, rx1:rx2, ry1:ry2])
        pt2 = Utils.TO_PIL(img2[:, rx1:rx2, ry1:ry2])

        hist1, col1 = self.get_dominant_colors(pt1, k=self._color_cnt)
        hist2, col2 = self.get_dominant_colors(pt2, k=self._color_cnt)
        difference = self._comp_colors_arr(hist1, col1, hist2, col2)
        return x, y, difference

    @classmethod
    def _find_histogram(cls, clt):
        """
        create a histogram with k clusters
        :param: clt
        :return:hist
        """
        num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=num_labels)

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist

    def get_dominant_colors(self, image, k=_color_cnt):
        image = np.array(image.convert('RGB'))
        image = image[:, :, ::-1].copy()
        image = image.reshape((image.shape[0] * image.shape[1], 3))  # represent as row*column,channel number
        # clt = KMeans(n_clusters=k)  # cluster number
        clt = MiniBatchKMeans(n_clusters=k, init='k-means++')  # initial)  # cluster number
        clt.fit(image)
        self._last_colors = clt.cluster_centers_
        return self._find_histogram(clt), clt.cluster_centers_

    def comp_colors(self, col1, col2):
        color1_rgb = sRGBColor(col1[0], col1[1], col1[2])
        color2_rgb = sRGBColor(col2[0], col2[1], col2[2])
        if self._color_mode == self.DeltaECIEMode:
            color1_lab = convert_color(color1_rgb, LabColor)
            color2_lab = convert_color(color2_rgb, LabColor)
            diff = delta_e_cie2000(color1_lab, color2_lab, Kl=1)
            if diff < 2:
                return 0
        else:
            color1_hsl = convert_color(color1_rgb, HSLColor)
            color2_hsl = convert_color(color2_rgb, HSLColor)
            diff = abs(color1_hsl.hsl_h - color2_hsl.hsl_h)
            if diff > 180:
                diff = 180
        return diff

    def _comp_colors_arr(self, hist1, col_arr1, hist2, col_arr2):
        try:
            col_arr2_hsv = [convert_color(sRGBColor(col[0] if (max(col) + min(col)) * 0.5 != 1.0 else col[0] - 0.1,
                                                    col[1] if (max(col) + min(col)) * 0.5 != 1.0 else col[1] - 0.15,
                                                    col[2] if (max(col) + min(col)) * 0.5 != 1.0 else col[2] - 0.09),
                                          HSLColor) for col in col_arr2]
            col_arr1_hsv = [convert_color(sRGBColor(col[0] if (max(col) + min(col)) * 0.5 != 1.0 else col[0] - 0.1,
                                                    col[1] if (max(col) + min(col)) * 0.5 != 1.0 else col[1] - 0.15,
                                                    col[2] if (max(col) + min(col)) * 0.5 != 1.0 else col[2] - 0.09),
                                          HSLColor) for col in col_arr1]
        except Exception as ex:
            self._logger.error(ex)
            self._logger.debug(col_arr1)
            self._logger.debug(col_arr2)
            raise ex
        combined1 = np.array(
            [np.array(a) for a in sorted(zip(col_arr1_hsv, col_arr1, hist1), key=lambda x: x[0].hsl_h, reverse=True)])
        combined2 = np.array(
            [np.array(a) for a in sorted(zip(col_arr2_hsv, col_arr2, hist2), key=lambda x: x[0].hsl_h, reverse=True)])
        _sum = 0
        max_indx = len(combined2) if len(combined1) >= len(combined2) else len(combined1)
        for i in range(0, max_indx):
            diff = self.comp_colors(combined1[i][1], combined2[i][1])
            weight = ((combined1[i][2] + combined2[i][2]) / 2)
            if self._value_mode == self.SumValueMode:
                if self._weighted:
                    _sum += (diff * weight)
                else:
                    _sum += diff
            else:
                if _sum < diff:
                    _sum = diff
        return _sum

    def compare_colors_of_differences(self, diff_matrix, img1, img2, color_cnt, oversizing_x=15, oversizing_y=2,
                                      change_threshold=0.3, callback=None, weight_with_diff=False):

        params = []

        x_, y_ = diff_matrix.size()

        _, img_x_, img_y_ = img1.size()
        dif_x = img_x_ / x_
        dif_y = img_y_ / y_
        last_x2 = 0
        last_y2 = 0
        cnt = 0
        for y in range(0, y_):
            temp_y1 = int(y * dif_y)
            if temp_y1 > last_y2:
                temp_y1 = last_y2
            temp_y2 = img_y_ - 1 if int(y * dif_y + dif_y) >= img_y_ else int(y * dif_y + dif_y)
            last_y2 = temp_y2
            for x in range(0, x_):
                temp_x1 = int(x * dif_x)
                if temp_x1 > last_x2:
                    temp_x1 = last_x2
                temp_x2 = img_x_ - 1 if int(x * dif_x + dif_x) >= img_x_ else int(x * dif_x + dif_x)
                last_x2 = temp_x2
                if diff_matrix[x, y] > change_threshold:
                    rx1 = 0 if temp_x1 - oversizing_x < 0 else temp_x1 - oversizing_x
                    # rx2 = temp_x2 if temp_x2 + oversizing_x >= img_x_ else temp_x2 + oversizing_x
                    rx2 = (img_x_ - 1 if temp_x2 + oversizing_x >= img_x_ else temp_x2 + oversizing_x)  # img_x_ -
                    ry1 = 0 if temp_y1 - oversizing_y < 0 else temp_y1 - oversizing_y
                    ry2 = (img_y_ - 1 if temp_y2 + oversizing_y >= img_y_ else temp_y2 + oversizing_y)  # img_y_ -
                    # ry2 = temp_y2 if temp_y2 + oversizing_y >= img_y_ else temp_y2 + oversizing_y
                    # sub1 = to_pil(img1[:, rx1:rx2, ry1:ry2])
                    # sub2 = to_pil(img2[:, rx1:rx2, ry1:ry2])
                    # params.append([sub1, sub2, x, y, color_cnt])  # , color_cnt

                    if rx1 - rx2 != 0 and ry1 - ry2 != 0:
                        params.append([img1, img2, x, y, rx1, rx2, ry1, ry2, color_cnt])
                        cnt += 1
        col_diff_matrix = self.start_pooled(params, callback)

        output = torch.zeros_like(diff_matrix)
        results = col_diff_matrix.get()
        for x, y, difference in results:
            if weight_with_diff:
                difference = difference * diff_matrix[x, y]
            else:
                difference = difference
            # if difference >= 24:  # HSV 25
            output[x, y] = difference  # (temp_y2 if temp_y2+int(dif_y) >= img_y_ else temp_y2+int(dif_y))
        return output


class ModelThread(Thread):

    def __init__(self, model, args):
        self.args = args
        self.result = None
        self.model = model
        super(ModelThread, self).__init__()

    def run(self):
        self.result = self.model(self.args)
        print(self.result)


def _main():
    images = ['vid1.jpg', 'vid2.jpg', 'vid3.jpg', 'vid4.jpg', 'vid5.jpg', 'vid6.jpg', 'vid7.jpg', 'vid8.jpg']

    test = [[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
            [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0.5, 0.33, 0., 0.], [0., 0., 0., 0.]],
            [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
            [[0., 0., 0., 0.], [0., 0., -1, 0.], [0., 0.33, 0.5, 0.], [0.33, 0., 0., 0.], [0., 0., 1, 0.]],
            [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0.5, 0.], [0., 0., 0., 0.], [-1., 0., 0.5, 0.]],
            [[0., 0., 0., 0.], [0., 0., -1, 0.], [0., 0., 0.33, 0.], [1, 0., 0., 0.], [-1., 0., 1, 0.]],
            [[0., 0., 0.5, 0.], [0., 0., 1, 0.], [0., 0., 0.5, 0.], [0., 1, 1, 0.], [-1., 0., 0.5, 0.]]]

    # Utils.evaluate(images, test, scale_factor=0.25, color_diff_threshold=[0.1, 0.05], oversizing_vert=[5, 10, 15, 25],
    #                oversizing_horiz=[5, 10, 15, 25],
    #                num_classes=[30, 50, 100, 15], color_cnt=[7, 9], pre_scale_size=(384, 384),
    #                change_threshold=[0.38, 0.41, 0.4, 0.39, 0.42])
    Utils.evaluate(images, test, scale_factor=0.2, color_diff_threshold=[40, 50, 60, 70, 80, 90],
                   oversizing_vert=[10, 15],
                   oversizing_horiz=[15, 10],
                   num_classes=[30, 50], color_cnt=[7, 9], pre_scale_size=(384, 384),
                   change_threshold=[0.26, 0.24, 0.2, 0.23, 0.22, 0.21, 0.3, 0.29, 0.28, 0.31, 0.25, 0.32, 0.27, 0.35,
                                     0.36, 0.37, 0.38, 0.39, 0.4], evaluation_file='my_eval_20_60_col_diff.json')


if __name__ == '__main__':
    _main()

'''
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50),
                      color.astype("uint8").tolist(), -1)
        start_x = end_x

    # return the bar chart
    return bar


def get_dominant_color(image, k=4, image_processing_size=None):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # resize image if new dims provided
    image = np.array(image)
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)

5
def get_model():
    model = unet11(pretrained='carvana')
    model.eval()
    return model
'''
'''# Classes 250
# model = UNet16(num_classes=num_classes, pretrained=True)
img1 = 'vid1.jpg'
img2 = 'vid1.jpg'
img1 = Image.open(img1)
img2 = Image.open(img2)
img1 = Utils.TO_TENSOR(img1.resize((1024,1024))).unsqueeze(0)
img2 = Utils.TO_TENSOR(img2.resize((1024,1024))).unsqueeze(0)'''
'''model1 = AlbuNet(pretrained=True, num_classes=num_classes)
model2 = AlbuNet(pretrained=True, num_classes=num_classes)
res1 = model1(img1)
res2 = model2(img1)
print((res1-res2).sum())
del model1
del model2
model1 = AlbuNet(pretrained=True, num_classes=num_classes)
model2 = AlbuNet(pretrained=True, num_classes=num_classes)
res1 = model1(img1)
res2 = model2(img1)
print((res1-res2).sum())
torch.manual_seed(0)
model1 = AlbuNet(pretrained=True, num_classes=num_classes)
model2 = AlbuNet(pretrained=True, num_classes=num_classes)
res1 = model1(img1)
res2 = model2(img1)
print((res1-res2).sum())
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
model1 = AlbuNet(pretrained=True, num_classes=num_classes)

torch.manual_seed(0)
model2 = AlbuNet(pretrained=True, num_classes=num_classes)

#torch.manual_seed(0)
model3 = AlbuNet(pretrained=True, num_classes=num_classes)
model1.eval()
model2.eval()
model3.eval()
with torch.no_grad():
    res1 = model1(img1)
    res1 = model1(img1)
    res2 = model2(img2)
    res3 = model3(img1)
    print(res1.sum())
    print(res2.sum())
    print(res3.sum())
    print(res1.sum()-res2.sum())
    print(res2.sum()-res3.sum())
    print(res1.max())
    print(res2.max())
    print(res3.max())
'''

'''for scale_factor in [0.20]:  # [0.038]:  # np.arange(0.08, 0.1, 0.01): , 0.03, 0.025, 0.032, 0.027  ,0.03,0.08
    # scale_factor = round(scale_factor, 2)
    for change_threshold in [0.38,0.4,0.42]:  # np.arange(0.05, 0.45, 0.05): ,0.2,0.3
        change_threshold = round(change_threshold, 2)
        for color_cnt in [9,7]:  # range(3, 8, 2):
            for oversizing_x in [25, 15, 10, 5, 0]:  # [0,2,5]:
                for oversizing_y in [25, 15, 10, 5, 0]:  # vertical
                    for num_classes in [30, 50, 100, 150]:  # , 50, 100, 200,250]:
                        for color_diff_threshold in [0.1]:
                            dif_det.get_differences('vid7.jpg', 'vid8.jpg', scale_factor, change_threshold, color_cnt,
                                                    oversizing_x=oversizing_x, oversizing_y=oversizing_y,
                                                    num_classes=num_classes, color_diff_threshold=color_diff_threshold,
                                                    pre_scale_size=(384, 384))
                                                    '''

# pre_scale_factor=0.75
# dif_det = DifferenceDetection(file_prefix='score_test')
#
# scale_factor = 0.20
# color_diff_threshold = 0.1
#
# settings = [[7, 25, 5, 30],
#             [7, 5, 25, 150],
#             [7, 10, 0, 30],
#             [7, 10, 25, 30],
#             [7, 10, 25, 50],
#             [7, 15, 10, 30],
#             [7, 15, 25, 100],
#             [7, 0, 25, 30],
#             [7, 0, 25, 50],
#             [9, 10, 10, 150],
#             [9, 15, 10, 50],
#             [9, 15, 15, 100],
#             ]

# arr = []
# if os.path.isfile('./evaluate.txt'):
#     with open('evaluate.txt', 'r') as f:
#         arr = json.load(f)
#
# for setting in settings:
#     color_cnt = setting[0]
#     oversizing_x = setting[1]
#     oversizing_y = setting[2]
#     num_classes = setting[3]
#     for change_threshold in [0.38, 0.41, 0.4, 0.39, 0.42]:  # np.arange(0.05, 0.45, 0.05): ,0.2,0.3
#         change_threshold = round(change_threshold, 2)
#         score = torch.zeros(1)
#         score -= 1
#
#         if [elem for elem in arr if elem['setting'] == setting and elem['change_thres'] == change_threshold]:
#             continue
#
#         for i in range(0, len(images) - 2):
#             result = dif_det.get_differences(images[i], images[i + 1], scale_factor, change_threshold, color_cnt,
#                                              oversizing_x=oversizing_x, oversizing_y=oversizing_y,
#                                              num_classes=num_classes, color_diff_threshold=color_diff_threshold,
#                                              pre_scale_size=(384, 384), overwrite=True)
#             if result is None:
#                 continue
#             if score.sum() == -1:
#                 score += 1
#             score += torch.tensor(result - torch.tensor(test[i])).abs().sum()
#         if score.sum() != -1:
#             temp = dict()
#             temp['setting'] = setting
#             temp['score'] = score.item()
#             temp['change_thres'] = change_threshold
#             arr.append(temp)
#             print(temp)
#             with open('evaluate.txt', 'w') as file:
#                 file.write(json.dumps(arr))
# print(min(arr, key=operator.itemgetter('score')))
'''
# main(0.05, 0.1, 5, oversizing_x=5, oversizing_y=10)
print(f'{x}')


    model = models.vgg16(pretrained=True)
    # Set model to evaluation mode
    model.eval()
    my_embedding = torch.zeros([1,128,56,56])
    emb1 = torch.zeros([1,128,56,56])
    emb2 = torch.zeros([1,128,56,56])
    def copy_data(m, i, o):
      my_embedding.copy_(o.data)
      # print(o.data.size())
    h = layer.register_forward_hook(copy_data)
    plt.imshow(transforms.ToPILImage()(img1_tens-img2_tens))
    plt.show()
    model(img1_tens.unsqueeze(0))
    emb1.copy_(my_embedding)
    model(img2_tens.unsqueeze(0))
    emb2.copy_(my_embedding)
    h.remove()
    sum = 0
    sum_tens =(emb1[0][0]-emb2[0][0]).abs()
    for cnt in range(1, 127):
      # diff = ((emb1[0][cnt]-emb2[0][cnt]).mean())
      # sum += diff
      sum_tens += (emb1[0][cnt]-emb2[0][cnt]).abs()
    plt.figure()
    sum_tens = sum_tens / torch.max(sum_tens)
    sum_tens = torch.clamp(sum_tens, min=0.3)
    ten = transforms.ToPILImage()(sum_tens).resize((width, height), resample=Image.NEAREST, box=None)

    print(sum_tens.numpy().size)
    print(transforms.ToTensor()(ten).numpy()[0])'''
