
import os
import csv
import argparse
from collections import OrderedDict
from tqdm import tqdm
import cv2
import re
import numpy as np
from pycocotools import mask as m

def init_config(config, default_config, name=None):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if name and config['PRINT_CONFIG']:
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config


def update_config(config):
    """
    Parse the arguments of a script and updates the config values for a given value if specified in the arguments.
    :param config: the config to update
    :return: the updated config
    """
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    return config


def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def validate_metrics_list(metrics_list):
    """Get names of metric class and ensures they are unique, further checks that the fields within each metric class
    do not have overlapping names.
    """
    metric_names = [metric.get_name() for metric in metrics_list]
    # check metric names are unique
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException('Code being run with multiple metrics of the same name')
    fields = []
    for m in metrics_list:
        fields += m.fields
    # check metric fields are unique
    if len(fields) != len(set(fields)):
        raise TrackEvalException('Code being run with multiple metrics with fields of the same name')
    return metric_names


def write_summary_results(summaries, cls, output_folder, flex_div, count_edges):
    """Write summary results to file"""

    fields = sum([list(s.keys()) for s in summaries], [])
    values = sum([list(s.values()) for s in summaries], [])

    # In order to remain consistent upon new fields being adding, for each of the following fields if they are present
    # they will be output in the summary first in the order below. Any further fields will be output in the order each
    # metric family is called, and within each family either in the order they were added to the dict (python >= 3.6) or
    # randomly (python < 3.6).
    default_order = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)',
                     'HOTALocA(0)', 'MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'CLR_TP', 'CLR_FN',
                     'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'sMOTA', 'IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP',
                     'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
    default_ordered_dict = OrderedDict(zip(default_order, [None for _ in default_order]))
    for f, v in zip(fields, values):
        default_ordered_dict[f] = v
    for df in default_order:
        if default_ordered_dict[df] is None:
            del default_ordered_dict[df]
    fields = list(default_ordered_dict.keys())
    values = list(default_ordered_dict.values())

    flex_div = '_flex_div' if flex_div else ''
    edges = '' if count_edges else '_no_edges'

    out_file = os.path.join(output_folder, (cls + '_summary' + flex_div + edges + '.txt'))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(fields)
        writer.writerow(values)


def write_detailed_results(details, cls, output_folder, flex_div, count_edges):
    """Write detailed results to file"""
    sequences = details[0].keys()
    fields = ['seq'] + sum([list(s['COMBINED_SEQ'].keys()) for s in details], [])
    flex_div = '_flex_div' if flex_div else ''
    edges = '' if count_edges else '_no_edges'

    out_file = os.path.join(output_folder, cls + '_detailed' + flex_div + edges + '.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for seq in sorted(sequences):
            if seq == 'COMBINED_SEQ':
                continue
            writer.writerow([seq] + sum([list(s[seq].values()) for s in details], []))
        writer.writerow(['COMBINED'] + sum([list(s['COMBINED_SEQ'].values()) for s in details], []))


def load_detail(file):
    """Loads detailed data for a tracker."""
    data = {}
    with open(file) as f:
        for i, row_text in enumerate(f):
            row = row_text.replace('\r', '').replace('\n', '').split(',')
            if i == 0:
                keys = row[1:]
                continue
            current_values = row[1:]
            seq = row[0]
            if seq == 'COMBINED':
                seq = 'COMBINED_SEQ'
            if (len(current_values) == len(keys)) and seq != '':
                data[seq] = {}
                for key, value in zip(keys, current_values):
                    data[seq][key] = float(value)
    return data


class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""
    ...


def convert_CTC_to_MOTS(hotapath,ctcpath):

    data_format = 'MOTS' # ['MOT','MOTS']
    hotapath.mkdir(exist_ok=True)

    ctc_folders = sorted([x for x in ctcpath.iterdir() if x.is_dir() and re.findall('\d\d$',x.name)])
                
    for ctc_folder in ctc_folders:
        
        if (ctc_folder.parent / (ctc_folder.name + '_GT')).exists():
            fps = sorted((ctc_folder.parent / (ctc_folder.name + '_GT') / 'TRA').glob('*.tif'))
            man_track_path = ctc_folder.parent / (ctc_folder.name + '_GT') / 'TRA' / 'man_track.txt'

            (hotapath / ctc_folder.name).mkdir(exist_ok=True)
            (hotapath / ctc_folder.name / 'gt').mkdir(exist_ok=True)

            if (hotapath / ctc_folder.name / 'gt' / 'gt.txt').exists():
                (hotapath / ctc_folder.name / 'gt' / 'gt.txt').unlink()

        else:
            fps = sorted(ctc_folder.glob('*.tif'))
            man_track_path = ctc_folder / 'res_track.txt'

            hotapath.mkdir(exist_ok=True)
            if (hotapath / f'{ctc_folder.name}.txt').exists():
                (hotapath / f'{ctc_folder.name}.txt').unlink()

        with open(man_track_path) as f:
            track_file = []
            for line in f:
                line = line.split() # to deal with blank 
                if line:            # lines (ie skip them)
                    line = [int(i) for i in line]
                    track_file.append(line)
            track_file = np.stack(track_file)

        all_cellnbs = set()

        # Get all cell numbers using in the CTC
        # To be compatible for HOTA, we need all cellnbs to be 1-X. It can't skip numbers
        for fp in fps:
            gt = cv2.imread(str(fp),cv2.IMREAD_ANYDEPTH)
            cellnbs = np.unique(gt)
            all_cellnbs.update(set(cellnbs))

        missing_cellnbs = [cellnb for cellnb in range(1,max(list(all_cellnbs))+1) if cellnb not in all_cellnbs]
        
        for counter,fp in enumerate(tqdm(fps)):

            gt = cv2.imread(str(fp),cv2.IMREAD_ANYDEPTH)

            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]

            framenb = int(re.findall('\d+',fp.name)[-1])
        
            # If no cell is present in image, we skip it
            if len(cellnbs) == 0:
                continue

            for cellnb in cellnbs:

                buffer = sum(1 for missing_cellnb in missing_cellnbs if missing_cellnb < cellnb)

                mask = gt == cellnb
                parent = track_file[track_file[:,0]==cellnb,-1][0] if track_file[track_file[:,0]==cellnb,1] == framenb and track_file[track_file[:,0]==cellnb,-1] > 0 else 0
                parent_buffer = sum(1 for missing_cellnb in missing_cellnbs if missing_cellnb < parent)

                if data_format == 'MOTS':
                    rle = m.encode(np.asfortranarray(mask))['counts'].decode("utf-8")
                    line = f'{counter+1} {cellnb-buffer} {1} {gt.shape[0]} {gt.shape[1]} {rle} {parent-parent_buffer}'

                elif data_format == 'MOT':
                    y, x = np.where(mask != 0)
                    width = (np.max(x) - np.min(x) ) 
                    height = (np.max(y) - np.min(y)) 
                    bbox = (np.min(x) ,np.min(y),width,height)
                    line = f'{counter+1} {cellnb-buffer} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} -1 -1 -1 -1 {parent-parent_buffer}'
                else:
                    raise NotImplementedError

                if (ctc_folder.parent / (ctc_folder.name + '_GT')).exists():
                    with open(hotapath / ctc_folder.name / 'gt' / 'gt.txt', 'a') as file:
                        file.write(line + '\n')
                else:
                    with open(hotapath / f'{ctc_folder.name}.txt', 'a') as file:
                        file.write(line + '\n')

