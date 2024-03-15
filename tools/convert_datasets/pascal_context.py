# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp


import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid ={
        #0:255,
        1:13,
        2:14,
        3:15,
        4:16,
        5:17,
        6:11,
        7:18,
        8:19,
        9:20,
        10:21,
        11:2,
        12:	9,
        13:22,
        14:	7,
        15:	23,
        16:24,
        17:25,
        18:26,
        19:27,
        20:28,
        21:29,
        22:30,
        23:31,
        24:32,
        25:4,
        26:33,
        27:34,
        28:35,
        #29:255,
        #30:255,
        31:36,
        32:37,
        33:38,
        34:10,
        35:39,
        36:40,
        #37:255,
        38:41,
        39:42,
        40:43,
        41:0,
        42:44,
        43:45,
        44:46,
        45:1,
        46:47,
        47:6,
        48:48,
        49:49,
        50:50,
        51:51,
        #52:255,
        53:5,
        54:8,
        55:52,
        56:3,
        57:53,
        58:54,
        59:55,
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmsegmentation format')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--gt-dir', default='', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    gta_path = '/data/dragoon0905/datasets/UniDA_pascal2cs_12/VOC_2010'
    out_dir = args.out_dir if args.out_dir else gta_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = '/data/dragoon0905/datasets/UniDA_pascal2cs_12/VOC_2010/SegmentationClassContext'

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
    
    
    