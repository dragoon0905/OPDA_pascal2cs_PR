# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# UDA with Thing-Class ImageNet Feature Distance + Increased Alpha
_base_ = ['dacs.py']
uda = dict(
    alpha=0.999,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,34,35,36,37,38,40,41,43,44,45,46,48,49,50,52,54,55],
    imnet_feature_dist_scale_min_ratio=0.75,
)
