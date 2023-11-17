# Copyright (c) OpenMMLab. All rights reserved.
import copy
import copy as cp
import os.path as osp
from collections import defaultdict

import numpy as np
from mmaction.datasets.transforms import (DecompressPose, GeneratePoseTarget,
                                          GenSkeFeat, JointToBone,
                                          MergeSkeFeat, MMCompact, MMDecode,
                                          MMUniformSampleFrames, PadTo,
                                          PoseCompact, PoseDecode,
                                          PreNormalize2D, PreNormalize3D,
                                          ToMotion, UniformSampleFrames)

from mmaction.datasets.transforms import (CenterCrop, ColorJitter, Flip, Fuse,
                                          MultiScaleCrop, RandomCrop,
                                          RandomResizedCrop, Resize, TenCrop,
                                          ThreeCrop)

from mmaction.datasets.transforms import (FormatAudioShape, FormatGCNInput,
                                          FormatShape, PackActionInputs,
                                          Transpose)

from io import BytesIO
import pickle

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        value = f.read()
    
    f = BytesIO(value)
    data = pickle.load(f)
    return data

def load_data_list(filepath): 
    split_mode = "xsub_val"
    data_list = load_pkl(filepath) 
    split, annos = data_list['split'], data_list['annotations']
    identifier = 'filename' if 'filename' in annos[0] else 'frame_dir'
    split = set(split[split_mode])
    data_list = [x for x in annos if x[identifier] in split]
    return data_list

def main(): 
    ann_file = "/data1.local/vinhpt/trongminh/mmaction2/data/skeleton/annotations_first_split.pkl"
    data_list = load_data_list(ann_file) 
    left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
    right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
    # print(data_list)
    data_copy = copy.deepcopy(data_list[0])
    transform_components = []
    transform_components.append(UniformSampleFrames(clip_len=48, num_clips=10, test_mode=True)) 
    transform_components.append(PoseDecode()) 
    transform_components.append(PoseCompact(hw_ratio=1, allow_imgpad=True)) 
    transform_components.append(Resize(scale=(-1, 64)))
    transform_components.append(CenterCrop(crop_size=64)) 
    transform_components.append(GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, 
                                                   with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp))
    transform_components.append(FormatShape(input_format='NCTHW_Heatmap')) 
    transform_components.append(PackActionInputs())
    transform_components
    data_processed = []
    data_processed.append(copy.deepcopy(data_copy))
    with open("data_processed.txt", "w") as file: 
        for component in transform_components: 
            data_copy = component.transform(data_copy)
            if component.__class__.__name__ != "PackActionInputs":
                b = copy.deepcopy(data_copy)
                a = data_copy['keypoint']
                del b['keypoint']
                del b['keypoint_score']
                file.write(component.__class__.__name__ + "\n")
                # file.write(str(b) + "\n")
                file.write("keypoint\n")
                for i in a:
                    for j in i:
                        file.write(str(j) + "\n")
                a = data_copy['keypoint_score']
                file.write("keypoint_score\n")
                for i in a:
                    for j in i:
                        file.write(str(j) + "\n")
                if component.__class__.__name__ == "GeneratePoseTarget" or component.__class__.__name__ == "FormatShape": 
                    a = data_copy['imgs'] 
                    del b['imgs']
                    file.write("imgs\n")  
                    x = a.flatten()
                    file.write(str(sum(x)) + "\n")
                file.write(str(b) + "\n")
            else: 
                file.write(component.__class__.__name__ + "\n")
                a = data_copy['inputs']
                file.write("inputs\n") 
                import torch
                t = torch.flatten(a) 
                file.write(str(torch.sum(t)) + "\n")

            data_processed.append(copy.deepcopy(data_copy))

    # sampling = UniformSampleFrames(clip_len=48, num_clips=10, test_mode=True) 
    # result = sampling.transform(data_copy)
    # print(result)
    import IPython; IPython.embed()
    


if __name__ == '__main__':
    main()
