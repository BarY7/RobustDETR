import torch

from typing import Dict


def get_one_output_from_batch(in_dict, i):
    new_in_dict: Dict = {}
    for key in in_dict:
        if torch.is_tensor(in_dict[key]):
            new_in_dict[key] = in_dict[key][i].unsqueeze(0)
        else:
            new_in_dict[key] = in_dict[key]
    return new_in_dict
