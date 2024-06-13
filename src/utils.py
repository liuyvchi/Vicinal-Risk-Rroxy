import torch
import cv2
import numpy as np
import random
import re


def match_keys(d, ref):
    """
    Matches the format of keys between d (a dict) and ref (a list of keys).
    Helper function for situations where two algorithms share the same model, and we'd like to warm-start one
    algorithm with the model of another. Some algorithms (e.g. FixMatch) save the featurizer, classifier within a sequential,
    and thus the featurizer keys may look like 'model.module.0._' 'model.0._' or 'model.module.model.0._',
    and the classifier keys may look like 'model.module.1._' 'model.1._' or 'model.module.model.1._'
    while simple algorithms (e.g. ERM) use no sequential 'model._'
    """
    # hard-coded exceptions
    d = {re.sub('model.1.', 'model.classifier.', k): v for k,v in d.items()}
    d = {k: v for k,v in d.items() if 'pre_classifier' not in k} # this causes errors

    # probe the proper transformation from d.keys() -> reference
    # do this by splitting d's first key on '.' until we get a string that is a strict substring of something in ref
    success = False
    probe = list(d.keys())[0].split('.')
    for i in range(len(probe)):
        probe_str = '.'.join(probe[i:])
        matches = list(filter(lambda ref_k: len(ref_k) >= len(probe_str) and probe_str == ref_k[-len(probe_str):], ref))
        matches = list(filter(lambda ref_k: not 'layer' in ref_k, matches)) # handle resnet probe being too simple, e.g. 'weight'
        if len(matches) == 0: continue
        else:
            success = True
            append = [m[:-len(probe_str)] for m in matches]
            remove = '.'.join(probe[:i]) + '.'
            break
    if not success: raise Exception("These dictionaries have irreconcilable keys")

    return_d = {}
    for a in append:
        for k,v in d.items(): return_d[re.sub(remove, a, k)] = v

    # hard-coded exceptions
    if 'model.classifier.weight' in return_d:
       return_d['fc.weight'], return_d['fc.bias'] = return_d['model.classifier.weight'], return_d['model.classifier.bias']
    return return_d

def Get_DoC_source(path):
    DoC_source = np.load(path, allow_pickle=True).item()
    acc = DoC_source['test_acc']
    AC = DoC_source['AC']
    return AC, acc

def Get_ATC_source(path):
    ATC_source = np.load(path, allow_pickle=True).item()
    acc = ATC_source['test_acc']
    threshold = ATC_source['Threshold']
    return threshold

def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def generate_flip_grid(w, h, device):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid