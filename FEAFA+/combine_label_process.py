import os
import numpy as np
import pandas as pd

list_prefixes = ['../../FEAFA+/FEAFA-A/list/FEAFA', '../../FEAFA+/FEAFA-B/list/DISFA']
save_path = '../../FEAFA+/list/'

for fold in [1, 2, 3]:
    for path in list_prefixes:
        for phase in ['test', 'train']:
            for type in ['img_path', 'label']:
                new_file_path = f'{save_path}{phase}_{type}_fold{fold}.txt'
                with open(new_file_path, 'w') as f:
                    for pre in list_prefixes:
                        with open(f'{pre}_{phase}_{type}_fold{fold}.txt', 'r') as temp:
                            f.write(temp.read())