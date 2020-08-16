##### this code aims to make every target sentence into a single file.

import os

inputs = {'valid': '/data/datasets/LCSTS/valid.tgt', 'test': '/data/datasets/LCSTS/test.tgt'}

for key in inputs:
    with open(inputs[key], 'r') as file:
        temp = file.read().splitlines()
        file.close()
    
    for idx, content in enumerate(temp):
        save_dir = './{}'.format(key)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        with open(os.path.join(save_dir, 'ref.{}.txt'.format(idx)), 'w') as filename:
            filename.write(content)
            filename.close()