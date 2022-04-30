import os
import numpy as np
import pandas as pd
from PIL import Image

class FileHandling():
    
    def create_file_list(self, my_dir, format='.jpg'):
        file_list = []
        for root, dirs, files in os.walk(my_dir, topdown=False):
            for name in files:
                if name.endswith(format): 
                    full_name = os.path.join(root, name)
                    file_list.append(full_name)
        file_list = pd.DataFrame(file_list, columns=['Images paths'])
        return file_list
    
    def load_images(self, paths):
        v = []
        for path in paths['Images paths']:
            v.append(np.reshape(np.asarray(Image.open(path)), -1))
        x_raw = pd.DataFrame(v)
        return x_raw
        
        