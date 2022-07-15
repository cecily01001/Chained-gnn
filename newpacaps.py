import os
import re
from pathlib import Path
path = '/home/user1/PangBo/CGNN/Dataset/Origin_Data/app/'
files = os.listdir(path)
for i, file in enumerate(files):
    if Path(file).name.split('.')[0]=='02':
        NewFileName = os.path.join(path, Path(file).name.split('.')[4]+'.pcap')
        OldFileName = os.path.join(path, file)
        if re.search(path, NewFileName) is None:
            file_name = file_name.replace('.', '(0).')
            os.rename(OldFileName, NewFileName)
