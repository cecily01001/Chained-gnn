import csv
import os
import shutil
cwd="C:/Users/cecil/Desktop/AppData_download/AppData_download"
filelist = os.listdir(cwd)
accuracyfile = open('accuracyfile.csv', 'w', encoding='utf-8', newline="")
accuracyfile_writer = csv.writer(accuracyfile)
for file in filelist:
    names = file.split('.')
    pcap = os.listdir("C:/Users/cecil/Desktop/AppData_download/AppData_download/"+file)
    accuracyfile_writer.writerow([names,len(pcap)])
    # input()
    # if names[-1] == 'pcap':
    #     filepath = os.path.join(cwd, file)
    #     category = names[4]
    #     desdir = os.path.join(cwd, category)
    #     if not os.path.isdir(desdir) :
    #         os.mkdir(desdir)
    #     shutil.move(filepath, desdir)
