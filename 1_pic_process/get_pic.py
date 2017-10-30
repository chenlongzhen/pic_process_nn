#encoding=utf-8

import urllib.request,sys 
S_PATH = sys.path[0]
DATA_PATH=S_PATH+"/../../data"

input_name=sys.argv[1]
output_name=sys.argv[2]
pic_path = sys.argv[3]

series_col=1
pic_col=3

with open(DATA_PATH+"/base_data/"+input_name,'r') as rfile,open(DATA_PATH+"/base_data/"+output_name,'w') as wfile:
    count=0
    for line in rfile:
        segs = line.split('\t')
        serie = segs[series_col]
        pic_url = segs[pic_col]
        if pic_url.strip().split(".")[-1] == 'jpg':
            tail = ".jpg"
        elif pic_url.strip().split(".")[-1] == 'png':
            tail = ".png"
        else:
            print( pic_url.split(".")[-1] )
            continue 
        try:
            print("{}\tdownloading...".format(pic_url))
            urllib.request.urlretrieve(pic_url,DATA_PATH + "/" +pic_path + "/"+str(count)+tail)
            count+=1
        except urllib.error.HTTPError:
            print('【错误】当前图片无法下载')
            continue
        wfile.write("{} {}\n".format(DATA_PATH + "/" +pic_path + "/"+str(count)+tail,serie)) 

#python get_pic.py base_pic_data train.txt images 
