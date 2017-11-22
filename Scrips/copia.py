import shutil
import os
import glob
path = "/home/jorgher/Documents/faceRec/CFEE/"
dire = glob.glob(path+'ren*')
AU = range(1,23)
lista = []
for au in AU:
    if au < 10:
        os.mkdir(os.path.join(path+'Emo_0'+ str(au)))
        dst = os.path.join(path+'Emo_0'+str(au))
        
        for dir in dire:
            files = os.listdir(dir)
            for f in files:                    
                if f.endswith('0'+str(au), 0, 2):
                    src = dir + '/' + f 
                    dst_ = dst + '/' + f 
                    shutil.copy(src, dst_)
                    
    else:
        os.mkdir(os.path.join(path+'Emo_'+ str(au)))
        dst = os.path.join(path+'Emo_'+str(au))
        for dir in dire:
            files = os.listdir(dir)
            for f in files:
                if f.endswith(str(au), 0, 2):
                    src = dir + '/' + f 
                    dst_ = dst + '/' + f 
                    shutil.copy(src, dst_)

