import pandas as pd
import glob
import csv
import os

path = "/home/jorgher/Documents/faceRec/CFEE/"
files = sorted(glob.glob(path+'*.csv'))
#files = sorted(glob.glob('*.csv'))
member = []
fd = open('membeshipFunct.csv','wt')
writer = csv.writer(fd)
writer.writerow(("Emotion","Angl","XSmall","XMediumMinus","XMedium","XMediumPlus","XLarge"))
conta = 1

for f in files:
    file = pd.read_csv(f)
    db = pd.DataFrame(file)
    Esta = pd.DataFrame.describe(db)
    Esta_ = Esta[['A0','A1','A2','A3','A4','A5']]
    Angl = pd.DataFrame(data = Esta_[::],index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    ang = 0
    Emo = os.path.join('Emo_'+str(conta))
    conta += 1
    for i in range(0,6):
        XS = min(Angl.loc['min'][i])
        XMM = Angl.loc['25%'][i]
        XM = Angl.loc['mean'][i]
        XMP = Angl.loc['75%'][i]
        XL = max(Angl.loc['max'][i])
        Angul = os.path.join('A'+str(ang))
        ang += 1
        member.append([Emo,Angul,XS,XMM,XM,XMP,XL])
        writer.writerow((Emo,Angul,XS,XMM,XM,XMP,XL))

