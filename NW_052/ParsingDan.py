import os
import copy
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras import utils
import numpy as np

import regex as re
import pymorphy2
import pprint

## class ReadDanDisk:
class ReadDanDisk:
    def __init__(self, *args, **kwargs):
        self.type_disk = "os"
        self.path_file = ""
        self.path_dir = ""
        self.morph = pymorphy2.MorphAnalyzer()

        ReadDanDisk.set_param(self, *args, **kwargs)

    def set_param(self, *args, **kwargs):
        if args.__len__() >0:
            for item in args:
                if item == "os":
                    self.type_disk="os"

                if item == "google":
                    self.type_disk="google"

        for key0, value in kwargs.items():
            if key0 == "d":
                for key1, value1 in value.items():
                    if key1 =="dir":
                        self.path_dir=value1
                    if key1 =="file":
                        self.path_file=value1

    def read_dir(self):
        ls_dir=[]
        if self.path_dir == "":
            retirn -1
        name_file = os.listdir(self.path_dir)

        self.caterodiy = {x: name_file[x].split('.')[0].lower() for x in range(len(name_file))}
        file_dan_d= {x: "" for x in range(len(name_file))}
        self.file_dan={}

        ls_dir =[self.path_dir+"/"+x for x in name_file]

        for item in range(len(ls_dir)):
            file_dan_d[item]= ReadDanDisk.__read_dan_(self, ls_dir[item])

        for key, val in file_dan_d.items():
            self.file_dan[key]=ReadDanDisk.__parsing_one_categor(self, val)

        return self.file_dan, self.caterodiy

    def __read_dan_(self, path_item):
        with open(path_item, encoding='utf-8') as f:
            myList = [ line.split("\n")[0].lower() for line in f ]
        return myList
      
    def __filter_char_(self, s_):
        ss=copy.deepcopy(s_)

        shablon1=r'[,.;—_()--:!{}+«»=@\t\n]' # 

        s0=str(re.sub(shablon1,'',ss).lower().strip(" "))
        s0=re.sub(r'[//\\]',' ', s0).split(' ')

        s = ' '.join ([self.morph.parse(x)[0].normal_form for x in s0])
        
        s = re.sub(" +", " ", s)
        return s

    def __parsing_one_categor(self, ls_):
        ls=[]
        s=""
        for item in ls_:
            if len(item) >0:
                s = item if len(s)==0 else s+" "+item
            else:
                if len(s)>0:
                    ls.append(ReadDanDisk.__filter_char_(self, s))
                    s=""
        if len(s)>0:
            ls.append(ReadDanDisk.__filter_char_(self, s))
            s=""
        return dict(dan=ls, count = len(ls))

    def read_file(self):
        pass

class ParsingDan:
    def __init__(self, *args, **kwargs):
        self.split_test = 0.2
        self.type_par=0
        self.sum_word = set()
 
        self.xtrain_all={}
        self.train=[]
        self.test=[]
        self.category={}

        self.All_maska=set()
        self.db_maska={}

        ParsingDan.set_param(self, *args, **kwargs)
    
    def set_param(self, *args, **kwargs):
        for key0, value in kwargs.items():
            if key0 == "d":
                for key1, value1 in value.items():
                    if key1 == "split_test":
                        self.split_test=value1
                    if key1 == "type_par":
                        self.type_par = value1

        if args.__len__() >0:
            self.file_dan=args[0]
        if args.__len__() >1:
            self.category = args[1]
    
    def __split_dan(self, db):
        xtrain, xtest = {}, {}
        for key, val in db.items():
            ls=val['dan']
            count=val['count']
            if self.split_test == 0.0:
                xtrain[key] =ls
                xtest =[]
            else:
                count=count- int(count*self.split_test)
                xtrain[key] =ls[:count]
                xtest[key] =ls[count:]
        return xtrain, xtest

    def sprlit_train_test(self):
        if self.type_par==0:
            return ParsingDan.__split_dan(self, self.file_dan)
        elif self.type_par==1:
            ParsingDan.__keywords_by_category(self, self.file_dan)
            db=ParsingDan.__and_dan_mmask(self, self.file_dan)
            return ParsingDan.__split_dan(self, db)
        else:
            pass

    def __keywords_by_category(self, db):
        new_db0={}
        for key, val in db.items():
            ls=val['dan']
            x_ob_set=set()
            for it in ls:
                x_ob_set.update(str(it).rstrip().split(" "))
            new_db0[key]=x_ob_set

        ls_key=list(new_db0.keys())    
        self.db_maska=copy.deepcopy(new_db0)

        key_word=copy.deepcopy(new_db0)

        self.All_maska=set()
        for key, val in new_db0.items():
            x=val
            for key1, val1 in key_word.items():
                if key == key1:
                    continue
                x.difference_update(val1)
            self.db_maska[key]=x
            self.All_maska.update(x)

    def __and_dan_mmask(self, db):
        new_db=copy.deepcopy(db)
        for key, val in db.items():
            ls=val['dan']
            maska= self.db_maska[key]
            new_ls=[]
            for it in ls:
                it0=copy.deepcopy(it)
                it_set =set(it.split(" "))
                it_set.intersection_update(maska)
                if len(it_set)<=0:
                    new_ls.append(it0)
                else:                    
                    new_ls.append(' '.join ([str(x) for x in it_set]))

            new_db[key]['dan']=new_ls
        return new_db

    def __convert_01(self, db):
        for key, val in db.items():
            ls=val
            new_ls=[]
            for it in ls:
                new_set=set(str(it).split(" "))
                new_set.intersection_update(self.sum_word)
                ls0=list(new_set)    
                new_ls.append(' '.join ([x for x in ls0]))

            db[key]=new_ls
        return db

    def convert_to_standart_word(self):
         self.xtrain_filtr = copy.deepcopy(ParsingDan.__convert_01(self, self.xtrain_basa))
         self.xtest_filtr = copy.deepcopy(ParsingDan.__convert_01(self, self.xtest_basa))
         return self.xtrain_filtr, self.xtest_filtr
    
    def __sum_word(self, db):
        ls_db = []
        lsd={}
        for key, val in db.items():
            ncount=0
            lsd[key]=[]
            for item in val:
                count= len(str(item).split(' '))
                lsd[key].append(count)
                ncount +=count

            ls_db.append(ncount)
        sum_count_db=sum(ls_db)
        return ls_db, sum_count_db, lsd

    def __calc0(self, ls_train, ls_test):
            rez = float(ls_test)/float(ls_train + ls_test)*100
            dop= float(ls_train)/4-float(ls_test)
            return rez, dop

    def __static_train_test(self, ls_train, sum_count_train, ls_test, sum_count_test):
        rez = float(sum_count_test)/float(sum_count_train + sum_count_test)*100
        print("отношение сумма слов ={} ",format(rez)) 

        for it in range(len(ls_test)):
#            rez = float(ls_test[it])/float(ls_train[it] + ls_test[it])*100
#            dop= float(ls_train[it])/4-float(ls_test[it])
            rez, dop = ParsingDan.__calc0(self, ls_train[it], ls_test[it])
            print("отношение сумма слов  катигории = {}  = {} <- ".format( rez, dop)) 

    def normalization_test_dan(self, xtrain, xtest):
        ls_train, sum_count_train, dx = ParsingDan.__sum_word(self, xtrain)
        ls_test, sum_count_test, dx = ParsingDan.__sum_word(self, xtest)
        ParsingDan.__static_train_test(self, ls_train, sum_count_train, ls_test, sum_count_test)
        for key, val in xtest.items():
            rez, dop = ParsingDan.__calc0(self, ls_train[key], ls_test[key])
            if dop > 0:
                s0 = str( ' '.join ([str(x) for x in val])).split(" ")
                if len(s0)<int(dop):
                    s11=[]
                    s00=[s11.append(x) for x in s0]
                    s01=[s11.append(x) for x in s0]
                    s01=[s11.append(x) for x in s0]
                    s01=[s11.append(x) for x in s0]
                    s0=s11
                    
                s1=s0[:int(dop)+1]
                s2 = ' '.join ([str(x) for x in s1])
                xtest[key].append(s2)
        ls_test, sum_count_test, dx = ParsingDan.__sum_word(self, xtest)
        ParsingDan.__static_train_test(self, ls_train, sum_count_train, ls_test, sum_count_test)
        return xtrain, xtest
