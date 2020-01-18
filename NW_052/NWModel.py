import tensorflow as tf

#from google.colab import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import utils

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from keras.layers import Input, Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
#from keras.layers import Input, Dense, Dropout, SpatialDropout1D, BatchNormalization, Flatten, Activation
#from tensorflow.keras.layers Input, Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.optimizers import Adam,  RMSprop


from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import concatenate

import copy

class FormDanToAI0:
    def __init__(self, xtrain_, xtest_, catigory_, maxwordscount=150):
        self.xtrain_ = xtrain_
        self.xtest_ = xtest_
        self.category=catigory_
        self.maxWordsCount=maxwordscount   #определяем макс.кол-во слов/индексов, учитываемое при обучении текстов
    def train_test_text(self):
        self.train_text=[]
        self.test_text=[]

        for key, val in self.xtrain_.items(): self.train_text.append(' '.join ([x for x in val]))
        for key, val in self.xtest_.items(): self.test_text.append(' '.join ([x for x in val]))

        #для этого воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
        self.tokenizer = Tokenizer(num_words=self.maxWordsCount  \
                              , filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0' \
                              , lower=True  \
                              , split=' '   \
                              , char_level=False)
                                #(num_words=maxWordsCount) - определяем макс.кол-во слов/индексов, учитываемое при обучении текстов
                                #(char_level=False) - просим токенайзер не удалять однобуквенные слова

        self.tokenizer.fit_on_texts(self.train_text)                 # "скармливаем" наши тексты, т.е даём в обработку методу, который соберет словарь частотности
#        items = list(self.tokenizer.word_index.items())        # Вытаскиваем индексы слов для просмотра
                                                                # преобразовываем текст в последовательность индексов согласно частотному словарю
        self.trainWordIndexes = self.tokenizer.texts_to_sequences(self.train_text)          #обучающие тесты в индексы
        self.testWordIndexes  = self.tokenizer.texts_to_sequences(self.test_text)           #проверочные тесты в индексы
        return self.train_text, self.test_text, self.tokenizer
    def print_statist(self):

        print("Статистика по обучающим текстам:")

        len_train =sum([len(x) for x in self.train_text])
        len_trainWordIndexes =sum ([len(x) for x in self.trainWordIndexes])

        len_test =sum([len(x) for x in self.test_text])
        len_testWordIndexes =sum ([len(x) for x in self.testWordIndexes])

        for i in range(len(self.category)):
            print(self.category[i], " ", len(self.train_text[i]), " символов, ", len(self.trainWordIndexes[i]), " слов")
        print("В сумме ", len_train, " символов, ", len_trainWordIndexes, " слов", "\n")
        
        print("Статистика по обучающим текстам:")
        for i in range(len(self.category)):
            print(self.category[i], " ", len(self.test_text[i]), " символов, ", len(self.testWordIndexes[i]), " слов")

        print("В сумме ", len_test, " символов, ", len_testWordIndexes, " слов")
    ###########################
    # Формирование обучающей выборки по листу индексов слов
    # (разделение на короткие векторы)
    ##########################
    def __getSetFromIndexes(self, wordIndexes, xLen, step):
      xSample, index = [], 0
      wordsLen = len(wordIndexes)
  
      #Идём по всей длине вектора индексов
      #"Откусываем" векторы длины xLen и смещаеммся вперёд на step
  
      while (index + xLen <= wordsLen):
        xSample.append(wordIndexes[index:index+xLen])
        index += step
      return xSample    
    def createSetsMultiClasses(self, wordIndexes, xLen, step): #функция принимает последовательность индексов, размер окна, шаг окна
      #Для каждого из N классов
      #Создаём обучающую/проверочную выборку из индексов
      nClasses = len(wordIndexes)       #задаем количество классов выборки
      classesXSamples = []              #здесь будет список размером "кол-во классов*кол-во окон в тексте*длину окна(например N по x0*xlen)"
      for wI in wordIndexes:            #для каждого текста выборки из последовательности индексов
        classesXSamples.append(FormDanToAI0.__getSetFromIndexes(self, wI, xLen, step)) #добавляем в список очередной текст индексов, разбитый на "кол-во окон*длину окна" 

      #Формируем один общий xSamples
      xSamples = []                     #здесь будет список размером "суммарное кол-во окон во всех текстах*длину окна(например 15779*1000)"
      ySamples = []                     #здесь будет список размером "суммарное кол-во окон во всех текстах*вектор длиной 6"
  
      for t in range(nClasses):         #в диапазоне кол-ва классов(6)
        xT = classesXSamples[t]         #берем очередной текст вида "кол-во окон в тексте*длину окна"(например 1341*1000)
        for i in range(len(xT)):        #и каждое его окно
          xSamples.append(xT[i])        #добавляем в общий список выборки
    
        #Формируем ySamples по номеру класса
        currY = utils.to_categorical(t, nClasses)       #текущий класс переводится в вектор длиной 6 вида [0.0.0.1.0.0.]
        for i in range(len(xT)):        #на каждое окно выборки 
          ySamples.append(currY)        #добавляем соответствующий вектор класса

      xSamples = np.array(xSamples) #переводим в массив numpy для подачи в нейронку
      ySamples = np.array(ySamples) #переводим в массив numpy для подачи в нейронку
  
      return (xSamples, ySamples) #функция возвращает выборку и соответствующие векторы классов

class NWModel(object):
    def __init__(self, *args, **kwargs):
        if args.__len__() ==1:
            self.xTrain = args[0][0]
            self.yTrain = args[0][1] 
            self.xTest = args[0][2] 
            self.yTest = args[0][3] 
            self.category = args[0][4] 
        else:
            self.xTrain = args[0]
            self.yTrain = args[1] 
            self.xTest = args[2] 
            self.yTest = args[3] 
            self.category = args[4] 

    def __dens_drop(self, layers_, n0, s0, d0):
        layers_= Dense(n0,  activation=s0)(layers_)
        return Dropout(d0)(layers_) if d0>0 else layers_

    def __dens2(self, layers_, n0, s0, n1, s1):
        layers_= Dense(n0,  activation=s0)(layers_)
        return Dense(n1,  activation=s1)(layers_)

    def __dens2_drop(self, layers_, n0, s0, n1, s1, d0):
        return Dropout(d0)(NWModel.__dens2(self,  layers_, n0, s0, n1, s1))

    def __dens3(self, layers_, n0, s0, n1, s1, n2, s2):
        layers_= Dense(n0,  activation=s0)(layers_)
        layers_ = Dense(n1,  activation=s1)(layers_)
        return Dense(n2,  activation=s2)(layers_)

    def __dens3_drop(self, layers_, n0, s0, n1, s1, n2, s2, d0):
        return Dropout(d0)(NWModel.__dens3(self,  layers_, n0, s0, n1, s1, n2, s2))
    
    def model_000(self): #Создаём полносвязную сеть
        inputs_basa = Input(shape=(self.xTrain.shape[1],), name="inputs_basa" ) #self.xTrain.shape[1]
        x = BatchNormalization() (inputs_basa)
        x = NWModel.__dens_drop(self, x, 3000, "relu", 0.2)
        x = NWModel.__dens_drop(self, x, 3000, "relu", 0.1)
        x = NWModel.__dens3(self, x, 2000,  "sigmoid", 1000,  "relu", 500,  "sigmoid")
        x = Dense(100, activation="relu") (x)
        out_0 = Dense(len(self.category), activation='softmax')(x)
        model = Model(inputs=inputs_basa, outputs=out_0)
        NWModel.__fit(self, model, 40, 16) #70, 23
        i=1

    def model02_01_balans(self): #Создаём полносвязную сеть
        inputs_basa = Input(shape=(self.xTrain.shape[1],), name="inputs_basa" ) #self.xTrain.shape[1]
        x = BatchNormalization() (inputs_basa)
        x = NWModel.__dens_drop(self, x, 5000, "relu", 0.2)
        x = NWModel.__dens_drop(self, x, 3000, "relu", 0.2)
        x = NWModel.__dens2_drop(self, x, 2000,  "sigmoid", 1000,  "relu", 0.2)
        x = NWModel.__dens2_drop(self, x, 1550,  "relu", 1250,  "sigmoid", 0.2)
        x = NWModel.__dens2(self, x, 500,  "sigmoid", 200,  "relu")

        out_0 = Dense(len(self.category), activation='softmax')(x)
        model = Model(inputs=inputs_basa, outputs=out_0)
        NWModel.__fit(self, model, 90, 19)
        i=1

    def model_Embedding_0(self, maxWordsCount, xLen, matr):
        inputs_Embedding = Input(shape=(xLen,), name="inputs_Embedding" )
        xE = Embedding(maxWordsCount, matr, input_length=xLen) (inputs_Embedding)
        xE = SpatialDropout1D(0.2) (xE)
        xE = Flatten()(xE)
        xE = BatchNormalization() (xE)

#        xE = NWModel.__dens3_drop(self, xE, 5000, "relu", 5000, "relu", 3000, "relu", 0.2)
#        xE = NWModel.__dens3(self, xE, 2000, "relu", 1000, "relu", 1000, "relu")
#        xE = NWModel.__dens2_drop(self, xE, 500,  "relu", 100,  "sigmoid", 0.1)

#        xE = NWModel.__dens3_drop(self, xE, 2000, "relu", 1000, "relu", 3000, "relu", 0.1)
        xE = NWModel.__dens3_drop(self, xE, 2000, "relu", 5000, "relu", 1000, "relu", 0.1)
#        xE = NWModel.__dens2(self, xE, 1000, "relu", 500, "relu")
        xE = NWModel.__dens2(self, xE, 100,  "relu", 50,  "relu")

        out_E=Dense(len(self.category), activation='softmax')(xE)
        modelE = Model(inputs=inputs_Embedding, outputs=out_E)
        NWModel.__fit(self, modelE, 70, 21)

    def __plot(self, histor, acc, val_acc):
        plt.plot(histor.history[acc], 
                 label='Доля верных ответов на обучающем наборе')
        plt.plot(histor.history[val_acc], 
                 label='Доля верных ответов на проверочном наборе')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Доля верных ответов')
        plt.legend()
        plt.show()

    def __fit(self, model, epochs, batch_size):
        model.compile(optimizer=RMSprop(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(self.xTrain, 
                    self.yTrain, 
                    epochs=epochs, 
                    batch_size=batch_size,  #19
                    shuffle=True,
                    verbose=1,
                    validation_data=(self.xTest, self.yTest))

        NWModel.__plot(self, history, 'accuracy', 'val_accuracy')
        return history


