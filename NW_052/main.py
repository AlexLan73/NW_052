import ParsingDan as pdan
import  NWModel as nwm

def _form_dan(dan, category, param):
    maxWordsCount, xLen, step, type_dan, type_dan0 = param[0], param[1], param[2], param[3], param[4]
    par= pdan.ParsingDan(dan, category)                     # стандартный
    xtrain, xtest= par.sprlit_train_test()
    if type_dan == 1:
        xtrain, xtest= par.normalization_test_dan(xtrain, xtest)
    form_dan= nwm.FormDanToAI0(xtrain, xtest, category, maxWordsCount)
    trein_text, test_text, tokenizer = form_dan.train_test_text()    # maxWordsCount=100
    #Формируем обучающую и тестовую выборку
    xTrain, yTrain = form_dan.createSetsMultiClasses(form_dan.trainWordIndexes, xLen, step) #извлекаем обучающую выборку
    xTest, yTest = form_dan.createSetsMultiClasses(form_dan.testWordIndexes, xLen, step)    #извлекаем тестовую выборку
    if type_dan0 == 1:
        return xTrain, yTrain, xTest, yTest, category

    #Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
    xTrain01 = tokenizer.sequences_to_matrix(xTrain.tolist())                       # подаем xTrain в виде списка чтобы метод успешно сработал
    xTest01 = tokenizer.sequences_to_matrix(xTest.tolist())                         # подаем xTest в виде списка чтобы метод успешно сработал
    return xTrain01, yTrain, xTest01, yTest, category


if __name__ == "__main__":
#    path_treatment="E:/Python/NW_05/Болезни"
    path_treatment="O:/Python/NW/NW_052/Болезни"

    p_dan = pdan.ReadDanDisk("os", d=dict(dir=path_treatment))
    dan, category = p_dan.read_dir()

    #param=  [100, 50, 10, 0, 0] 
    # maxWordsCount, xLen, step, 
    # 0/1 не сбалансированные/сбалансированные данные, 
    # 0/1 данные в форрмате "колбасы"/Embedding

    param=[110, 50, 10, 1, 0]  #[110, 50, 10, 0]
    nwm_=nwm.NWModel(_form_dan(dan, category, param))
    nwm_.model_000()

# с балансированные входные данные
#    param=[110, 50, 10, 1, 0]  
#    nwm_=nwm.NWModel(_form_dan(dan, category, param))
#    nwm_.model02_01_balans()

# Test c Embedding сбалансированные
#    param=[100, 50, 10, 1, 1] 
#    nwm_=nwm.NWModel(_form_dan(dan, category, param))
#    nwm_.model_Embedding_0(100, 50, 30)
