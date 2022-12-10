import cv2
import glob
import os
import numpy as np
import random
import basic_functions as bf
import function_validation as fv
import matplotlib.pyplot as plt
#from sklearn import metrics

lpbh = cv2.face.LBPHFaceRecognizer_create()

#./darknet.exe detector train .\face.data .\face.cfg yolov4.conv.137

idsFrgc, facesFrgc, grupoteste = bf.getImagemComIdFrgc()
idsArface, facesArfaces = bf.getImagemComIdArface()

bf.trainamentoFrgc(idsFrgc, facesFrgc)
bf.trainamentoArface(idsArface, facesArfaces, )

frgc = bf.detectorFacialFRGC(grupoteste)
arface = bf.detectorFacialArfaces()

# Y_test é o vetor dos valores com os quais você quer testar a previsão
# Y_probas é obtido com Y_probas = dt.predict_proba( X_test )
# Y_probas é o array contendo probabilidades da previsão ser
# positiva no nível das folhas da sua árvore de decisão

#fpr, tpr, thresholds = metrics.roc_curve(Y_test.values, Y_probas[:, 1])

#fv.plot_ROC( fpr, tpr, auc )
#plt.show()

#train = open("train.txt", "x")
#train = open("train.txt", "a")

#test = open("test.txt", "x")
#test = open("test.txt", "a")

#caminho = (glob.glob("./datasets/FRGC/Treinamento/*.txt"))
#randomValue = random.sample(caminho, 48)

#for f in caminho:
#    
#    if f not in randomValue:
#        train.write(f + '\n')
#    else:
#        test.write(f + '\n')


