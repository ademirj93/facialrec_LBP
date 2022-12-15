import numpy as np
import basic_functions as bf
import function_validation as fv

print('\nDataset FRGC')
idsFrgc, facesFrgc = bf.getImagemComIdFrgc()
bf.trainamentoFrgc(idsFrgc, facesFrgc)
frgc = bf.detectorFacialFRGC()

print('\nDataSet ARFACES')
idsArface, facesArfaces = bf.getImagemComIdArface()
bf.trainamentoArface(idsArface, facesArfaces)
arface = bf.detectorFacialArfaces()
print('\n')

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


