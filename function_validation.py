# importe suas outras libraries, tipo:  import numpy as np

# para calcular a ROC
#from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def plot_ROC( falsePositiveRate, truePositiveRate, areaUnderCurve ):
    fig = plt.figure()
    fig.set_size_inches( 15, 5 )
    rocCurve = fig.add_subplot( 1, 2, 1 )

    rocCurve.plot( falsePositiveRate, truePositiveRate, color = 'darkgreen',
             lw = 2, label = 'ROC curve (area = %0.2f)' % areaUnderCurve )
    rocCurve.plot( [0, 1], [0, 1], color = 'navy', lw = 1, linestyle = '--' )
    rocCurve.grid()
    plt.xlim( [0.0, 1.0] )
    rocCurve.set_xticks( np.arange( -0.1, 1.0, 0.1 ) )
    plt.ylim( [0.0, 1.05] )
    rocCurve.set_yticks( np.arange( 0, 1.05, 0.1 ) )
    plt.xlabel( 'False Positive Rate' )
    plt.ylabel( 'True Positive Rate' )
    plt.title( 'ROC' )
    rocCurve.legend( loc = "lower right" )
    return plt