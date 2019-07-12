##### Script To Validate DNN for Track Length Reconstruction in the water tank
import Store
import sys
import glob
import numpy #as np
import pandas #as pd
import tensorflow #as tf
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot #as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as K

#import sys
#def load(name):
#        if name in sys.modules: # and name not in self.plugins:
#            # already loaded: return it from sys.modules and add it to our list
#            module = sys.modules[name]
#        else:
#            #module = importlib.import_module(name)
#            module = __import__(name)
#        self.plugins.add(name)
#        return module

import pprint
def Initialise(pyinit):
    print("Initialising DNNFindTrackLengthInWater_test.py")
    #print("Listing what's in globals")
    #pprint.pprint(globals())
#    if (pyinit==1):
#        print("importing all the things")
#        #from myimports import *
#        import myimports
#    else:
#        print("skipping the import")
#    
#    import Store
#    import sys
#    import glob
#    import numpy as np
#    import pandas as pd
#    import tensorflow as tf
#    import tempfile
#    import random
#    import csv
#    import matplotlib
#    matplotlib.use('Agg')
#    import matplotlib.pyplot as plt
#    from array import array
#    from sklearn import datasets
#    from sklearn import metrics
#    from sklearn import model_selection
#    from sklearn import preprocessing
#    from tensorflow import keras
#    from tensorflow.keras.models import Sequential
#    from tensorflow.keras.layers import Dense
#    from tensorflow.keras.callbacks import ModelCheckpoint
#    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
    return 1

def Finalise():
    return 1

def Execute(Toolchain=True, testingdatafilename=None, weightsfilename=None, predictionsdatafilename=None, firstfilesentries=None, predictionsdatafilename2=None):
    print("DNNFindTrackLengthInWater_test.py Executing")
    
    # Load Data
    #-----------------------------
    if Toolchain:
        testingdatafilename = Store.GetStoreVariable('Config','TrackLengthTestingDataFile')
    # open the file
    print("opening testing data file "+testingdatafilename)
    testfile = open(testingdatafilename)
    print("evts for testing in: ",testfile)
    # read into a pandas structure
    print("reading file with pandas")
    testfiledata = pandas.read_csv(testfile)
    print("closing file")
    testfile.close()
    # convert to 2D numpy array
    print("converting to numpy array")
    TestingDataset = numpy.array(testfiledata)
    # split the numpy array up into sub-arrays
    testfeatures, testlambdamax, testlabels, testrest = numpy.split(TestingDataset,[2203,2204,2205],axis=1)

    # print info
    print( "lambdamax ", testlambdamax[:2], testlabels[:2])
    print(testfeatures[0])
    num_events, num_pixels = testfeatures.shape
    print(num_events, num_pixels)
    
    # Preprocess data and load model
    #-----------------------------
    
    # rename variables for obfuscation
    test_x = testfeatures
    test_y = testlabels
    print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

    # Scale data to 0 mean and unit standard deviation.
    print("scalign to 0 mean and unit std-dev")
    scaler = preprocessing.StandardScaler()
    x_transformed = scaler.fit_transform(test_x)  # are we ok doing fit_transform on test data?
    # scale the features
    testfeatures_transformed = scaler.transform(testfeatures)
    
    # define keras model, loading weight from weights file
    print("defining the model")
    model = Sequential()
    print("adding layers")
    model.add(Dense(50, input_dim=2203, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='he_normal', activation='relu'))

    # load weights
    if Toolchain:
        weightsfilename = Store.GetStoreVariable('Config','TrackLengthWeightsFile')
    print("loading weights from file "+weightsfilename)
    model.load_weights(weightsfilename)

    # Compile model
    print("compiling model")
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    print("Created model and loaded weights from file"+weightsfilename)

    # Score accuracy / Make predictions
    #----------------------------------
    print('predicting...')
    y_predicted = model.predict(x_transformed)
    
    # estimate accuracy on dataset using loaded weights
    print("evalulating model on test")
    scores = model.evaluate(test_x, test_y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # Score with sklearn.
    print("scoring sk mse")
    score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
    print('MSE (sklearn): {0:f}'.format(score_sklearn))

    # Write to the output csv file
    #-----------------------------
    
    # firstly, maybe we don't want to save the predictions at all. See if we've been given at least one file:
    if Toolchain:
        predictionsdatafilename = Store.GetStoreVariable('Config','TrackLengthPredictionsDataFile')
    if (predictionsdatafilename is None) or ( predictionsdatafilename == ''):
        # no output files today
        print("no output file specified, not writing to file")
        return 1
    
    print("writing predictions to output file "+predictionsdatafilename)
    # build a dataframe from the true and predicted track lengths
    print("building output dataframe")
    outputdataarray = numpy.concatenate((test_y, y_predicted),axis=1)
    outputdataframe=pandas.DataFrame(outputdataarray, columns=['TrueTrackLengthInWater','DNNRecoLength'])
    
    # append as additional columns to the input dataframe
    print("inserting True and Predicted lengths into file data")
    testfiledata.insert(2217, 'TrueTrackLengthInWater', outputdataframe['TrueTrackLengthInWater'].values, allow_duplicates="True")
    testfiledata.insert(2218, 'DNNRecoLength', outputdataframe['DNNRecoLength'].values, allow_duplicates="True")
    
    # check if we're splitting the output into two files (for training/testing the BDTs)
    if Toolchain:
        firstfilesentries = Store.GetStoreVariable('Config','FirstFileEntries')
        predictionsdatafilename2 = Store.GetStoreVariable('Config','TrackLengthPredictionsDataFile2')
    print("will write first "+str(firstfilesentries)+" entries to first output file")
    print("remaining entries will go into "+predictionsdatafilename2)
    
    # write to csv file(s)
    if (firstfilesentries is None) or (firstfilesentries == 0) or (predictionsdatafilename2 is None) or (predictionsdatafilename2 == ''):
        print("writing all data to "+predictionsdatafilename)
        testfiledata.to_csv(predictionsdatafilename, float_format = '%.3f')
    else:
        print("writing split data to files "+predictionsdatafilename+" and "+predictionsdatafilename2)
        testfiledata[firstfilesentries:].to_csv(predictionsdatafilename, float_format = '%.3f')
        testfiledata[:firstfilesentries].to_csv(predictionsdatafilename2, float_format = '%.3f')

    print("clearing session")
    K.clear_session()

    print("done; returning")
    return 1

if __name__ == "__main__":
    # Make the script runnable as a standalone python script too?
    testingdatafilename = '../LocalFolder/DNN_testing_input.csv'
    weightsfilename = '../LocalFolder/weights_bets.hdf5'
    predictionsdatafilename = '../LocalFolder/BDT_training_input.csv'
    firstfilesentries = 1000
    predictionsdatafilename2 = '../LocalFolder/BDT_testing_input.csv'
    Execute(False, testingdatafilename, weightsfilename, predictionsdatafilename, firstfilesentries, predictionsdatafilename2)
