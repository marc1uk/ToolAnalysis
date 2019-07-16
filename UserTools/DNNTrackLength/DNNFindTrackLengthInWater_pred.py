##### Script Track Length Reconstruction in the water tank
# bend over backwards for reproducible results
# see https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import tensorflow as tf
import random as rn
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(0) #42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from tensorflow.keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import Store
import sys
import glob
#import numpy as np
import pandas as pd
#import tensorflow as tf
import tempfile
#import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def Initialise():
    return 1

def Finalise():
    # retrieve the accumulated square errors and calculate the MSE
    sum_square_errors = Store.GetStoreVariable('EnergyReco','TrackLengthInWaterSumSquaredErrors')
    evtnum = Store.GetStoreVariable('ANNIEEvent','EventNumber')
    mean_sq_err = sum_square_errors / evtnum
    print('MSE (sklearn): {0:f}'.format(mean_sq_err))
    # This is, of course, meaningless unless we have truth data
    
    return 1

def Execute():
    
    # Load Data
    #-----------------------------
    print( "--- loading input variables from store!")
    hit_lambdas = Store.GetStoreVariable('EnergyReco','lambda_vec')  # std::vectors in the Store are returned
    hit_times = Store.GetStoreVariable('EnergyReco','digit_ts_vec')  # as python lists
    lambdamax = Store.GetStoreVariable('EnergyReco','lambda_max')
    num_pmt_hits = Store.GetStoreVariable('EnergyReco','num_pmt_hits')
    num_lappd_hits = Store.GetStoreVariable('EnergyReco','num_lappd_hits')
    features = np.concatenate([hit_lambdas,hit_times,lambdamax,num_pmt_hits,num_lappd_hits], axis=None)
    labels = Store.GetStoreVariable('EnergyReco','TrueTrackLengthInWater')
    
    # print info
    print( "lambdamax ", lambdamax[:2], labels[:2])
    print(features[0])
    num_events, num_pixels = features.shape
    print(num_events, num_pixels)
    
    # Preprocess data and load model
    #-----------------------------
    
    # rename variables for obfuscation
    test_x = features
    test_y = labels

    # Scale data to 0 mean and unit standard deviation.
    scaler = preprocessing.StandardScaler()
    x_transformed = scaler.transform(test_x)
    
    # define keras model, loading weight from weights file
    model = Sequential()
    model.add(Dense(50, input_dim=2203, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='he_normal', activation='relu'))

    # load weights
    if Toolchain:
        weightsfilename = Store.GetStoreVariable('Config','TrackLengthWeightsFile')
    model.load_weights(weightsfilename)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    print("Created model and loaded weights from file"+weightsfilename)
    
    # Do prediction
    #-----------------------------
    print('predicting...')
    y_predicted = model.predict(x_transformed)
    
    # Score accuracy (if the truth label is meaningful)
    #-----------------------------
    # Update the accumulated square error for the MSE calculation (DNN accuracy score)
    evtnum = Store.GetStoreVariable('ANNIEEvent','EventNumber')
    if evtnum==0:
        sum_square_errors = 0
    else:
        sum_square_errors = Store.GetStoreVariable('EnergyReco','TrackLengthInWaterSumSquaredErrors')
    sum_square_errors += ((y_predicted-test_y)**2.)
    # the final result will be calculated and printed in 'finalise'
    
    
    # Store outputs
    #-----------------------------
    # put the predicted lengths and errors if we have them into the BoostStore
    Store.SetStoreVariable('EnergyReco','DNNRecoLength',y_predicted)
    Store.SetStoreVariable('EnergyReco','TrackLengthInWaterSumSquaredErrors',sum_square_errors)
    #print("shapes: ", test_y.shape, ", ", y_predicted.shape)
    
    #-----------------------------
    # Backward Compatibility
    #-----------------------------
    # append this entry to the old-style csv file, for validation while we migrate
    # Append the predicted track length to the input file and write to new csv file
    #print(" saving .csv file with energy variables..")
    outputfilepath = Store.GetStoreVariable('Config','TrackLengthPredictionsDataFile')
    if outputfilepath == 'NA':
        return 1  # if not saving to legacy file, just return
    
    # check if this is the first execute iteration. if so, we'll create a header row first.
    if evtnum==0:
        headers=['']
        for i in range(len(hit_lambdas)):
            headers.append('l_'+str(i))
        for i in range(len(hit_lambdas)):
            headers.append('T_'+str(i))
        headers = headers+['lambda_max', 'totalPMTs', 'totalLAPPDs', 'TrueTrackLengthInWater', 'neutrinoE', 'trueKE', 'diffDirAbs', 'TrueTrackLengthInMrd', 'recoDWallR', 'recoDWallZ', 'dirX', 'dirY', 'dirZ', 'vtxX', 'vtxY', 'vtxZ','TrueTrackLengthInWater','DNNRecoLength']
        # convert to pandas dataframe
        headersframe = pd.DataFrame(headers)
        # write the headers to file
        headersframe.T.to_csv(outputfilepath,header=False,index=False)

    # Many of the variables normally written to legacy csv file are not needed for the DNN;
    # they're written to the file as they're needed for BDT training.
    truenue = Store.GetStoreVariable('EnergyReco','trueNeuE')
    truemue = Store.GetStoreVariable('EnergyReco','trueEnergy')
    diffdirabs = Store.GetStoreVariable('EnergyReco','diffDirAbs2')
    truemrdtracklen = Store.GetStoreVariable('EnergyReco','TrueTrackLengthInMrd2')
    recodwallr2 = Store.GetStoreVariable('EnergyReco','recoDWallR2')
    recodwallz2 = Store.GetStoreVariable('EnergyReco','recoDWallZ2')
    dirvec = Store.GetStoreVariable('EnergyReco','dirVec')  # these are 3-element lists
    vtxvec = Store.GetStoreVariable('EnergyReco','vtxVec')  #
    
    # combine everything into a numpy array
    data = np.concatenate((evtnum,features,labels,truenue,truemue,diffdirabs,truemrdtracklen,recodwallr2,recodwallz2,dirvec,vtxvec,test_y, y_predicted),axis=1)
    # convert to pandas dataframe
    filedata=pd.DataFrame(data)
    
    # then append this event's data row to the output file
    filedata.to_csv(outputfilepath, float_format = '%.3f', mode='a', header=False, index=False)

    # some quick validation tests:
    print("checking..."," df0.shape[0]: ",df0.shape[0]," len(y_predicted): ", len(y_predicted))
    assert(df0.shape[0]==len(y_predicted))
    assert(df_final.shape[0]==df.shape[0])

    #---if asserts fails check dimensions with these print outs:
    #print("df: ",df.head())
    #print(df.iloc[:,2200:])
    #print(df0.head())
    #print(df0.shape)
    #print(df0.iloc[:,2200:])
    #print(df_final.shape)
    #print(df_final.iloc[:,2200:])


    return 1

