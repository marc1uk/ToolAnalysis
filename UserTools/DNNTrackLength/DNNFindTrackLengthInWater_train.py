##### Script to Train DNN for Track Length Reconstruction in the water tank
def Initialise(pyinit):
    if (pyinit==1):
        print("importing all the things")
        #from myimports import *
#        import myimports
        global Store
        global sys
        global glob
        global numpy
        global pandas
        global tensorflow
        global tempfile
        global random
        global csv
        global matplotlib
        global pyplot
        global array
        global sklearn
        global datasets
        global metrics
        global model_selection
        global preprocessing
        global keras
        global Sequential
        global Dense
        global ModelCheckpoint
        global KerasRegressor
        importlib = __import__('importlib')
        Store = __import__('Store', globals(), locals())
        sys = __import__('sys', globals(), locals())
        glob = __import__('glob', globals(), locals())
        numpy = __import__('numpy', globals(), locals())
        pandas = __import__('pandas', globals(), locals())
        tensorflow = __import__('tensorflow', globals(), locals())
        tempfile = __import__('tempfile', globals(), locals())
        random = __import__('random', globals(), locals())
        csv = __import__('csv', globals(), locals())
        matplotlib = __import__('matplotlib', globals(), locals())
        matplotlib.use('Agg')
        pyplot = __import__('matplotlib.pyplot',globals(),locals(),['matplotlib'])
        array = __import__('array',globals(), locals(),fromlist=['array'])
        sklearn = __import__('sklearn', globals(), locals())
        _datasets = __import__('sklearn', globals(), locals(),['datasets'])
        datasets=_datasets.datasets
        _metrics = __import__('sklearn', globals(), locals(),['metrics'])
        metrics=_metrics.metrics
        _model_selection = __import__('sklearn', globals(), locals(),['model_selection'])
        model_selection=_model_selection.model_selection
        _preprocessing = __import__('sklearn', globals(), locals(),['preprocessing'])
        preprocessing=_preprocessing.preprocessing
        keras = tensorflow.keras
        Sequential = keras.models.Sequential
        Dense = keras.layers.Dense
        ModelCheckpoint = tensorflow.keras.callbacks.ModelCheckpoint
        KerasRegressor = keras.wrappers.scikit_learn.KerasRegressor

#        import Store
#        import sys
#        import glob
#        import numpy #as np
#        import pandas #as pd
#        import tensorflow #as tf
#        import tempfile
#        import random
#        import csv
#        import matplotlib
#        matplotlib.use('Agg')
#        import matplotlib.pyplot #as plt
#        from array import array
#        from sklearn import datasets
#        from sklearn import metrics
#        from sklearn import model_selection
#        from sklearn import preprocessing
#        from tensorflow import keras
#        from tensorflow.keras.models import Sequential
#        from tensorflow.keras.layers import Dense
#        from tensorflow.keras.callbacks import ModelCheckpoint
#        from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
    else:
        print("skipping the import")
    return 1


#def Initialise():
#    return 1

def Finalise():
    print("DNNFineTrackLengthInWater_train.py finalise")
    return 1

def create_model():
    print("DNNFineTrackLengthInWater_train.py defining Create Model")
    # create model
    model = Sequential()
    print("DNNFineTrackLengthInWater_train.py Sequential done")
    model.add(Dense(50, input_dim=2203, kernel_initializer='he_normal', activation='relu'))
    print("Addded first layer, adding more")
    model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='he_normal', activation='relu'))
    # Compile model
    print("Compiling model")
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    print("returning model")
    return model

def Execute(Toolchain=True, trainingdatafilename=None, weightsfilename=None):
    print("DNNFineTrackLengthInWater_train.py Executing")
    # train the model
    # Set TF random seed to improve reproducibility
    seed = 150
    numpy.random.seed(seed)

    #--- events for training - MC events
    # get training data file path from store
    if Toolchain:
        trainingdatafilename = Store.GetStoreVariable('Config','TrackLengthTrainingDataFile')
    # open the file
    print("opening training file "+trainingdatafilename)
    trainingfile = open(trainingdatafilename)
    print("evts for training in: ",trainingfile)
    # read into a pandas structure
    print("reading file with pandas")
    trainingfiledata = pandas.read_csv(trainingfile)
    print("closing file")
    trainingfile.close()
    # convert to 2D numpy array
    print("converting to numpy array")
    TrainingDataset = numpy.array(trainingfiledata)
    # split the numpy array up into sub-arrays
    print("splitting up into features, labels etc")
    features, lambdamax, labels, rest = numpy.split(TrainingDataset,[2203,2204,2205],axis=1)
    # This puts splits the arrays column-wise as follows:
    # 0-2202 into 'features', element 2203 into 'lambdamax', 2204 into 'labels' and 2205+ into 'rest'
    # csv file columns are:
    # 0-1099: hit lambda values, 1100-2199: hit times, 2200: lambda_max, 2201: Num PMT hits,
    # 2202: Num LAPPD hits, 2203: lambda_max (again), 2204: TrueTrackLengthInWater, 2205+: nuE, muE ... etc
    
    # print info, initialize seed
    print( "lambdamax ", lambdamax[:2], labels[:2])
    print(features[0])
    num_events, num_pixels = features.shape
    print(num_events, num_pixels)
    numpy.random.seed(0)
    
    # rename variables for obfuscation
    train_x = features
    train_y = labels
    print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

    # Scale the training set to 0 mean and unit standard deviation.
    print("scaling to 0 mean and unit std-dev")
    scaler = preprocessing.StandardScaler()
    train_x = scaler.fit_transform(train_x)
    
    # Construct the DNN model
    print("constructing KerasRegressor")
    estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=2, verbose=0)

    # load weights
    if Toolchain:
        weightsfilename = Store.GetStoreVariable('Config','TrackLengthWeightsFile')
    print("setting up checkpoint callback to save weights to "+weightsfilename)
    checkpoint = ModelCheckpoint(weightsfilename, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    callbacks_list = [checkpoint]
    # Run the model
    print('training....')
    history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=10, batch_size=2, callbacks=callbacks_list, verbose=0)
    print("done training")

    # summarize history for loss
    print("making loss plots")
    f, ax2 = matplotlib.pyplot.subplots(1,1)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('Performance')
    ax2.set_xlabel('Epochs')
    #ax2.set_xlim(0.,10.)
    ax2.legend(['loss', 'val_loss'], loc='upper left')
    matplotlib.pyplot.savefig("../LocalFolder/keras_DNN_training_loss.pdf")

    print("DNNFindTrackLengthInWater_train.py returning from Execute")
    return 1

if __name__ == "__main__":
    # Make the script runnable as a standalone python script too?
    print("DNNFindTrackLengthInWater_train.py called as main")
    trainingdatafilename = '../LocalFolder/DNN_training_input.csv'
    weightsfilename = '../LocalFolder/weights_bets.hdf5'
    print("calling Execute with training data "+trainingdatafilename+" to save weights to "+weightsfilename)
    Execute(False, trainingdatafilename, weightsfilename)
