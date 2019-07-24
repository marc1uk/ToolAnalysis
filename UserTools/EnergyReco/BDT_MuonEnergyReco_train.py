##### Script for Muon Energy Reconstruction in the water tank
import Store
import sys
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
import pickle

def Initialise():
    return 1

def Finalise():
    return 1

def Execute(Toolchain=True, trainingdatafilename=None, E_threshold=None, modelfilename=None):
    # Set TF random seed to improve reproducibility
    seed = 170
    np.random.seed(seed)

    #--- load events for training
    if Toolchain:
        trainingdatafilename = Store.GetStoreVariable('Config','MuonEnergyTrainingDataFile')
    print( "--- opening training file "+trainingdatafilename)
    trainingfile = open(trainingdatafilename)
    print("evts for training in: ",trainingfile)
    #--- load csv file into pandas DataFrame
    trainingfiledata=pd.read_csv(trainingfile)
    trainingfile.close()
    #--- Extract named columns
    TrainingDataset=trainingfiledata[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','neutrinoE','trueKE','diffDirAbs','TrueTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
    #--- place an upper limit on Muon energies we will train on, filtering only passing rows
    if Toolchain:
        E_threshold = Store.GetStoreVariable('Config','BDT_NuE_threshold')
    #dfsel_train=TrainingDataset.loc[TrainingDataset['neutrinoE'] < E_threshold]
    dfsel_train=TrainingDataset

    #--- print to check:
    print("check training sample: ",dfsel_train.head())
#    print(dfsel_train.iloc[5:10,0:5])
    # check fr NaN values:
    assert(dfsel_train.isnull().any().any()==False)

    # apply normalization scalings
    dfsel_train_normalised = pd.DataFrame([ dfsel_train['DNNRecoLength']/600., dfsel_train['TrueTrackLengthInMrd']/200., dfsel_train['diffDirAbs'], dfsel_train['recoDWallR'], dfsel_train['recoDWallZ'], dfsel_train['totalLAPPDs']/200., dfsel_train['totalPMTs']/200., dfsel_train['vtxX']/150., dfsel_train['vtxY']/200., dfsel_train['vtxZ']/150. ]).T

    print("check train sample normalisation: ", dfsel_train_normalised.head())

    #--- convert features and labels DataFrames to numpy arrays:
    features_train = np.array(dfsel_train_normalised[['DNNRecoLength','TrueTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']])
    labels_train = np.array(dfsel_train[['trueKE']])
    print('events for training: ',len(labels_train))
    print("features_train.shape: ",features_train.shape)
    
    ########### BDTG ############
    # build model
    n_estimators=500
    params = {'n_estimators':n_estimators, 'max_depth': 10,
              'learning_rate': 0.05, 'loss': 'lad'}
    
    # train
    print("training BDTG...")
    net_hi_E = ensemble.GradientBoostingRegressor(**params)
    model = net_hi_E.fit(features_train, labels_train.reshape(-1))  # n.b. returns self: model==net_hi_E
    net_hi_E

    # save the model to disk
    if Toolchain:
        modelfilename = Store.GetStoreVariable('Config','BDTMuonModelFile')
    print("saving model to ",modelfilename)
    pickle.dump(model, open(modelfilename, 'wb'))
    
    return 1

if __name__ == "__main__":
    # Make the script runnable as a standalone python script too?
    trainingdatafilename =  '../LocalFolder/BDT_training_input.csv'
    modelfilename = '../LocalFolder/finalized_BDTmodel_forMuonEnergy.sav'
    E_threshold=2.
    Execute(False, trainingdatafilename, E_threshold, modelfilename)
