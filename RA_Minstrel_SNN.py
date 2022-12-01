import pandas
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint


class RA_Minstrel_SNN:
    # constants
    # date file column type
    DATA_FILE_COL_TYPE_MCS_INDEX = 1;               # numberical number of Letian defined types
    DATA_FILE_COL_TYPE_SNR = 2;                     # linear SNR (the interference is attached inside)
    DATA_FILE_COL_TYPE_TRANSMISSION_RES = 3;        # 1: success, 0: failure
    
    
    # SNN models
    snn_model_mcs_10 = None;
    snn_model_mcs_11 = None;
    snn_model_mcs_12 = None;
    snn_model_mcs_13 = None;
    snn_model_mcs_14 = None;
    snn_model_mcs_15 = None;
    snn_model_mcs_16 = None;
    snn_model_mcs_17 = None;
    snn_model_mcs_18 = None;
    snn_model_mcs_19 = None;
    snn_model_mcs_110 = None;
    snn_model_mcs_20 = None;
    snn_model_mcs_21 = None;
    snn_model_mcs_22 = None;
    snn_model_mcs_23 = None;
    snn_model_mcs_24 = None;
    snn_model_mcs_25 = None;
    snn_model_mcs_26 = None;
    snn_model_mcs_27 = None;
    snn_model_mcs_28 = None;
    snn_model_mcs_40 = None;
    snn_model_mcs_41 = None;
    snn_model_mcs_42 = None;
    snn_model_mcs_43 = None;
    snn_model_mcs_44 = None;
    snn_model_mcs_45 = None;
    snn_model_mcs_46 = None;
    snn_model_mcs_47 = None;
    snn_model_mcs_48 = None;
    snn_model_mcs_49 = None;
    
    '''
    init 
    @snn_model_path_prefix: the default folder to store trained models
    @snn_model_mcs_10_path: 
    @snn_model_mcs_10_path: 
    @snn_model_mcs_11_path: 
    @snn_model_mcs_12_path: 
    @snn_model_mcs_13_path: 
    @snn_model_mcs_14_path: 
    @snn_model_mcs_15_path: 
    @snn_model_mcs_16_path: 
    @snn_model_mcs_17_path: 
    @snn_model_mcs_18_path: 
    @snn_model_mcs_19_path: 
    @snn_model_mcs_110_path: 
    @snn_model_mcs_20_path: 
    @snn_model_mcs_21_path: 
    @snn_model_mcs_22_path: 
    @snn_model_mcs_23_path: 
    @snn_model_mcs_24_path: 
    @snn_model_mcs_25_path: 
    @snn_model_mcs_26_path: 
    @snn_model_mcs_27_path: 
    @snn_model_mcs_28_path: 
    @snn_model_mcs_40_path: 
    @snn_model_mcs_41_path: 
    @snn_model_mcs_42_path: 
    @snn_model_mcs_43_path: 
    @snn_model_mcs_44_path: 
    @snn_model_mcs_45_path: 
    @snn_model_mcs_46_path: 
    @snn_model_mcs_47_path: 
    @snn_model_mcs_48_path: 
    @snn_model_mcs_49_path: 
    '''
    def __init__(self, *, models_paths = None):
        return None;
    
    '''
    train 30 models for MCS1_0 to MCS4_10 (LeTian MCS) in `MCS (802.11ah), NSS = 1, Guard Time(GI) = 8us`
    '''
    def train(self):
        return None;
    