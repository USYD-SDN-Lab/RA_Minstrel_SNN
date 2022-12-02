import os
import pandas
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

# warning messages
MSG_INIT_SNN_MODEL_PREFIX_NOT_STRING    = "[snn_model_prefix] is not a string.";
MSG_INIT_SNN_MODEL_PREFIX_EMPTY         = "[snn_model_prefix] cannot be empty.";
MSG_INIT_SNN_MODEL_PREFIX_NOT_EXIST     = "[snn_model_prefix] does not exist.";

class RA_Minstrel_SNN:
    # constants
    # date file column type
    DATA_FILE_COL_TYPE_MCS_INDEX = 1;               # numberical number of Letian defined types
    DATA_FILE_COL_TYPE_SNR = 2;                     # linear SNR (the interference is attached inside)
    DATA_FILE_COL_TYPE_TRANSMISSION_RES = 3;        # 1: success, 0: failure
    # SNN model names
    SNN_MODEL_PREFIX_DEFAULT    = "_build/models/";
    SNN_MODEL_NAME_MCS_10       = "SNN_MODEL_NAME_MCS_10.h5";
    SNN_MODEL_NAME_MCS_11       = "SNN_MODEL_NAME_MCS_11.h5";
    SNN_MODEL_NAME_MCS_12       = "SNN_MODEL_NAME_MCS_12.h5";
    SNN_MODEL_NAME_MCS_13       = "SNN_MODEL_NAME_MCS_13.h5";
    SNN_MODEL_NAME_MCS_14       = "SNN_MODEL_NAME_MCS_14.h5";
    SNN_MODEL_NAME_MCS_15       = "SNN_MODEL_NAME_MCS_15.h5";
    SNN_MODEL_NAME_MCS_16       = "SNN_MODEL_NAME_MCS_16.h5";
    SNN_MODEL_NAME_MCS_17       = "SNN_MODEL_NAME_MCS_17.h5";
    SNN_MODEL_NAME_MCS_18       = "SNN_MODEL_NAME_MCS_18.h5";
    SNN_MODEL_NAME_MCS_19       = "SNN_MODEL_NAME_MCS_19.h5";
    SNN_MODEL_NAME_MCS_110      = "SNN_MODEL_NAME_MCS_110.h5";
    SNN_MODEL_NAME_MCS_20       = "SNN_MODEL_NAME_MCS_20.h5";
    SNN_MODEL_NAME_MCS_21       = "SNN_MODEL_NAME_MCS_21.h5"; 
    SNN_MODEL_NAME_MCS_22       = "SNN_MODEL_NAME_MCS_22.h5"; 
    SNN_MODEL_NAME_MCS_23       = "SNN_MODEL_NAME_MCS_23.h5"; 
    SNN_MODEL_NAME_MCS_24       = "SNN_MODEL_NAME_MCS_24.h5"; 
    SNN_MODEL_NAME_MCS_25       = "SNN_MODEL_NAME_MCS_25.h5"; 
    SNN_MODEL_NAME_MCS_26       = "SNN_MODEL_NAME_MCS_26.h5"; 
    SNN_MODEL_NAME_MCS_27       = "SNN_MODEL_NAME_MCS_27.h5"; 
    SNN_MODEL_NAME_MCS_28       = "SNN_MODEL_NAME_MCS_28.h5"; 
    SNN_MODEL_NAME_MCS_40       = "SNN_MODEL_NAME_MCS_40.h5"; 
    SNN_MODEL_NAME_MCS_41       = "SNN_MODEL_NAME_MCS_41.h5"; 
    SNN_MODEL_NAME_MCS_42       = "SNN_MODEL_NAME_MCS_42.h5"; 
    SNN_MODEL_NAME_MCS_43       = "SNN_MODEL_NAME_MCS_43.h5"; 
    SNN_MODEL_NAME_MCS_44       = "SNN_MODEL_NAME_MCS_44.h5"; 
    SNN_MODEL_NAME_MCS_45       = "SNN_MODEL_NAME_MCS_45.h5"; 
    SNN_MODEL_NAME_MCS_46       = "SNN_MODEL_NAME_MCS_46.h5"; 
    SNN_MODEL_NAME_MCS_47       = "SNN_MODEL_NAME_MCS_47.h5"; 
    SNN_MODEL_NAME_MCS_48       = "SNN_MODEL_NAME_MCS_48.h5"; 
    SNN_MODEL_NAME_MCS_49       = "SNN_MODEL_NAME_MCS_49.h5";
    
    # SNN models
    snn_model_prefix = SNN_MODEL_PREFIX_DEFAULT;
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
    @snn_model_path_prefix:     the default folder to store trained models (supporting windows & Linux formats, a path prefix or a folder path)
    '''
    def __init__(self, *, snn_model_prefix = None):
        # set the model prefix
        if snn_model_prefix is not None:
            if not isinstance(snn_model_prefix, str):
                raise Exception(MSG_INIT_SNN_MODEL_PREFIX_NOT_STRING);
            elif len(snn_model_prefix) < 1:
                raise Exception(MSG_INIT_SNN_MODEL_PREFIX_EMPTY);
            elif not os.path.exists(snn_model_prefix):
                raise Exception(MSG_INIT_SNN_MODEL_PREFIX_NOT_EXIST);
            else:
                # change '\' into '/' 
                snn_model_prefix = snn_model_prefix.replace('\\', '/');
                # if the last character isn't '/', then add '/' to the last
                if snn_model_prefix[-1] != '/':
                    snn_model_prefix = ''.join((snn_model_prefix, '/'))
                # assign
                self.snn_model_prefix = snn_model_prefix;
        # try to load all models
        try:
            self.snn_model_mcs_10 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_10}');
        except:
            pass;
        try:
            self.snn_model_mcs_11 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_11}');
        except:
            pass;
        try:
            self.snn_model_mcs_12 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_12}');
        except:
            pass;
        try:
            self.snn_model_mcs_13 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_13}');
        except:
            pass;
        try:
            self.snn_model_mcs_14 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_14}');
        except:
            pass;
        try:
            self.snn_model_mcs_15 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_15}');
        except:
            pass;
        try:
            self.snn_model_mcs_16 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_16}');
        except:
            pass;
        try:
            self.snn_model_mcs_17 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_17}');
        except:
            pass;
        try:
            self.snn_model_mcs_18 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_18}');
        except:
            pass;
        try:
            self.snn_model_mcs_19 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_19}');
        except:
            pass;
        try:
            self.snn_model_mcs_110 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_110}');
        except:
            pass;
        try:
            self.snn_model_mcs_20 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_20}');
        except:
            pass;
        try:
            self.snn_model_mcs_21 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_21}');
        except:
            pass;
        try:
            self.snn_model_mcs_22 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_22}');
        except:
            pass;
        try:
            self.snn_model_mcs_23 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_23}');
        except:
            pass;
        try:
            self.snn_model_mcs_24 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_24}');
        except:
            pass;
        try:
            self.snn_model_mcs_25 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_25}');
        except:
            pass;
        try:
            self.snn_model_mcs_26 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_26}');
        except:
            pass;
        try:
            self.snn_model_mcs_27 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_27}');
        except:
            pass;
        try:
            self.snn_model_mcs_28 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_28}');
        except:
            pass;
        try:
            self.snn_model_mcs_40 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_40}');
        except:
            pass;
        try:
            self.snn_model_mcs_41 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_41}');
        except:
            pass;
        try:
            self.snn_model_mcs_42 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_42}');
        except:
            pass;
        try:
            self.snn_model_mcs_43 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_43}');
        except:
            pass;
        try:
            self.snn_model_mcs_44 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_44}');
        except:
            pass;
        try:
            self.snn_model_mcs_45 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_45}');
        except:
            pass;
        try:
            self.snn_model_mcs_46 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_46}');
        except:
            pass;
        try:
            self.snn_model_mcs_47 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_47}');
        except:
            pass;
        try:
            self.snn_model_mcs_48 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_48}');
        except:
            pass;
        try:
            self.snn_model_mcs_49 = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_NAME_MCS_49}');
        except:
            pass;
            
    '''
    train 30 models for MCS1_0 to MCS4_10 (LeTian MCS) in `MCS (802.11ah), NSS = 1, Guard Time(GI) = 8us`
    '''
    def train(self):
        return None;
    