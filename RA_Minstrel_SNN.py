import warnings
import os
import pandas
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

# warning messages
MSG_INIT_SNN_MODEL_PREFIX_NOT_STRING                        = "[snn_model_prefix] is not a string.";
MSG_INIT_SNN_MODEL_PREFIX_EMPTY                             = "[snn_model_prefix] cannot be empty.";
MSG_INIT_SNN_MODEL_PREFIX_NOT_EXIST                         = "[snn_model_prefix] does not exist.";
MSG_INIT_DATA_FILE_NOT_EXIST                                = "[data_file_path] does not exist.";
MSG_INIT_DATA_FILE_COL_SNR_ILLEGAL                          = "[col_no_snr] must be an integer.";
MSG_INIT_DATA_FILE_COL_SNR_INDEX_ERR                        = "[col_no_snr] outstrips data index range.";
MSG_INIT_DATA_FILE_COL_SNR_EQUAL_TO_MCS                     = "[col_no_snr] equal to [col_no_mcs].";
MSG_INIT_DATA_FILE_COL_SNR_EQUAL_TO_TRANSMISSION_RESULT     = "[col_no_snr] equal to [col_no_transmission_result].";
MSG_INIT_DATA_FILE_COL_MCS_INDEX_ERR                        = "[col_no_mcs] outstrips data index range.";
MSG_INIT_DATA_FILE_COL_TRANSMISSION_RESULT_INDEX_ERR        = "[col_no_transmission_result] outstrips data index range.";
MSG_INIT_DATA_FILE_COL_TRANSMISSION_RESULT_ILLEGAL          = "[col_no_transmission_result] must be an integer.";
MSG_INIT_DATA_FILE_COL_TRANSMISSION_RESULT_EQUAL_TO_MCS     = "[col_no_transmission_result] equal to [col_no_mcs].";
MSG_INIT_DATA_FILE_MCS_NOT_ASSIGNED                         = "[data_file_mcs_assigend] is not assigned while [col_no_mcs] is empty.";
MSG_INIT_DATA_FILE_MCS_ILLEGAL                              = "[data_file_mcs_assigend] is not supported.";
MSG_INIT_DATA_FILE_COL_SNR_TYPE_ILLEGAL                     = "[col_no_snr_type] is not supported.";

class RA_Minstrel_SNN:
    # constants
    # Date Colunm
    COL_SNR_DEFAULT_INDEX                   = 0;
    COL_MCS_DEFAULT_INDEX                   = None;
    COL_TRANSMISSION_RESULT_DEFAULT_INDEX   = 1;
    # Date Types
    COL_SNR_TYPE_LOG                        = 0;
    COL_SNR_TYPE_LINEAR                     = 1;
    COL_SNR_TYPES                           = [COL_SNR_TYPE_LOG, COL_SNR_TYPE_LINEAR];

    # SNN model names
    SNN_MODEL_PREFIX_DEFAULT    = "_build/models/";
    SNN_MODEL_FILE_NAME_PREFIX  = "SNN_MODEL_NAME_MCS_";
    SNN_MODEL_FILE_NAME_SUFFIX  = ".h5";
    SNN_MODEL_SUPPORTED_MCS     = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 110,
                                   20, 21, 22, 23, 24, 25, 26, 27, 28,
                                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49];
    
    # date file
    col_no_snr                  = None;
    col_no_mcs                  = None;
    col_no_transmission_result  = None;
    
    # SNN models
    snn_model_prefix            = SNN_MODEL_PREFIX_DEFAULT;
    snn_models                  = [];
    
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
        for snn_model_mcs_index in RA_Minstrel_SNN.SNN_MODEL_SUPPORTED_MCS:
            try:
                snn_model_tmp = keras.models.load_model(f'{snn_model_prefix}{RA_Minstrel_SNN.SNN_MODEL_FILE_NAME_PREFIX}{snn_model_mcs_index}{RA_Minstrel_SNN.SNN_MODEL_FILE_NAME_SUFFIX}');
                self.snn_models.append(snn_model_tmp);
            except:
                self.snn_models.append(None);
            
    '''
    train 30 models for MCS1_0 to MCS4_10 (LeTian MCS) in `MCS (802.11ah), NSS = 1, Guard Time(GI) = 8us`
    @data_file_path:                  the data file path
    @data_file_mcs_assigend:          the mcs index
    @col_no_snr:                      snr column index
    @col_no_mcs:                      mcs column index
    @col_no_transmission_result:      transmission result index
    '''
    def train(self, data_file_path, *, 
              data_file_mcs_assigend        = None,
              col_no_snr                    = COL_SNR_DEFAULT_INDEX,
              col_no_mcs                    = COL_MCS_DEFAULT_INDEX,
              col_no_transmission_result    = COL_TRANSMISSION_RESULT_DEFAULT_INDEX,
              col_no_snr_type               = COL_SNR_TYPE_LINEAR):
        # date file
        # check whether the data file exists
        if not os.path.exists(data_file_path):
            raise Exception(MSG_INIT_DATA_FILE_NOT_EXIST);
        # columns
        # check format
        if not isinstance(col_no_snr, int):
            raise Exception(MSG_INIT_DATA_FILE_COL_SNR_ILLEGAL);
        if not isinstance(col_no_transmission_result, int):
            raise Exception(MSG_INIT_DATA_FILE_COL_TRANSMISSION_RESULT_ILLEGAL);
        # check repetation
        if isinstance(col_no_mcs, int) and col_no_snr == col_no_mcs:
            raise Exception(MSG_INIT_DATA_FILE_COL_SNR_EQUAL_TO_MCS);
        if col_no_snr == col_no_transmission_result:
            raise Exception(MSG_INIT_DATA_FILE_COL_SNR_EQUAL_TO_TRANSMISSION_RESULT);
        if isinstance(col_no_mcs, int) and col_no_transmission_result == col_no_mcs:
            raise Exception(MSG_INIT_DATA_FILE_COL_TRANSMISSION_RESULT_EQUAL_TO_MCS);
        # check whether MCS is assigned
        if not isinstance(data_file_mcs_assigend, int) and not isinstance(col_no_mcs, int):
            raise Exception(MSG_INIT_DATA_FILE_MCS_NOT_ASSIGNED);
        # check SNR type
        if col_no_snr_type not in RA_Minstrel_SNN.COL_SNR_TYPES:
            raise Exception(MSG_INIT_DATA_FILE_COL_SNR_TYPE_ILLEGAL);
        
        # load the data file
        data_frame = pandas.read_csv(data_file_path, header=None);
        # sample date (here, we keep them all but disorder them)
        data_frame_sampled = data_frame.sample(frac=1);
        # remove the index
        data = data_frame_sampled.values;
        # retrieve the features (SINR) & targets
        features = None;
        targets = None;
        try:
            features = data[:, col_no_snr].astype(float);
        except IndexError:
            raise Exception(MSG_INIT_DATA_FILE_COL_SNR_INDEX_ERR);
        try:
            targets = data[:, col_no_transmission_result].astype(float);
        except IndexError:
            raise Exception(MSG_INIT_DATA_FILE_COL_TRANSMISSION_RESULT_INDEX_ERR);
        
        # train the neural network
        # train one neural
        if isinstance(data_file_mcs_assigend, int):
            # only train while the MCS is supported 
            if data_file_mcs_assigend not in RA_Minstrel_SNN.SNN_MODEL_SUPPORTED_MCS:
                warnings.warn(MSG_INIT_DATA_FILE_MCS_ILLEGAL);
            else:
                # expand dimensions
                features = np.expand_dims(features, -1);
                targets = np.expand_dims(targets, -1);
                pass
        # train all neurals (based on the current MCS)
        elif isinstance(col_no_mcs, int):
            mcs = None;
            try:
                mcs =  data[:, col_no_mcs].astype(int);
            except IndexError:
                raise Exception(MSG_INIT_DATA_FILE_COL_MCS_INDEX_ERR);
            pass
        
        return None;
    