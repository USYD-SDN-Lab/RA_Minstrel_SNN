import sys 
sys.path.append("../..");
import os
from RA_Minstrel_SNN import RA_Minstrel_SNN;

# set prefix to store models and 
model_prefix = "_build_/models_TestAllFunctions/";
if not os.path.exists(model_prefix):
    os.makedirs(model_prefix);

# init the model
rms = RA_Minstrel_SNN(snn_model_prefix = model_prefix);

# train 1MHz
rms.train('Train_Data_New/mcs1.csv', data_file_mcs_assigend = 11);
rms.train('Train_Data_New/mcs2.csv', data_file_mcs_assigend = 12);
rms.train('Train_Data_New/mcs3.csv', data_file_mcs_assigend = 13);
rms.train('Train_Data_New/mcs4.csv', data_file_mcs_assigend = 14);
rms.train('Train_Data_New/mcs5.csv', data_file_mcs_assigend = 15);
rms.train('Train_Data_New/mcs6.csv', data_file_mcs_assigend = 16);
rms.train('Train_Data_New/mcs7.csv', data_file_mcs_assigend = 17);
rms.train('Train_Data_New/mcs8.csv', data_file_mcs_assigend = 18);
rms.train('Train_Data_New/mcs9.csv', data_file_mcs_assigend = 19);
# train 2MHz
rms.train('Train_Data_New/mcs10.csv', data_file_mcs_assigend = 20);
rms.train('Train_Data_New/mcs11.csv', data_file_mcs_assigend = 21);
rms.train('Train_Data_New/mcs12.csv', data_file_mcs_assigend = 22);
rms.train('Train_Data_New/mcs13.csv', data_file_mcs_assigend = 23);
rms.train('Train_Data_New/mcs14.csv', data_file_mcs_assigend = 24);
rms.train('Train_Data_New/mcs15.csv', data_file_mcs_assigend = 25);
rms.train('Train_Data_New/mcs16.csv', data_file_mcs_assigend = 26);
rms.train('Train_Data_New/mcs17.csv', data_file_mcs_assigend = 27);
rms.train('Train_Data_New/mcs18.csv', data_file_mcs_assigend = 28);
# train 4MHz
rms.train('Train_Data_New/mcs19.csv', data_file_mcs_assigend = 40);
rms.train('Train_Data_New/mcs20.csv', data_file_mcs_assigend = 41);
rms.train('Train_Data_New/mcs21.csv', data_file_mcs_assigend = 42);
rms.train('Train_Data_New/mcs22.csv', data_file_mcs_assigend = 43);
rms.train('Train_Data_New/mcs23.csv', data_file_mcs_assigend = 44);
rms.train('Train_Data_New/mcs24.csv', data_file_mcs_assigend = 45);
rms.train('Train_Data_New/mcs25.csv', data_file_mcs_assigend = 46);
rms.train('Train_Data_New/mcs26.csv', data_file_mcs_assigend = 47);
rms.train('Train_Data_New/mcs27.csv', data_file_mcs_assigend = 48);
rms.train('Train_Data_New/mcs28.csv', data_file_mcs_assigend = 49);