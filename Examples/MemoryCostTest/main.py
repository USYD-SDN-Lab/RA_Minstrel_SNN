import sys 
sys.path.append("../..");
import time
import os
import pandas
import numpy as np
from matplotlib import pyplot as plt
from RA_Minstrel_SNN import RA_Minstrel_SNN;

# set prefix to store models and 
model_prefix = "_build_/models_TestAllFunctions/";
if not os.path.exists(model_prefix):
    os.makedirs(model_prefix);
# set the train data file
file_test = "Data/track_yas-wifi-phy_data-beacon.csv";
# set the train data file
file_train_prefix = "Data/";
file_train_list = ["mcs1.csv",  "mcs2.csv",  "mcs3.csv",  "mcs4.csv",  "mcs5.csv",  "mcs6.csv",  "mcs7.csv",  "mcs8.csv",  "mcs9.csv",
                   "mcs10.csv", "mcs11.csv", "mcs12.csv", "mcs13.csv", "mcs14.csv", "mcs15.csv", "mcs16.csv", "mcs17.csv", "mcs18.csv",
                   "mcs19.csv", "mcs20.csv", "mcs21.csv", "mcs22.csv", "mcs23.csv", "mcs24.csv", "mcs25.csv", "mcs26.csv", "mcs27.csv", "mcs28.csv"];
# set the mcs we need
mcss = [11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49];
# init the model
rms = RA_Minstrel_SNN(snn_model_prefix = model_prefix);
# train
for mcs_index in range(0, len(mcss)):
    mcs = mcss[mcs_index];
    file_train = file_train_prefix + file_train_list[mcs_index];
    if not rms.is_model_loaded(mcs):
        print("MCS %d has to be trained because it was not trained before."%mcs);
        rms.train(file_train, data_file_mcs_assigend = mcs);
        
# have trained
print("Trained models: %d/%d"%(rms.get_loaded_models_num(), rms.get_supported_models_num()));

    
# tell the purpose
print("Tensorflow: model.predict() leaks memory!");
input("Here, we test. Please open your memory monitor app and press enter")

while True:
    for i in range(0, 8192):
        rms.predict(2000);
        print("\r%.2f%%"%(i/8192), end=" ");
    print('\r100%                                 ');
    time.sleep(1);

print("okey");