import sys 
sys.path.append("../..");
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
        rms.train(file_train, data_file_mcs_assigend = mcs);
# have trained
print("Trained models: %d/%d"%(rms.get_loaded_models_num(), rms.get_supported_models_num()));
    
# predict
test_data_frame = pandas.read_csv(file_test, header=None);
test_data = test_data_frame.values;
#test_data = test_data[0:23,:];
#test_data = test_data[12197:12392,:];
print("Predicted Data Length = %d"%len(test_data));
beacon_time = [];
time = [];
snr  = [];
snr_snn = [];
snr_snn_tmp_bef = 0;
snr_snn_tmp_cur = 0;
test_mcs_snn = [];
test_mcs_snn_bef = 10;
test_mcs_snn_cur = 10;
test_mcs_snn_perfect = [];
test_mcs_minstrel = [];

for i in range(0, len(test_data)):
    percent = i/len(test_data)*100;
    print("\r%.2f%%"%percent, end=" ");
    # base on the packet size to do
    if test_data[i, 0] != 166:
        # not data packet
        # add beacon time
        beacon_time.append(test_data[i, 2]);
        # update the current mcs
        test_mcs_snn_cur = test_mcs_snn_bef;
        # update the current snr
        snr_snn_tmp_cur = snr_snn_tmp_bef;
    else:
        # data packet
        # update time
        time.append(test_data[i, 2]);
        # update minstrel mcs
        test_mcs_minstrel.append(test_data[i, -2]);
        # update SNR
        snr.append(10*np.log(test_data[i, 3]));
        # update SNR-SNN
        snr_snn_tmp_bef = 10*np.log(test_data[i, 3]);
        snr_snn.append(snr_snn_tmp_cur);
        # update SNN mcs
        mcs = rms.predict(test_data[i, 3]);
        test_mcs_snn_bef = mcs;
        test_mcs_snn.append(test_mcs_snn_cur);
        test_mcs_snn_perfect.append(mcs);
print('\r100%                                 ');

# draw
# draw - SNR
plt.figure(1, figsize=(15, 6), dpi=80);
plt.semilogy(time, snr, label='SNR(actual)');
plt.semilogy(time, snr_snn, label='SNR(SNN)');
plt.xlabel('time');
plt.ylabel('snr');
plt.title("SNR Change");
plt.legend(loc='best')  
plt.grid();
plt.show();
# draw - MCS
plt.figure(2, figsize=(15, 6), dpi=1200);
plt.semilogy(time, test_mcs_minstrel, 'bs-.', label='Minstrel');
plt.semilogy(time, test_mcs_snn, 'r-', label='SNN');
plt.semilogy(time, test_mcs_snn_perfect, 'c--', label='SNN (ideal)');
#plt.xlim([min(time), max(time)]);
for beacon in beacon_time:
    plt.axvline(x=beacon, color='y')
plt.xlabel('time')
plt.ylabel('MCS')
plt.title('MCS Change');
plt.legend(loc='best')  
plt.grid();
plt.show();
# draw - SNR vs MCS perfect
fig3 = plt.figure(3, figsize=(15, 6), dpi=1200);
fig3ax1 = fig3.add_subplot(311);
fig3ax1.semilogy(time, snr, 'r--', label='SNR');
for beacon in beacon_time:
    plt.axvline(x=beacon, color='y')
fig3ax1.set_xlabel('time')
fig3ax1.set_ylabel('SNR(dB)')
fig3ax1.set_title('SNN based on SNR');
fig3ax2 = fig3ax1.twinx();
fig3ax2.semilogy(time, test_mcs_snn, 'go-.', label='SNN');
fig3ax2.set_ylabel('MCS')
plt.legend(loc='best')  
plt.grid();
plt.show();