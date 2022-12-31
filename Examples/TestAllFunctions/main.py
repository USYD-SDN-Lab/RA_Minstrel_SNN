import sys 
sys.path.append("../..");
import os
from RA_Minstrel_SNN import RA_Minstrel_SNN;

# set prefix to store models and 
model_prefix = "_build_/models_TestAllFunctions/";
if not os.path.exists(model_prefix):
    os.makedirs(model_prefix);

rms = RA_Minstrel_SNN(snn_model_prefix = model_prefix);
