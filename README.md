
# Rate Adaption - Minstrel - SNN
* Dependency
    * `python 3.9`
    * `pandas`
    ```sh
    pip install pandas
    ```
    * [tensorflow](https://www.tensorflow.org/install/pip#windows-wsl2_1)
    ```sh
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install tensorflow
    ```
    test
    ```sh
    python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

## MCS (802.11ah), NSS = 1, Guard Time(GI) = 8us

| NS3 String              | Mod     | code rate           | data rate (Kbps) | MCS Index | Bandwidth | LeTian MCS | Vincent MCS |
|-------------------------|---------|---------------------|------------------|-----------|-----------|------------|-------------|
| OfdmRate300KbpsBW1MHz   | BPSK    | 1/2                 | 300              | 0         | 1MHz      | MCS1_0     | 0           |
| OfdmRate600KbpsBW1MHz   | QPSK    | 1/2                 | 600              | 1         | 1MHz      | MCS1_1     | 1           |
| OfdmRate900KbpsBW1MHz   | QPSK    | 3/4                 | 900              | 2         | 1MHz      | MCS1_2     | 2           |
| OfdmRate1_2MbpsBW1MHz   | 16-QAM  | 1/2                 | 1200             | 3         | 1MHz      | MCS1_3     | 3           |
| OfdmRate1_8MbpsBW1MHz   | 16-QAM  | 3/4                 | 1800             | 4         | 1MHz      | MCS1_4     | 4           |
| OfdmRate2_4MbpsBW1MHz   | 64-QAM  | 2/3                 | 2400             | 5         | 1MHz      | MCS1_5     | 5           |
| OfdmRate2_7MbpsBW1MHz   | 64-QAM  | 3/4                 | 2700             | 6         | 1MHz      | MCS1_6     | 6           |
| OfdmRate3MbpsBW1MHz     | 64-QAM  | 5/6                 | 3000             | 7         | 1MHz      | MCS1_7     | 7           |
| OfdmRate3_6MbpsBW1MHz   | 256-QAM | 3/4                 | 3600             | 8         | 1MHz      | MCS1_8     | 8           |
| OfdmRate4MbpsBW1MHz     | 256-QAM | 5/6                 | 4000             | 9         | 1MHz      | MCS1_9     | 9           |
| OfdmRate150KbpsBW1MHz   | 256-QAM | 1/2 (2x repetition) | 150              | 10        | 1MHz      | MCS1_10    |             |
| OfdmRate650KbpsBW2MHz   | BPSK    | 1/2                 | 650              | 0         | 2MHz      | MCS2_0     | 10          |
| OfdmRate1_3MbpsBW2MHz   | QPSK    | 1/2                 | 1300             | 1         | 2MHz      | MCS2_1     | 11          |
| OfdmRate1_95MbpsBW2MHz  | QPSK    | 3/4                 | 1950             | 2         | 2MHz      | MCS2_2     | 12          |
| OfdmRate2_6MbpsBW2MHz   | 16-QAM  | 1/2                 | 2600             | 3         | 2MHz      | MCS2_3     | 13          |
| OfdmRate3_9MbpsBW2MHz   | 16-QAM  | 3/4                 | 3900             | 4         | 2MHz      | MCS2_4     | 14          |
| OfdmRate5_2MbpsBW2MHz   | 64-QAM  | 2/3                 | 5200             | 5         | 2MHz      | MCS2_5     | 15          |
| OfdmRate5_85MbpsBW2MHz  | 64-QAM  | 3/4                 | 5850             | 6         | 2MHz      | MCS2_6     | 16          |
| OfdmRate6_5MbpsBW2MHz   | 64-QAM  | 5/6                 | 6500             | 7         | 2MHz      | MCS2_7     | 17          |
| OfdmRate7_8MbpsBW2MHz   | 256-QAM | 3/4                 | 7800             | 8         | 2MHz      | MCS2_8     | 18          |
| not supported           |         |                     |                  | 9         | 2MHz      |            |             |
| not supported           |         |                     |                  | 10        | 2MHz      |            |             |
| OfdmRate1_35MbpsBW4MHz  | BPSK    | 1/2                 | 1350             | 0         | 4MHz      |            | 19          |
| OfdmRate2_7MbpsBW4MHz   | QPSK    | 1/2                 | 2700             | 1         | 4MHz      |            | 20          |
| OfdmRate4_05MbpsBW4MHz  | QPSK    | 3/4                 | 4050             | 2         | 4MHz      |            | 21          |
| OfdmRate5_4MbpsBW4MHz   | 16-QAM  | 1/2                 | 5400             | 3         | 4MHz      |            | 22          |
| OfdmRate8_1MbpsBW4MHz   | 16-QAM  | 3/4                 | 8100             | 4         | 4MHz      |            | 23          |
| OfdmRate10_8MbpsBW4MHz  | 64-QAM  | 2/3                 | 10800            | 5         | 4MHz      |            | 24          |
| OfdmRate12_15MbpsBW4MHz | 64-QAM  | 3/4                 | 12150            | 6         | 4MHz      |            | 25          |
| OfdmRate13_5bpsBW4MHz   | 64-QAM  | 5/6                 | 13500            | 7         | 4MHz      |            | 26          |
| OfdmRate16_2MbpsBW4MHz  | 256-QAM | 3/4                 | 16200            | 8         | 4MHz      |            | 27          |
| OfdmRate18MbpsBW4MHz    | 256-QAM | 5/6                 | 18000            | 9         | 4MHz      |            | 28          |
| not supported           |         |                     |                  | 10        | 4MHz      |            |             |
