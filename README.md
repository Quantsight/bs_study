build_pred -> bslpcross -> write_data_set

============================== DATA GENERATION ==============================
write_data_set.m generates the data set.

Here are the results of running that script:
        0.22   5987058.57
       -0.17   5692807.48
        0.16  11677338.04
The key result is the last one, since I like to force symmetry between buy and sell.

In actual production trading, the threshold on the predictor is set much higher in order to be more discriminating because of buying power limitations.

============================== DATA SIZE ==============================
The total data set is:
>> size(ds)
ans =   11078373.00         30.00
In memory that takes: 2.7 GBytes.  Not too bad, however this is only the top
20 symbols.  There are 440 symbols total.
The compressed file size of the whole data set is 163 Mbytes.


============================== PLAN ==============================
reproduce current linear predictor
    understand data generation
    reproduce "Walk Forward" training framework
    understand current model
        performance distribution
        analysis of each predictor
    ? other linear algorithm variants ?

Polynomial (Linear) model
    analysis of each predictor

RandomForest
    analysis
    what are the typical major splits?
    "manual" subsegmentaion of linear models

Nearest Neighbors

SVM/KernelRidge/SGD/Gaussian Process Regression

Factorization Machine
    Will need (want) symbols for each row

Neural Network (Deep Learning...)