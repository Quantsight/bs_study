build_pred -> bslpcross -> write_data_set

============================== DATA GENERATION ==============================
write_data_set.m generates the data set.
Date ranges are:
TrainStartDate TrainStopDate TestStartDate TestStopDate
01-Jan-2016 01-May-2016 01-Jan-2016 01-Feb-2016
01-Jan-2016 01-May-2016 01-Feb-2016 01-Mar-2016
01-Jan-2016 01-May-2016 01-Mar-2016 01-Apr-2016
01-Jan-2016 01-Apr-2016 01-Apr-2016 01-May-2016
01-Feb-2016 01-May-2016 01-May-2016 01-Jun-2016
01-Mar-2016 01-Jun-2016 01-Jun-2016 01-Jul-2016
01-Apr-2016 01-Jul-2016 01-Jul-2016 01-Aug-2016
01-May-2016 01-Aug-2016 01-Aug-2016 01-Sep-2016
01-Jun-2016 01-Sep-2016 01-Sep-2016 01-Oct-2016
01-Jul-2016 01-Oct-2016 01-Oct-2016 29-Oct-2016
Max Profit Results are:
>> build_pred
>> write_data_set
        0.23   6846023.16
       -0.17   6709962.16
        0.21  13553076.93
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
Generate train/test sets.  train months will be "nearest" months that are preferably earlier, but could be later.
  - Use calendar months, so (for instance) 28 test days for "February" test or train sets, but other test or train months could have 30 or 31 days.

- I don't know what <ib> & <is> are in build_pred.m