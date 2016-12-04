% dv(:,1) = Year
% dv(:,2) = Month
% dv(:,3) = Day
% Time    = time of day as fraction of Day, i.e. .4 = 09:30AM EST
% SymCode = integer code for each unique symbol
% pi      = predictor inputs
%           last columne of pi is 1 for buy, -1 for sell
% xpi     = extra predictor inputs not currently used by linear predictor - Good Luck!
% Target  = Target used for training predictor (clamped value of RawTarget)
% RawTarget = Estimated gain or loss for that trade, assuming it was a buy.  For buys +gain is good, - is bad.  For sells, -gain is good.
% po      = predictor output using current methods

dv=datevec(Date(ibs));
ds=[dv(:,1) dv(:,2) dv(:,3) Time(ibs) SymCode(ibs) pi(ibs,1:21) xpi(ibs,1:10) Target(ibs) RawTarget(ibs) po(ibs)];

% Training Intervals:
% Training Dates always exclude Test Dates
% TrainStartDate TrainStopDate TestStartDate TestStopDate



% find max profit on buy side
sorted=sortrows([po(ib) RawTarget(ib)],-1);
cumsort=cumsum(sorted(:,2));
i=find(cumsort==max(cumsort),1);
fprintf('%12.2f ',sorted(i,1),cumsort(i));
newl;

% find max profit on the sell side
sorted=sortrows([po(is) RawTarget(is)],+1);
cumsort=-cumsum(sorted(:,2));
i=find(cumsort==max(cumsort),1);
fprintf('%12.2f ',sorted(i,1),cumsort(i));
newl;

% find max profit on buy and sell side
sorted=sortrows([[po(ib);-po(is)] [RawTarget(ib);-RawTarget(is)]],-1);
cumsort=cumsum(sorted(:,2));
i=find(cumsort==max(cumsort),1);
fprintf('%12.2f ',sorted(i,1),cumsort(i));
newl;

save('ds.txt','ds','-ascii');

