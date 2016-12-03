
% ib is an index into the buy opportunities
% is is an index into the sell opportunities
ibs=sort([ib;is]);

% pi are all predictor inputs
% first columns are common inputs to buy and sell predictors (1-19)
% second group of columns are inputs for only the buy or the sell predictor (20)
% last column is +/-1 indicating buy or sell (21)


GroupFit=0; % linear fit over entire group of symbols, else stock by stock

BackOnly=1;         % predictor built from backwards looking data, mostly
TrainMonthSpan=3;   % span of training interval if BackOnly
Skip='month';       % skip = month, week, or day, as of 12/10/2014 daily updates does no good

po=zeros(size(Date));

if BackOnly
	if GroupFit
	    irange=0;       % irange set to zero will do all symbols simultaneously
	else
	    irange=1:NSym;  % irange will do one symbol at a time
	end
	for i=irange        % loop over all symbols (or not)
	    if i==0
		    jb=ib;                          % if doing all symbols, find all samples on buy side
		    js=is;                          % sell side
	    else
            jb=ib(find(SymCode(ib)==i));    % just find the i'th symbol on buy side
            js=is(find(SymCode(is)==i));    % sell side
	    end
	    jbs=[jb;js];                        % index to both buy and sell side

        po(jbs)=lpcross(pi(jbs,:),Target(jbs),Date(jbs),TrainMonthSpan,Skip);  % use all buy and sell samples to find coefficients
    end
end
