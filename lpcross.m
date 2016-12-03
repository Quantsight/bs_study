function [crosslpout lpout w]=lpcross(x,y,dates,TrainMonthSpan,skip,alpha,wght)
% function [crosslpout lpout w]=lpcross(x,y,dates,TrainMonthSpan,skip,alpha,wght)
% train span is number of months to train over, skip is 'month' or 'day'
% alpha is Tikhonov regularization factor
% w is weighting function

        PrintIntervalDates=0;

        if size(x,1)~=size(y,1) || size(x,1)~=size(dates,1)
                error('matrix len mismatch');
        end
        if nargin<3
                error('must spec dates');
        end
        if nargin<4
                error('must spec span');
        end
        if nargin<5
                skip='month';
        end
        if nargin<6
                alpha=0;
        end
        if nargin<7
                wght=[];
        end

        dv=datevec(min(dates));
        LoopTestStartDate=datenum([dv(1) dv(2) 1 0 0 0]);
        MinDates=LoopTestStartDate;

        dv=datevec(addtodate(max(dates),1,'day'));
        TestStopDate=datenum([dv(1) dv(2) dv(3) 0 0 0]);

        crosslpout=nan(size(y));
        while LoopTestStartDate<TestStopDate

            LoopTestStopDate=min(TestStopDate,addtodate(LoopTestStartDate,1,skip));

            LoopTrainStartDate=max(MinDates,addtodate(LoopTestStartDate,-TrainMonthSpan,'month'));
            % training period stop  date is TrainMonthSpan after trading StartDate but no later than TrainStopDate
            if addtodate(LoopTrainStartDate,TrainMonthSpan,'month')>LoopTestStartDate
                 LoopTrainStopDate=min(max(dates),addtodate(addtodate(LoopTrainStartDate,TrainMonthSpan,'month'),1,skip));
            else
                 LoopTrainStopDate=min(max(dates),addtodate(LoopTrainStartDate,TrainMonthSpan,'month'));
            end

            itrain=find(dates>=LoopTrainStartDate & dates<LoopTrainStopDate);
            itest =find(dates>=LoopTestStartDate  & dates<LoopTestStopDate );
            itrain=setdiff(itrain,itest);           % VERY IMPORTANT: exclude test samples from training set

            [myz w]=lp(x,y,itrain,alpha,wght);      % use ONLY TRAINING SAMPLES to find best fit coeefficients w

            if PrintIntervalDates
                fprintf('%s %s %s %s ',datestr(LoopTrainStartDate),datestr(LoopTrainStopDate),datestr(LoopTestStartDate),datestr(LoopTestStopDate));
                fprintf('%12.6f ',w);newl;
            end

            crosslpout(itest)=myz(itest);           % set output ONLY IN TEST SAMPLES

            LoopTestStartDate=min(TestStopDate,addtodate(LoopTestStartDate,1,skip));

        end
        [lpout]=lp(x,y,1:size(x,1),alpha,wght);     % this is a test output which is the the best fit over the entire interval, do not use for simulation
end
