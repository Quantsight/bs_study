% what is 'dates'? 6-element vector?
% what is bx/sx ?
%
%
%
%


bslpcross.m

function [buyout selout w]=bslpcross(x,bx,sx,y,dates,TrainMonthSpan,skip,alpha,wght)
% function [buyout selout w]=bslpcross(x,bx,sx,y,dates,TrainMonthSpan,skip,alpha,wght)
% assumes bx/sx replace the last columns of x for making buyout and selout
% train span is number of months to train over, skip is 'month' 'week' or 'day'
% alpha is Tikhonov regularization factor
% w is weighting function
        PrintIntervalDates=0;
        if size(x,1)~=size(y,1) || size(x,1)~=size(dates,1)
                error('matrix len mismatch');
        end
        if nargin<5
                error('must spec dates');
        end
        if nargin<6
                error('must spec span');
        end
        if nargin<7
                skip='month';
        end
        if nargin<8
                alpha=0;
        end
        if nargin<9
                wght=[];
        end
        dv=datevec(min(dates));
        if strcmp(skip,'week')
            LoopTestStartDate=min(dates)-weekday(min(dates))+2; % start on Monday
        else
            LoopTestStartDate=datenum([dv(1) dv(2) 1 0 0 0]);
        end
        MinDates=LoopTestStartDate;
        dv=datevec(addtodate(max(dates),1,'day'));
        TestStopDate=datenum([dv(1) dv(2) dv(3) 0 0 0]);
        buyout=nan(size(y));
        selout=nan(size(y));
        while LoopTestStartDate<TestStopDate
            if strcmp(skip,'week')
                LoopTestStopDate=min(TestStopDate,addtodate(LoopTestStartDate,7,'day'));
                LoopTrainStartDate=max(MinDates,addtodate(LoopTestStartDate,-TrainMonthSpan*28,'day')); % if using week skips, interpret a month as 4 weeks
                if addtodate(LoopTrainStartDate,TrainMonthSpan*28,'day')>LoopTestStartDate
                    LoopTrainStopDate=min(max(dates),addtodate(addtodate(LoopTrainStartDate,TrainMonthSpan*28,'day'),7,'day'));
                else
                    LoopTrainStopDate=min(max(dates),addtodate(LoopTrainStartDate,TrainMonthSpan*28,'day'));
                end
            else
                LoopTestStopDate=min(TestStopDate,addtodate(LoopTestStartDate,1,skip));
                LoopTrainStartDate=max(MinDates,addtodate(LoopTestStartDate,-TrainMonthSpan,'month'));
                % training period stop  date is TrainMonthSpan after trading StartDate but no later than TrainStopDate
                if addtodate(LoopTrainStartDate,TrainMonthSpan,'month')>LoopTestStartDate
                    LoopTrainStopDate=min(max(dates),addtodate(addtodate(LoopTrainStartDate,TrainMonthSpan,'month'),1,skip));
                else
                    % prevent LoopTrainStopDate from running off end of data:
                    LoopTrainStopDate=min(max(dates),addtodate(LoopTrainStartDate,TrainMonthSpan,'month'));
                end
            end

            % Indices of elements from <dates> vector that meet "train" and "test" criteria:
            itrain=find(dates>=LoopTrainStartDate & dates<LoopTrainStopDate);
            itest =find(dates>=LoopTestStartDate  & dates<LoopTestStopDate );

            % make sure (again!) to exclude test from train:
            itrain=setdiff(itrain,itest);   % VERY IMPORTANT: exclude test samples from training set

            % Create NaN vector with row for each column in x
            % The '1' is number of resulting columns (vectors):
            w=nan(size(x,2),1);

            if count(x(itrain)~=0)>size(x,2)
                % no intercept
                [myz w]=lp(x,y,itrain,alpha,wght);  % use ONLY TRAINING SAMPLES to find best fit coefficients w
            else
                fprintf('warning: insufficient data points to train: ');
                fprintf('%s %s %s %s ',datestr(LoopTrainStartDate),datestr(LoopTrainStopDate),datestr(LoopTestStartDate),datestr(LoopTestStopDate));
                newl;
            end

            if PrintIntervalDates
                fprintf('%s %s %s %s ',datestr(LoopTrainStartDate),datestr(LoopTrainStopDate),datestr(LoopTestStartDate),datestr(LoopTestStopDate));
                %fprintf('%12.6f ',w);
                newl;
            end

            r1=size(x,2)-size(bx,2);                                        % r1 is the column number of last common input
            buyout(itest)=x(itest,1:r1)*w(1:r1)+bx(itest,:)*w(r1+1:end);    % buy predictor uses common inputs and buy  inputs - set output ONLY IN TEST SAMPLES
            selout(itest)=x(itest,1:r1)*w(1:r1)+sx(itest,:)*w(r1+1:end);    % sel predictor uses common inputs and sell inputs - set output ONLY IN TEST SAMPLES
            if strcmp(skip,'week')
                LoopTestStartDate=min(TestStopDate,addtodate(LoopTestStartDate,7,'day'));
            else
                LoopTestStartDate=min(TestStopDate,addtodate(LoopTestStartDate,1,skip));
            end
        end
end