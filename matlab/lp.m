function [z w]=lp(x,y,r,alpha,wght)
% function [z w]=lp(x,y,r,alpha,wght)
% alpha is Tikhonov regularization factor
% wght is weighting function
        if size(x,1)~=size(y,1)
                error('matrix len mismatch');
        end
        if nargin<3
                r=1:size(x,1);
        end
        if nargin<4
                alpha=0;
        end
        if nargin<5
                wght=[];
        end
        x=double(x);
        stdx=std(x(r,:));
        stdx(find(stdx==0))=1;
        %normx=x/diag(stdx);
        normx=zeros(size(x));
        for i=1:size(x,2)
            normx(:,i)=x(:,i)/stdx(i);
        end
        y=double(y);
        %s=sqrt(mean(x(r,:).^2));
        %s(find(abs(s)<=1e-9))=1;

        if isinf(alpha)
                C=x(r,:)'*y(r,:)/(length(r)-1);
                wght=C;
        else
                wid=size(x,2);
                gamma=(diag(ones(1,wid)));
                gamma=gamma*alpha;
                gamma2=gamma'*gamma;
                if length(wght)>0
                        xw(r,:)=normx(r,:).*(wght(r)*ones(1,wid));
                        yw(r,:)=y(r,:).*wght(r);
                        R=xw(r,:)'*xw(r,:)/(length(r)-1);
                        C=xw(r,:)'*yw(r,:)/(length(r)-1);
                else
                        R=normx(r,:)'*normx(r,:)/(length(r)-1);
                        C=normx(r,:)'*y(r,:)/(length(r)-1);
                end
                w=(R+gamma2)\C;
                w=w./stdx';
                %fprintf('%9.6f ',diag(R));
                %newl;
                %save test x y r R C 
                %error('stop');
        end

        %w=lscov(double(x(r,:)),double(y(r,:)));
        z=x*w;
        %w'.*std(x)
end
