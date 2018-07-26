function [ W1, W2 ] = fenner_MLP_BP_train1( data, class, J, L, rh, ro, epoch)
%Assume “data” is an nxm matrix whose columns are the input feature vectors/patterns.
%Assume “class” is a vector of associated class labels (Classes: 1, 2, 3, …, c).
%The learning rates for the hidden and output layers are rh and ro
%”epoch” is the desired number of full training cycles over the data set.
%is Jx(n+1) dimensional and holds the hidden-layer weights
%W2 is Lx(J+1) and holds the output-layer weights
%For a 3-class problem (c=3)
%Benjamin T. Fenner, FE3471
%7/10/2018, ECE 5995

%normalize data by dividing by largest feature.

%Beta equals one for hyperbolic tangent activation func.

B=1;

[n,m]=size(data);
 
%intitalize the wieghts for both hidden and output layers.
cW1=-1+2*rand(J,(n+1));
cW2=-1+2*rand(L,(J+1));

for cycle=1:1:epoch

    for colData=1:1:m

        %formats data for easy calulations.
        tempData=ones(n+1,1);
        for i=1:1:n
            tempData(i,1)=data(i,colData);   
        end

        %This holds all the sums of inputs filtered through tanh().
        %Initialize.
        Zmatrix=tanh(cW1*tempData);

        tempZ=ones(J+1,1);
        for row=1:1:J
            tempZ(row,1)=Zmatrix(row,1);  
        end

        %Gets the outputs.
        Ymatrix=tanh(cW2*tempZ);

        %Creates the d matrix for the outpur compare.
        %changing it from n to L
        d=-ones(L,1);
        for i=1:1:L
            if i==class(colData,1)
                d(i,1)=-d(i,1);
            end
        end

        %Output learning
        for col=1:1:(J+1)
            for row=1:1:L
                cW2(row,col)=cW2(row,col)+ro*(d(row,1)-Ymatrix(row,1))*tempZ(col,1);
            end
        end

        %Hidden Learning
        %sum((d-y).*cW2(:,J))

         for row=1:1:J
             for col=1:1:(n+1)
                 cW1(row,col)=cW1(row,col)+rh*(sum((d-Ymatrix).*cW2(:,row)))*(1-(tempZ(row,1).^2))*tempData(col,1);
             end
         end

    end
end

W1=cW1;
W2=cW2;

end