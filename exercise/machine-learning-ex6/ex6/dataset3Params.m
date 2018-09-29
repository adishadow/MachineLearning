function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
pred = svmPredict(model,Xval);
error_min = mean(double(pred~=yval));
C_min =C;
sigma_min = sigma;
sigma_temp = sigma;
C_temp = C;
for i=1:10;
	C_temp = C_temp*.3;
	sigma_temp = sigma;
	for j=1:10;
		sigma_temp = sigma_temp *.3;
		model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
		pred = svmPredict(model,Xval);
		error = mean(double(pred~=yval));
		if(error<error_min),
			error_min = error;
			C_min =C_temp;
			sigma_min =sigma_temp;
	end;
end;
C = C_min;
sigma= sigma_min;
	





% =========================================================================

end
