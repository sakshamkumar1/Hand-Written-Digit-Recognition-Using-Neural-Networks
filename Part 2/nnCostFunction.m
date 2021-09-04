function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
for i = 1:m
    Xd = X(i,:)';
    z2 = Theta1 * [1; Xd];
    a2 = sigmoid(z2);
    z3 = Theta2 * [1; a2];
    a3 = sigmoid(z3);
    for k = 1:num_labels
        yki = (k == y(i));
        J = J + (1/m) * ((-yki * log(a3(k))) - (1 - yki) * log(1 - a3(k)));
    end
end

firstTerm = 0;
[a, b] = size(Theta1);
for j = 1:a
    for k = 2:b
        firstTerm = firstTerm + (Theta1(j, k))^2;
    end
end
firstTerm = (firstTerm * lambda) / (2 * m);

secondTerm = 0;
[a, b] = size(Theta2);
for j = 1:a
    for k = 2:b
        secondTerm = secondTerm + (Theta2(j,k))^2;
    end
end
secondTerm = (secondTerm * lambda) / (2 * m);

J = J + firstTerm + secondTerm;

% % Now, implementing the backpropagation algorithm to compute gradients.

delta2 = zeros(size(a3, 1), size(a2, 1)+1);
delta1 = zeros(size(a2, 1), size(Theta1, 2));
for t = 1:m
    a1 = (X(t,:))';
    a1 = [1; a1];
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    ymat = zeros(size(a3));
    ymat(y(t)) = 1;
    del3 = a3 - ymat;
    ThetaTemp = Theta2';
    del2 = ThetaTemp(2:end, :) * del3 .* sigmoidGradient(z2);
    
    delta2 = delta2 + del3 * a2';
    delta1 = delta1 + del2 * a1';
    
end

Dij2 = (1 / m) * delta2;
Dij1 = (1 / m) * delta1;
 

    
    

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% Without regularization
% Theta1_grad = Dij1;
% Theta2_grad = Dij2;
% Now, let's calcualte for the case of regularization.
Theta1_grad = Dij1 + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = Dij2 + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
