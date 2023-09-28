n = 900;                 % Set the size of square matrix A to be 900x900
e = ones(n,1);           % Create a column vector 'e' with n elements, all set to 1
A = spdiags([e 2*e e],-1:1,n,n); % Construct a sparse tridiagonal matrix A, with the main diagonal set to 2*e, and the upper and lower diagonals set to e
b = sum(A,2);            % Compute the sum of elements in each row of matrix A, storing the result in vector b

maxit = 200;             % Set the maximum number of iterations for the GMRES method to 200
x1 = gmres(A,b,[],[],maxit);  % Call GMRES method to solve the linear system Ax=b with no preconditioner and a default initial guess of the zero vector

x0 = 0.99*e;             % Create an initial guess x0, a column vector with n elements all set to 0.99
x2 = gmres(A,b,[],[],maxit,[],[],x0); % Call GMRES method to solve Ax=b using x0 as the initial guess and no preconditioner
% with initial guess, converge quickly at 7 iteration default 1e-6