rng default                   % Set the random number generator to its default settings for reproducibility
A = sprandn(400,400,0.5) + 12*speye(400); % Create a 400x400 sparse random matrix A with normally distributed entries and 50% sparsity, then add 12 to its diagonal
b = rand(400,1);              % Create a 400x1 random vector b

x = gmres(A,b);               % Solve the linear system Ax = b using GMRES with default parameters (tolerance, max iterations)

tol = 1e-4;                   % Set the tolerance for the GMRES method to 1e-4
maxit = 100;                  % Set the maximum number of iterations for GMRES to 100
x = gmres(A,b,[],tol,maxit);  % Call GMRES to solve the linear system Ax=b with specified tolerance and maximum number of iterations

restart = 100;                % Set the restart parameter for GMRES to 100
maxit = 20;                   % Set the maximum number of iterations for GMRES to 20 (this is for outer iterations)
x = gmres(A,b,restart,tol,maxit); % Call GMRES to solve Ax=b with the specified restart parameter, tolerance, and maximum number of iterations
