load west0479              % Load the dataset named 'west0479'
A = west0479;             % Assign the loaded data to A, 479x479, Asymmetric, Real Sparse Matrix

b = sum(A,2);             % Compute the sum of elements in each row of matrix A, storing the result in vector b

tol = 1e-12;              % Set the tolerance for the GMRES method to 1e-12
maxit = 20;               % Set the maximum number of iterations for GMRES to 20    

% Call GMRES to solve the linear system Ax=b without preconditioner
% Solution, flag0, relative residual,[outer inner]iterations,history of residual 
[x,fl0,rr0,it0,rv0] = gmres(A,b,[],tol,maxit);
fl0, rr0, it0             % Output the flags (fl0), residual norm at last iteration (rr0), and iteration log (it0)

% Compute the ILU preconditioned matrices L and U with a drop tolerance of 1e-6
[L,U] = ilu(A,struct('type','ilutp','droptol',1e-6));
% Call GMRES to solve Ax=b using ILU preconditioner, and store the flags, residual norm, iteration log, and residuals
[x1,fl1,rr1,it1,rv1] = gmres(A,b,[],tol,maxit,L,U);
fl1, rr1, it1, rv1          % Output the flags (fl1), residual norm at last iteration (rr1), and iteration log (it1)

% Plot the relative residual for GMRES without preconditioner
semilogy(0:length(rv0)-1,rv0/norm(b),'-o')
hold on                   % Hold the current figure to add more plots to it
% Plot the relative residual for GMRES with ILU preconditioner
semilogy(0:length(rv1)-1,rv1/norm(U\(L\b)),'-o')
yline(tol,'r--');         % Add a red dashed line to represent the tolerance
% Add a legend to the plot
legend('No preconditioner','ILU preconditioner','Tolerance','Location','East')
xlabel('Iteration number') % Label the x-axis
ylabel('Relative residual') % Label the y-axis
