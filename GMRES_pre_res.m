load west0479                % Load the dataset named 'west0479'
A = west0479;               % Assign the loaded data to A

b = sum(A,2);               % Compute the sum of elements in each row of matrix A, storing the result in vector b

% Use ILUTP preconditioner with drop tolerance of 1e-6 for ILU decomposition to compute matrices L and U for matrix A
[L,U] = ilu(A,struct('type','ilutp','droptol',1e-6));

tol = 1e-12;                % Set the tolerance for the GMRES method to 1e-12
maxit = 20;                 % Set the maximum number of iterations for GMRES to 20
% Call GMRES to solve the linear system Ax=b, with different restart parameters (3, 4, 5), and using ILU preconditioner
[x3,fl3,rr3,it3,rv3] = gmres(A,b,3,tol,maxit,L,U);
[x4,fl4,rr4,it4,rv4] = gmres(A,b,4,tol,maxit,L,U);
[x5,fl5,rr5,it5,rv5] = gmres(A,b,5,tol,maxit,L,U);
fl3, it3, rr3, fl4, it4, rr4, fl5, it5, rr5            % Output the flags fl3, fl4, and fl5 for each solution
%3:5 3=13   4:2 4=8   5:2 4=9 total iterations
% Plot the relative residual in a semi-logarithmic graph to show the convergence of gmres(3), gmres(4), and gmres(5)
semilogy(1:1/3:6,rv3/norm(U\(L\b)),'-o');
h1 = gca;                   % Get handle of the current axes
h1.XTick = (1:6);           % Set the x-axis ticks
title('gmres(N) for N = 3, 4, 5') % Add title to the plot
xlabel('Outer iteration number'); % Label the x-axis
ylabel('Relative residual');      % Label the y-axis
hold on                     % Hold the current figure to add more plots to it
semilogy(1:1/4:3,rv4/norm(U\(L\b)),'-o'); % Add the convergence curve for gmres(4)
semilogy(1:1/5:2.8,rv5/norm(U\(L\b)),'-o'); % Add the convergence curve for gmres(5)
yline(tol,'r--');            % Add a red dashed line to represent the tolerance
hold off                    % Release the current figure
legend('gmres(3)','gmres(4)','gmres(5)','Tolerance') % Add legend
grid on                     % Enable grid lines
