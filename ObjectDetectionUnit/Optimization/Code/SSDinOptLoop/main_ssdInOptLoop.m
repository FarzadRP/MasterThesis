%% Start
clc
clear all
close all

% Start timer
tic; t1 = clock;

%% Fitness function
fitnessfcn = @ssdInOptLoop;

%% Initioalization
nvars = 2; lb = [0 0]; ub = [1 5]; A = []; b = []; Aeq = []; beq = [];


%% NSGA-II
options= gaoptimset('ParetoFraction', 0.5, 'PopulationSize', 50, 'Generations', 100, 'StallGenLimit', 100, 'TolFun', 1e-100, ...
    'PlotFcns', {@gaplotpareto, @gaplotmaxconstr}, 'Display', 'diagnose', 'InitialPopulation', []); 

rng default
[x, fval, exitflag, output, population, scores]=gamultiobj(fitnessfcn, nvars, A, b, Aeq, beq, lb, ub, options); 

%% Save results
save('Results', 'fval')
