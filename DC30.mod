# This is the optimized annual version with three new state of curtailment
# (N,LR,M)
# (8760, 14, 4)
reset;

param N = 8760;
set T = 1..N;
set Tminus = 1..N-1;

param LR = 15;
set L = 1..LR;
set Lminus = 1..LR-1;
set Lplus = 2..LR;
set Lclean = 2..LR-1;

param M = 4;
set I = 1..M;

set A = {-1, 0, 1};  # Actions of the Markov Decision Process
set E={0,1}; # Curtailment or not

param DD{T, L, I};
param D{T, L, I};
data DC30part1.dat;
let {i in T, j in L, z in I} DD[i,j,z] := DD[i,1,z];
let {i in T, j in L, z in I} D[i,j,z] := DD[i,j,z]*(1/1000);
param Q{T, L, I};

############################################################################################################################################
param variable_cost_ramping = 0.1;
param variable_cost_demand_response = 3;
param ramp_rate = 1.5; 
let {i in T,j in L,z in I} D[i,j,z] := D[i,1,z];
for{i in T, j in L, z in I} let  Q[i,j,z] := (j-1+4)*ramp_rate;
############################################################################################################################################
param extra{T, L, I} >= 0;  # Define extra
param E_value{T, L, I} binary;  # Binary parameter for E
param extraA{i in T, j in L, z in I, e in E,ep in E,epp in E, m in A}=extra[i,j,z];

# Calculate extra and determine the value of E for each state
let {t in T, l in L, i in I} extra[t, l, i] := if (D[t, l, i] - Q[t, l, i]) < 0 then 0 else D[t, l, i] - Q[t, l, i];
let {t in T, l in L, i in I} E_value[t, l, i] := if extra[t, l, i] > 0 then 1 else 0;
param E_valueA{T, L, I,E,E,E,A} binary;
let {t in T, l in L, i in I, e in E,ep in E,epp in E, a in A} E_valueA[t, l, i,e,ep,epp,a]:= if extraA[t, l, i,e,ep,epp,a] > 0 then 1 else 0;
############################################################################################################################################
set SS := {I,E,E,E};

param P{T, L, SS, SS, A} default 0;  # Adjusted transition probabilities
var x{T, L, SS, A} >= 0;  # Adjusted decision variable
data DC30part2.dat;
param c{T, L, SS, A};  # Adjusted cost parameter
 # GW
############################################################################################################################################
#Assignment for P 
for {i in T, j in L, z in I, zz in I, e in E, ep in E, epp in E, e2 in E, m in A} {
    # handle edge cases
    if ((j = 1 and m = -1) or (j = LR and m = 1)) then
        continue;  # Skip this iteration to avoid unnecessary zero assignments

    if (j+m >= 1 and j+m <= LR) then
        
        if not ((i < N and e2 = E_value[i+1, j+m, zz]) or (i = N and e2 = E_value[1, j+m, zz])) then
            continue;  # Skip this iteration to avoid unnecessary zero assignments
        else
            let P[i,j, z,e,ep,epp, zz,e2,e,ep, m] := P[i,1, z,0,0,0, zz,0,0,0, 0];  
}

############################################################################################################################################
let {i in T, j in L, (z,e,ep,epp) in SS , m in A} 
c[i,j,z,e,ep,epp, m] := Q[i,j,z]*variable_cost_ramping + (extra[i,j,z])*variable_cost_demand_response;
############################################################################################################################################

# Objective function
minimize Total_Cost: sum{t in T, l in L,(z,e,ep,epp) in SS, a in A} c[t,l,z,e,ep,epp, a] * x[t,l,z,e,ep,epp, a];

# Constraints
subject to Constraint1 {t in T}: 
    sum{l in L,(z,e,ep,epp) in SS} sum{a in A} x[t, l,z,e,ep,epp, a] = 1;


subject to Constraint2 {t in Tminus, l in Lclean,(z,e,ep,epp) in SS}: 
    sum{a in A} x[t+1, l,z,e,ep,epp, a]
    - sum{z2 in I, e2 in E,ep2 in E, epp2 in E} 
     (P[t, l-0,z2,e2,ep2,epp2,z,e,ep,epp, 0] * x[t, l-0,z2,e2,ep2,epp2, 0]
    + P[t, l-1,z2,e2,ep2,epp2,z,e,ep,epp, 1] * x[t, l-1,z2,e2,ep2,epp2, 1] 
    + P[t, l+1,z2,e2,ep2,epp2,z,e,ep,epp, -1] * x[t, l+1,z2,e2,ep2,epp2, -1]) = 0;

subject to Constraint3 {t in Tminus, l in {1},(z,e,ep,epp) in SS}: 
    sum{a in A} x[t+1, l,z,e,ep,epp, a]
    - sum{z2 in I, e2 in E,ep2 in E, epp2 in E} 
     (P[t, l-0,z2,e2,ep2,epp2,z,e,ep,epp, 0] * x[t, l-0,z2,e2,ep2,epp2, 0]
	+ P[t, l+1,z2,e2,ep2,epp2,z,e,ep,epp, -1] * x[t, l+1,z2,e2,ep2,epp2, -1])=0;

subject to Constraint4 {t in Tminus, l in {LR},(z,e,ep,epp) in SS}: 
    sum{a in A} x[t+1, l,z,e,ep,epp, a]
    - sum{z2 in I, e2 in E,ep2 in E, epp2 in E} 
     (P[t, l-0,z2,e2,ep2,epp2,z,e,ep,epp, 0] * x[t, l-0,z2,e2,ep2,epp2, 0]
	+ P[t, l-1,z2,e2,ep2,epp2,z,e,ep,epp, 1] * x[t, l-1,z2,e2,ep2,epp2, 1]) = 0;



subject to Constraint5 {t in {N}, l in Lclean,(z,e,ep,epp) in SS}: 
    sum{a in A} x[1, l,z,e,ep,epp, a]
    - sum{z2 in I, e2 in E,ep2 in E, epp2 in E} 
     (P[t, l-0,z2,e2,ep2,epp2,z,e,ep,epp, 0] * x[t, l-0,z2,e2,ep2,epp2, 0]
    + P[t, l-1,z2,e2,ep2,epp2,z,e,ep,epp, 1] * x[t, l-1,z2,e2,ep2,epp2, 1] 
    + P[t, l+1,z2,e2,ep2,epp2,z,e,ep,epp, -1] * x[t, l+1,z2,e2,ep2,epp2, -1]) = 0;

subject to Constraint6 {t in {N}, l in {1},(z,e,ep,epp) in SS}: 
    sum{a in A} x[1, l,z,e,ep,epp, a]
    - sum{z2 in I, e2 in E,ep2 in E, epp2 in E} 
     (P[t, l-0,z2,e2,ep2,epp2,z,e,ep,epp, 0] * x[t, l-0,z2,e2,ep2,epp2, 0]
	+ P[t, l+1,z2,e2,ep2,epp2,z,e,ep,epp, -1] * x[t, l+1,z2,e2,ep2,epp2, -1])=0;

subject to Constraint7 {t in {N}, l in {LR},(z,e,ep,epp) in SS}: 
    sum{a in A} x[1, l,z,e,ep,epp, a]
    - sum{z2 in I, e2 in E,ep2 in E, epp2 in E} 
     (P[t, l-0,z2,e2,ep2,epp2,z,e,ep,epp, 0] * x[t, l-0,z2,e2,ep2,epp2, 0]
	+ P[t, l-1,z2,e2,ep2,epp2,z,e,ep,epp, 1] * x[t, l-1,z2,e2,ep2,epp2, 1]) = 0;
	

subject to Constraint8: sum{t in T, l in L, z in I, e in {1}, ep in E ,epp in E, a in A} x[t, l,z,e,ep,epp, a]=0;

############################################################################################################################################
#option cplex_options 'primal';
#option cplex_options 'dual';
#option cplex_options 'baropt'; #Barrier Algorithm (Interior Point Method)
#Useful for large-scale LPs and QPs.
#Often more efficient for problems with a large number of constraints relative to the number of variables.
#Can be faster for dense problems and those with poor initial feasible solutions.
#Can be used with or without crossover to a simplex algorithm for final solution polishing.

#option cplex_options 'concurrentopt';
#option cplex_options 'feastol=1e-6 opttol=1e-6';

#option cplex_options 'outlev=1';
#option cplex_options 'display=2';
############################################################################################################################################
option solver gurobi;
#option gurobi_options 'presolve=-1 presparsify=-1 method=2 crossover=0 crossoverbasis=0';
option gurobi_options 'presolve=-1 presparsify=-1';
#option gurobi_options '';  # Clears all previously set options for Gurobi

# include DC30.mod
/*
option solver kestrel;
option kestrel_options 'solver=cplex';
option neos_server 'neos-server.org:3333';
option cplex_options 'primal mipgap=1e-5 threads=1';
option neos_username 'akhojaste@umass.edu';
option neos_user_password 'Westham1400@';
option email "akhojaste@umass.edu";
*/
############################################################################################################################################

solve;

#display {i in T, l in L, z in I, m in A }x[i, l,z,1,1,1, m], Total_Cost;
#expand Constraint7;
display x,E_valueA,_nvars,_ncons,_solve_time,_total_solve_time,Total_Cost> DC30r1NoOptions15.txt;

