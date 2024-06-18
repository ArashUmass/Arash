# (N,LR,M)
# (8760, LR, 4)
reset;
option ampl_mem_size 63G;
##################################################################################################################
param N = 8760;
set T = 1..N;
set Tminus=1..N-1;

param LR = 14;
set L=1..LR;
set Lminus=1..LR-1;
set Lplus=2..LR;

param M = 4;
set I=1..M;
##################################################################################################################

set A;   /* The actions of the Markov Decision Process */

set S = {L,I}; /* Next state of the Markov Decision Process */
set K = {L,I}; /* The state of the Markov Decision Process */
# State: (Trading point, Ramping operating level, residual demand level)

param P{T,K,S,A} default 0; # Transition probabilities from state K to state S with action A

var x{T,S,A} >= 0; 
##################################################################################################################

param c{T,S,A} ; /* Cost of being in state S with action A */

##################################################################################################################

param fixed_cost_base;
let fixed_cost_base:= 400000/365/N;

#param fixed_cost_ramping= 170000;

param fixed_cost_peak;
let fixed_cost_peak:= 100000/365/N;

param fixed_cost_demand_response;
let fixed_cost_demand_response:= 0/365/N;
# fixed cost for demand response = 0 $/MW/year = (0/365) $/MW/day
# for example when N=8m ==> fixed_cost_demand_response:= 0 $/MW/3hours

param variable_cost_ramping= 5;
param variable_cost_demand_response= 3000;

param capacity_of_ramping;
param ramp_rate=2000;
##################################################################################################################
param D{T,S};
param Q{T,S};

data N15.dat;

let {i in T,(j,z) in S} D[i,j,z] := D[i,1,z];
for{i in T, j in L} let {(j,z) in S} Q[i,j,z] := (j-1)*ramp_rate;
#let {(i,j,z) in S} Q[i,j,z] := Q[i,j,1];

for{i in T} let {(j,z) in K, (jj,zz) in S} P[i,j,z,j,zz,0] := P[i,1,z,1,zz,0];
for{i in T, j in Lminus} let {(j,z) in K, (jj,zz) in S} P[i,j,z,j+1,zz,1] := P[i,1,z,1,zz,0];
for{i in T, j in Lplus} let {(j,z) in K, (jj,zz) in S} P[i,j,z,j-1,zz,-1] := P[i,1,z,1,zz,0];


param extra{i in T, (j,z) in S}=  if (D[i,j,z]-Q[i,j,z])<0 then 0 else D[i,j,z]-Q[i,j,z];

let {i in T, (j,z) in S , m in A} 
#c[i,j,z,m] := fixed_cost_ramping*LR + Q[i,j,z]*variable_cost_ramping + (extra[i,j,z])*variable_cost_demand_response;
c[i,j,z,m] := Q[i,j,z]*variable_cost_ramping + (extra[i,j,z])*variable_cost_demand_response;

##################################################################################################################
var U{L} binary;
param RU{L} default 2000;
param RD{L} default 2000;
param fixed_cost_ramping{L} default 500;
for{j in L} let fixed_cost_ramping[j] := (j-1)*50;
#param capacity_of_ramping{L} default 500;
##################################################################################################################
minimize Total_Cost: sum{i in T,(j,z) in S} sum{m in A} c[i,j,z,m]*x[i,j,z,m]+ sum{j in L} fixed_cost_ramping[j]*U[j];
# Objective function

# First constraint
subject to Constraint1 {i in T}: sum{(j,z) in S} sum{m in A} x[i,j,z,m]=1;

# Second constraint
# Global Balance Constraint
subject to Constraint2 {i in Tminus, (j,z) in S} : sum {m in A} x[i+1,j,z,m]-sum {(j2,z2) in K} sum {m in A} P[i,j2,z2,j,z,m]*x[i,j2,z2,m]=0;
subject to Constraint3 {i in {N}, (j,z) in S} : sum {m in A} x[1,j,z,m]-sum {(j2,z2) in K} sum {m in A} P[i,j2,z2,j,z,m]*x[i,j2,z2,m]=0;
subject to Constrainty2 {(j,z) in S , i in T, m in A} : x[i,j,z,m] <= U[j];
option solver cplex;
solve;
# Solving the model

##################################################################################################################
param pi{T,S};
# Stationary Distribution
let {i in T,(j,z) in S} pi[i,j,z] := sum {m in A} x[i,j,z,m];

var optimal_policy{i in T,(j,z) in S , m in A} = if pi[i,j,z] =0 then 0 else x[i,j,z,m]/pi[i,j,z];
# Determining Optimal Policy
#param Total_extra=sum{(i,j,z) in S} extra[i,j,z];

display Total_Cost, x,optimal_policy,U;
#expand Total_Cost,Constraint1, Constraint2;


