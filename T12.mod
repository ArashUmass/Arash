reset;
param HP = 2;
set H = 1..HP; /* operating level of peaking plant, 1, 2,..., HP */

##################################################################################################################

param N = 4;
set T = 1..N;
# Trading points: periods of day, 1, 2,..., N
# 24hours/N
# Eight 3-hour periods in each day

param LR = 3;
set L=1..3;
# operating level of ramping plant, 1, 2,..., LR
# ramping operating level, 1 => 0 MW, 2 => 100 MW, 3 => 200 MW

param M = 2;
set I;
# residual demand level, 1, 2,..., M
# demand after wind and base load
# residual demand level, 1 is low, 2 is average, 3 is high

##################################################################################################################

set A;   /* The actions of the Markov Decision Process */
# action of ramping up or down in the next state
# action ==> -1 is ramping down, 0 is no change in ramping level, +1 is ramping up

set S = {T,L,I}; /* Next state of the Markov Decision Process */
set K = {T,L,I}; /* The state of the Markov Decision Process */
# State: (Trading point, Ramping operating level, residual demand level)

param P{K,S,A} default 0; # Transition probabilities from state K to state S with action A

var x{S,A} >= 0; 
#var Q{L}>= 0;

##################################################################################################################

param c{S,A} ; /* Cost of being in state s with action a*/

##################################################################################################################

param fixed_cost_base;
let fixed_cost_base:= 400000/365/N;
# fixed cost for base = 400,000 $/MW/year = (400,000/365) $/MW/day
# for example when N=8m ==> fixed_cost_base:= 136.98 $/MW/3hours

param fixed_cost_ramping;
let fixed_cost_ramping:= 200000/365/N;
# fixed cost for ramping = 200,000 $/MW/year = (200,000/365) $/MW/day
# for example when N=8m ==> fixed_cost_ramping:= 68.49 $/MW/3hours

param fixed_cost_peak;
let fixed_cost_peak:= 100000/365/N;
# fixed cost for peaking = 100,000 $/MW/year = (100,000/365) $/MW/day
# for example when N=8m ==> fixed_cost_peak:= 32.25 $/MW/3hours

param fixed_cost_demand_response;
let fixed_cost_demand_response:= 0/365/N;
# fixed cost for demand response = 0 $/MW/year = (0/365) $/MW/day
# for example when N=8m ==> fixed_cost_demand_response:= 0 $/MW/3hours

param variable_cost_base;
param variable_cost_ramping;
param variable_cost_peak;
param variable_cost_demand_response;

param capacity_of_ramping;
param capacity_of_peak;
param capacity_of_demand_response;
##################################################################################################################
set S1 = {T,L};
set S2 = {T,I};
param D{S};

#param Q{L};
param Q{S};
# ramping operating level generation upper limit which can be 0 MW, 100 MW, or 200 MW
# ramping operating level 1 => UQ=0 MW, 2 => UQ=100 MW, UQ=3 => 200 MW
data T12.dat;

param extra{(i,j,z) in S}=  if (D[i,j,z]-Q[i,j,z])<0 then 0 else D[i,j,z]-Q[i,j,z];

let {(i,j,z) in S , m in A} 
c[i,j,z,m] := fixed_cost_ramping*capacity_of_ramping + Q[i,j,z]*variable_cost_ramping + (extra[i,j,z])*variable_cost_demand_response;

##################################################################################################################
# Risk Averse version

param l default 1;
param beta=0.85;
var v{S,A};
var w{S,A};
var eta;
##################################################################################################################

minimize Total_Cost:sum{(i,j,z) in S} sum{m in A} (c[i,j,z,m]*x[i,j,z,m]
+ x[i,j,z,m]*l*((1-beta)*w[i,j,z,m]+beta*v[i,j,z,m]));

# Objective function
subject to Risk_Constraint {(i,j,z) in S , m in A}: eta+v[i,j,z,m]-w[i,j,z,m]=c[i,j,z,m];

# First constraint
subject to Constraint1: sum{(i,j,z) in S} sum{m in A} x[i,j,z,m]=1;

# Second constraint
# Global Balance Constraint
subject to Constraint2 {(i,j,z) in S} : sum {m in A} x[i,j,z,m]-sum {(i2,j2,z2) in K} sum {m in A} P[i2,j2,z2,i,j,z,m]*x[i2,j2,z2,m]=0;

#option solver couenne;
option solver conopt;
solve;
# Solving the model

##################################################################################################################
param pi{S};
# Stationary Distribution
let {(i,j,z) in S} pi[i,j,z] := sum {m in A} x[i,j,z,m];

var optimal_policy{(i,j,z) in S , m in A} = if pi[i,j,z] =0 then 0 else x[i,j,z,m]/pi[i,j,z];
# Determining Optimal Policy
param Total_extra=sum{(i,j,z) in S} extra[i,j,z];

display x,Total_extra,v, Total_Cost,w,eta,c;

expand Total_Cost,Risk_Constraint,Constraint1, Constraint2;





