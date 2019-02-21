using Revise
using ursa_functions
using Plots

ursa_0=URSA(0.95,3,0,2,0.8,10,100)  #parameter of URSA, see definition in ursa_functions
ursa_0.emit_prob_1=[0.4385 0.5615;0.5379 0.4621; 0.6860 0.3140] #opponent reaction probability

pbvi_solver=PBVI_solver(100,ursa_0)  #100 is the number of alpha vectors
solve!(pbvi_solver) #normal PBVI

N_sim=10000
a_v=zeros(Float64,N_sim)
for k=1:Int64(N_sim/2)
a_v[k]=simulate_true_reward_ursa(0.5,pbvi_solver,"non robust", "random", pbvi_solver.ursa)     #0.5 is the initial belief
end

pbvi_solver_rb=PBVI_solver(100,ursa_0)
robust_solve!(pbvi_solver_rb)  #robust PBVI

a_v=zeros(Float64,N_sim)
for k=1:Int64(N_sim/2)
a_v[k]=simulate_true_reward_ursa(0.5,pbvi_solver_rb,"robust", "norminal", pbvi_solver.ursa)
end

pbvi_solver_ch=PBVI_solver(100,ursa_0)
chance_solve!(pbvi_solver_ch,0.05,0.001)  #chance-constraint, 0.05 means 95% confidence

a_v=zeros(Float64,N_sim)
for k=1:Int64(N_sim/2)
a_v[k]=simulate_true_reward_ursa(0.5,pbvi_solver_ch,"chance", "norminal", pbvi_solver.ursa)
end




