__precompile__()
module ursa_functions

using Gallium
using Plots
using JuMP
using Clp
using Distributions

struct URSAState
    b::Real     #belief
    A::Int64    #allowed level of action
end

mutable struct URSA #<: MDP{URSAState, Int64}  #second parameter is the type of actions
    discount_factor::Float64
    Num_actions::Int64
    True_lambda::Int64
    Num_t_actions::Int64
    uncertain_factor::Float64
    N_steps::Int64
    N_iter::Int64
    URSA()=new()
    emit_prob_1::Array{Float64}#,Num_actions,Num_t_actions}
    emit_prob_0::Array{Float64}
    r_action_hostile::Array{Float64}
    r_action_civil::Array{Float64}
end

function URSA(df::Float64,Na::Int64,lam::Int64,Nta::Int64,u_f::Float64,n_step::Int64,n_iter::Int64)
    this=URSA()  #default discount factor, default three actions
    this.discount_factor=df
    this.Num_actions=Na
    this.True_lambda=lam
    this.Num_t_actions=Nta
    this.uncertain_factor=u_f
    this.N_steps=n_step
    this.N_iter=n_iter
    this.emit_prob_1=[0.4 0.6;0.45 0.55;0.7 0.3]
    this.emit_prob_0=[0.6 0.4;0.75 0.25;0.9 0.1]
    this.r_action_hostile=[0 0 0]
    this.r_action_civil=[-0.1 -0.3 -0.7]
    return this
end

mutable struct PBVI_solver
    N_B::Int64   #number of belief points
    ursa::URSA    #the ursa problem to be solved
    B::Array{Float64}   #set of belief point
    rho_alpha::Array{Array{Float64,1}}   #corresponding alpah vector for the belief dependent reward
    alpha::Array{Array{Float64,1}}
    Value::Array{Float64,1}
    opt_action::Array{Int64}
    p_opt::Array{Array{Float64,2}}
    PBVI_solver()=new()
end

function entropy_ursa(x::Float64)
    @assert (x>=0 && x<=1)
    if x==0 || x==1
        return 0
    else
        return x*log(x)+(1-x)*log(1-x)
    end
end

function PBVI_solver(N_B::Int64,ursa::URSA)
    this=PBVI_solver()
    this.N_B=N_B
    this.ursa=ursa
    db=0.998/(N_B-1)
    this.B=collect(0.001:db:0.999)
    slope=log.(this.B)-log.(1-this.B)
    entro=entropy_ursa.(this.B)
    #now solve for the rho_alpha vector
    this.rho_alpha=Array{Array{Float64,1}}(N_B)
    this.alpha=Array{Array{Float64,1}}(N_B)
    this.Value=Array{Float64,1}(N_B)
    this.p_opt=Array{Array{Float64,2}}(N_B,ursa.Num_actions)

    for k=1:N_B
        A=[-1 1;
        1-this.B[k] this.B[k]]
        b=[slope[k];entro[k]]
        this.rho_alpha[k]=A\b
        @assert abs(dot_alpha_b(this.rho_alpha[k],this.B[k])-entro[k])<10.0^(-6.0)
    end
    initialize_alpha!(this)
    return this
end
#return dot product between alpha and belief state b
dot_alpha_b(alpha::Array{Float64},b::Float64)=alpha[1]*(1-b)+alpha[2]*b

function initialize_alpha!(pbvi_solver::PBVI_solver)
    #pbvi_solver.alpha=Array{Array{Float64,1},1}(pbvi_solver.N_B)
    #pbvi_solver.Value=Array{Float64,1}(pbvi_solver.N_B)
    m_0=minimum(pbvi_solver.ursa.r_action_civil)+entropy_ursa(0.5)
    m_1=minimum(pbvi_solver.ursa.r_action_hostile)+entropy_ursa(0.5)
    for i=1:pbvi_solver.N_B
        pbvi_solver.alpha[i]=[m_0;m_1]/(1-pbvi_solver.ursa.discount_factor)
    end
    #=for i=1:pbvi_solver.N_B
        b=pbvi_solver.B[i]
        reward=-Inf
        max_action=0
        for j=1:pbvi_solver.ursa.Num_actions  #when we have escalation of action, here needs modification
            reward_j=get_reward(b,j,pbvi_solver)
            if reward_j>reward
                reward=reward_j
                max_action=j
            end
        end
        pbvi_solver.alpha[i]=pbvi_solver.rho_alpha[i]+[pbvi_solver.ursa.r_action_civil[max_action];pbvi_solver.ursa.r_action_hostile[max_action]]
        #initial alpha= rho_alpha+r_alpha
    end =#
end

function get_reward(b::Float64, a::Int64, pbvi_solver::PBVI_solver)
    temp=zeros(pbvi_solver.N_B)
    for i=1:pbvi_solver.N_B
        temp[i]=dot_alpha_b(pbvi_solver.rho_alpha[i],b)
    end
    reward_rho=maximum(temp)
    @assert reward_rho<=entropy_ursa(b)+10.0^(-6.0)
    reward_r=pbvi_solver.ursa.r_action_hostile[a]*b+pbvi_solver.ursa.r_action_civil[a]*(1-b)
    return reward_rho+reward_r
end

function solve!(pbvi_solver::PBVI_solver)
    N_action=pbvi_solver.ursa.Num_actions
    N_b=pbvi_solver.N_B
    N_ob=pbvi_solver.ursa.Num_t_actions
    gamma=pbvi_solver.ursa.discount_factor
    pbvi_solver.opt_action=Array{Int64}(N_b)
    #see PBVI paper section 3.1 pseudo code
    Gamma_a_star=Array{Array{Float64,1},2}(N_action,N_b)
    for i=1:N_action
        for j=1:N_b
            Gamma_a_star[i,j]=pbvi_solver.rho_alpha[j]+[pbvi_solver.ursa.r_action_civil[i];pbvi_solver.ursa.r_action_hostile[i]]
        end
    end

    for n_iter=1:pbvi_solver.ursa.N_iter
        Gamma_a_o=Array{Array{Float64,1},3}(N_action,N_ob,N_b)
        for i=1:N_action
            for j=1:N_ob
                for k=1:N_b
                    Gamma_a_o[i,j,k]=gamma*[pbvi_solver.ursa.emit_prob_0[i,j];pbvi_solver.ursa.emit_prob_1[i,j]].*pbvi_solver.alpha[k]
                end
            end
        end

        Gamma_a_b=Array{Array{Float64,1},2}(N_action,N_b)
        temp_alpha=Array{Array{Float64,1},3}(N_action,N_b,N_ob)
        for i=1:N_action
            for j=1:N_b
                Gamma_a_b[i,j]=Gamma_a_star[i,j]
                for k=1:N_ob
                    temp=get_argmax_alpha(Gamma_a_o[i,k,:],pbvi_solver.B[j])
                    @assert size(temp[1])==(2,)
                    Gamma_a_b[i,j]+=temp[1]
                    temp_alpha[i,j,k]=temp[1]
                end
            end
        end

        #pbvi_solver.Value=zeros(N_b)
        for i=1:N_b
            pbvi_solver.alpha[i],pbvi_solver.opt_action[i]=get_argmax_alpha(Gamma_a_b[:,i],pbvi_solver.B[i])
            pbvi_solver.Value[i]=dot_alpha_b(pbvi_solver.alpha[i],pbvi_solver.B[i])
        end
    end #end the iteration loop
    Plots.plot(pbvi_solver.B,pbvi_solver.Value,xlabel="Belief",ylabel="Value",label="Value function")
    Plots.plot!(pbvi_solver.B,pbvi_solver.opt_action, label="Optimal actions")
end

function get_argmax_alpha(array_alpha::Array{Array{Float64,1},1},b::Float64)
    max_dot=-Inf
    max_index=0
    for (i,alpha) in enumerate(array_alpha)
        dot=dot_alpha_b(alpha,b)
        if dot>max_dot
            max_dot=dot
            max_index=i
        end
    end
    return deepcopy(array_alpha[max_index]), max_index
end

function robust_solve!(pb::PBVI_solver)
    N_action=pb.ursa.Num_actions
    N_b=pb.N_B
    N_ob=pb.ursa.Num_t_actions
    gamma=pb.ursa.discount_factor

    for n_iter=1:pb.ursa.N_iter
        pb.opt_action=Array{Int64}(N_b)

        Lam_til=Array{Array{Float64,1},1}()
        for i=1:N_b
            Lam_b_til=Array{Array{Float64,1},1}()
            for j=1:pb.ursa.Num_actions
                pb.p_opt[i,j], alpha_z_opt= get_p_alpha(pb,pb.B[i],j)
                #line 12 of algorithm 3 robust DP backup
                alpha_opt=pb.rho_alpha[i]+[pb.ursa.r_action_civil[j];pb.ursa.r_action_hostile[j]]
                temp=zeros(2)
                for k=1:pb.ursa.Num_t_actions
                    temp+=pb.p_opt[i,j][k,:].*alpha_z_opt[k]
                end
                temp*=pb.ursa.discount_factor
                alpha_opt+=temp
                push!(Lam_b_til,deepcopy(alpha_opt))
            end
            @assert size(Lam_b_til)==(pb.ursa.Num_actions,)
            argmax_alpha, pb.opt_action[i]=get_argmax_alpha(Lam_b_til,pb.B[i])
            @assert size(argmax_alpha)==(2,)
            pb.Value[i]=dot_alpha_b(argmax_alpha,pb.B[i])
            push!(Lam_til, deepcopy(argmax_alpha))
        end
        pb.alpha=deepcopy(Lam_til)
    end
    Plots.plot(pb.B,pb.Value,xlabel="Belief",ylabel="Value",label="Robust Value function")
    Plots.plot!(pb.B,pb.opt_action, label="Optimal actions")
end


function get_p_alpha(pb::PBVI_solver,b::Float64,a::Int64)
    m=Model(solver=ClpSolver())
    @variable(m, U[1:pb.ursa.Num_t_actions])
    @variable(m, p[1:pb.ursa.Num_t_actions,1:2]) #last dim represents lambda, p's bound depends on a
    @variable(m, rhs_5[1:pb.N_B,1:pb.ursa.Num_t_actions])
    @objective(m, Min, sum(U))
    #first constraint in (5)
    for i=1:pb.N_B
        for j=1:pb.ursa.Num_t_actions
            @constraint(m, rhs_5[i,j]==(1-b)*p[j,1]*pb.alpha[i][1]+b*p[j,2]*pb.alpha[i][2])
            @constraint(m, U[j]>=rhs_5[i,j])
        end
    end
    #second constraint in (5)
    for i=1:pb.ursa.Num_t_actions
        @constraint(m, p[i,1]>=pb.ursa.emit_prob_0[a,i]*pb.ursa.uncertain_factor)
        @constraint(m, p[i,1]<=pb.ursa.emit_prob_0[a,i]/pb.ursa.uncertain_factor)
        @constraint(m, p[i,2]>=pb.ursa.emit_prob_1[a,i]*pb.ursa.uncertain_factor)
        @constraint(m, p[i,2]<=pb.ursa.emit_prob_1[a,i]/pb.ursa.uncertain_factor)
    end
    #normalization constraint
    @constraint(m, sum(p[:,1])==1.0)
    @constraint(m, sum(p[:,2])==1.0)
    status=solve(m)
    #now retrieve the needed variables
    U_opt=getvalue(U)
    alpha_opt=Array{Array{Float64,1},1}(pb.ursa.Num_t_actions)
    rhs_5_opt=getvalue(rhs_5)
    for j=1:pb.ursa.Num_t_actions
        for i=1:pb.N_B
            if rhs_5_opt[i,j]>=U_opt[j]-10.0^(-3.0)
                alpha_opt[j]=pb.alpha[i]  #retrieve the optimal alpha that make the constraint tight
            end
        end
    end
    return getvalue(p), alpha_opt
end

#pay attention!!! We are not actually solving for the finite horizon problem
#this solved value function is an approximate for the infiniy horizon problem
# we mix the use of N_iter and N_steps
function policy(b::Float64,pb::PBVI_solver,flag::String)
    diff=abs.(pb.B-b)
    nearest=indmin(diff) # index for the belief
    opt_action=pb.opt_action[nearest]
    if flag=="robust"
        return opt_action,pb.p_opt[nearest,opt_action]
    end
    if flag=="non robust"||flag=="chance"
        return pb.opt_action[nearest],[pb.ursa.emit_prob_0[opt_action,:] pb.ursa.emit_prob_1[opt_action,:]]
    end
end

function adversarial_p(b::Float64,pb_rob::PBVI_solver,a::Int64)
    #get the adversarial prob given belief, robust solved policy (to retrieve the p_opt) and an action
    diff=abs.(pb_rob.B-b)
    nearest=indmin(diff) # index for the belief
    #opt_action=pb.opt_action[nearest]
    return pb_rob.p_opt[nearest,a]
end

function belief_update(b::Float64, p::Array{Float64,2}, z::Int64)
    b_up=p[z,2]*b/(p[z,2]*b+p[z,1]*(1-b))
    return b_up
end

function interp_value(b::Float64, pb::PBVI_solver)
    #use linear interpolation to get the value at anywhere [0,1]
    @assert b<=1.0
    @assert b>=0
    if b<=pb.B[1]
        delta_b=pb.B[2]-pb.B[1]
        delta_value=pb.Value[2]-pb.Value[1]
        slope=delta_value/delta_b
        delta=b-pb.B[1]
        return pb.Value[1]+slope*delta
    end
    if b>=pb.B[end]
        delta_b=pb.B[end]-pb.B[end-1]
        delta_value=pb.Value[end]-pb.Value[end-1]
        slope=delta_value/delta_b
        delta=b-pb.B[end]
        return pb.Value[end]+slope*delta
    end
    for i=1:pb.N_B-1
        if b>=pb.B[i]&& b<=pb.B[i+1]  #b between i and i+1
            delta_b=pb.B[i+1]-pb.B[i]
            delta_value=pb.Value[i+1]-pb.Value[i]
            slope=delta_value/delta_b
            delta=b-pb.B[i]
            return pb.Value[i]+slope*delta
        end
    end
end


function chance_solve!(pb::PBVI_solver,epsilon::Float64,sigma2::Float64)
    #epsilon is the percentile
    #note that the get_chance_threshold only works for 2 observations
    N_action=pb.ursa.Num_actions
    N_b=pb.N_B
    N_ob=pb.ursa.Num_t_actions
    gamma=pb.ursa.discount_factor
    for n_iter=1:pb.ursa.N_iter
        pb.opt_action=Array{Int64}(N_b)
        for i=1:N_b
            value_a=zeros(pb.ursa.Num_actions)  #record the value of a at that belief point b
            for j=1:pb.ursa.Num_actions
                value_a[j]= get_chance_threshold(pb,pb.B[i],j,epsilon,sigma2)
            end
            pb.Value[i],pb.opt_action[i]=findmax(value_a)
        end
    end
    Plots.plot(pb.B,pb.Value,xlabel="Belief",ylabel="Value",label="Chance Value function")
    Plots.plot!(pb.B,pb.opt_action, label="Optimal actions")
end

function get_chance_threshold(pb::PBVI_solver,b::Float64,a::Int64,epsilon::Float64,sigma2::Float64)
    #solve a chance optimization
    #epsilon is the percentile, sigma2 is the variance
    reward=get_reward(b,a,pb)
    gamma=pb.ursa.discount_factor
    #check if this matrix is correct
    p=[pb.ursa.emit_prob_0[a,1] pb.ursa.emit_prob_1[a,1];
    pb.ursa.emit_prob_0[a,2] pb.ursa.emit_prob_1[a,2]]
    b1=belief_update(b,p,1)  #if receive obs1
    b2=belief_update(b,p,2)  #if receive obs2
    V1=interp_value(b1,pb)
    V2=interp_value(b2,pb)
    p11=pb.ursa.emit_prob_0[a,1]
    p21=pb.ursa.emit_prob_0[a,2]
    p12=pb.ursa.emit_prob_1[a,1]
    p22=pb.ursa.emit_prob_1[a,2]
    mean_lhs=(1-b)*(p11*V1+p21*V2)+b*(p12*V1+p22*V2)
    mean_lhs=gamma*mean_lhs
    var_lhs=gamma^2*sigma2*(b^2*(V1-V2)^2+(1-b)^2*(V1-V2)^2)
    y=mean_lhs-sqrt(var_lhs)*erfinv(1.0-epsilon)
    return reward+y
end

function ursa_target_action_sample(u::URSA,a::Int64)
    if u.True_lambda==1
        if !isprobvec(u.emit_prob_1[a,:])
            print(u.emit_prob_1[a,:])
        end
        @assert abs(sum(u.emit_prob_1[a,:])-1)<=10.0^(-5.0)
        dis=Categorical(u.emit_prob_1[a,:])
    elseif u.True_lambda==0
        if !isprobvec(u.emit_prob_0[a,:])
            print(u.emit_prob_0[a,:])
        end
        @assert abs(sum(u.emit_prob_0[a,:])-1)<=10.0^(-5.0)
        dis=Categorical(u.emit_prob_0[a,:])
    end
    sample_action=rand(dis)
    @assert sample_action>0
    return sample_action
end

function simulate_true_reward_ursa(i_b::Float64, pb::PBVI_solver, flag::String, flag_ursa::String, u_true::URSA)
    @assert flag=="robust"||flag=="non robust"||flag=="chance"
    @assert flag_ursa=="norminal"||flag_ursa=="random"||flag_ursa=="deter"
    N=pb.ursa.N_steps
    a=zeros(Int64, N)
    a_t=zeros(Int64, N)
    beliefs=zeros(N)
    beliefs[1]=i_b
    cumu_reward=0.0
    imme_reward=zeros(N)
    p_opt=Array{Array{Float64,2},1}(N)
    for i=1:N
        a[i], p_opt[i]=policy(beliefs[i],pb,flag)
        if flag_ursa=="random"
            u=sample_ursa(pb)
            a_t[i]=ursa_target_action_sample(u,a[i])
        elseif flag_ursa=="norminal"
            a_t[i]=ursa_target_action_sample(pb.ursa,a[i])
        elseif flag_ursa=="deter"
            a_t[i]=ursa_target_action_sample(u_true,a[i])
        end
        r=get_true_reward(beliefs[i], a[i], pb)
        imme_reward[i]=r
        cumu_reward+=r*pb.ursa.discount_factor^(i-1)
        if i<=N-1
            @assert a_t[i]>0
            beliefs[i+1]=belief_update(beliefs[i], p_opt[i], a_t[i])
        end
        #println("Step=",i,", belief=",round(beliefs[i],3),", action=",a[i],", t_action=",a_t[i],", i_reward=",round(r,3),", c_reward=",round(cumu_reward,3))
    end
    return cumu_reward
end

function simulate_true_reward_ursa_adversary(i_b::Float64, pb::PBVI_solver, flag::String, pb_rob::PBVI_solver)
    @assert flag=="robust"||flag=="non robust"||flag=="chance"
    N=pb.ursa.N_steps
    a=zeros(Int64, N)
    a_t=zeros(Int64, N)
    beliefs=zeros(N)
    beliefs[1]=i_b
    cumu_reward=0.0
    imme_reward=zeros(N)
    p_opt=Array{Array{Float64,2},1}(N)
    p_true=Array{Array{Float64,2},1}(N)
    for i=1:N
        a[i], p_opt[i]=policy(beliefs[i],pb,flag)
        p_true[i]=adversarial_p(beliefs[i],pb_rob,a[i])  #get an adversarial probability
        dis=Categorical(p_true[i][:,pb.ursa.True_lambda+1])
        a_t[i]=rand(dis)
        r=get_true_reward(beliefs[i], a[i], pb)
        imme_reward[i]=r
        cumu_reward+=r*pb.ursa.discount_factor^(i-1)
        if i<=N-1
            @assert a_t[i]>0
            beliefs[i+1]=belief_update(beliefs[i], p_opt[i], a_t[i])
        end
        #println("Step=",i,", belief=",round(beliefs[i],3),", action=",a[i],", t_action=",a_t[i],", i_reward=",round(r,3),", c_reward=",round(cumu_reward,3))
    end
    return cumu_reward
end

function get_true_reward(b::Float64, a::Int64, pbvi_solver::PBVI_solver)
    if pbvi_solver.ursa.True_lambda==1
        reward_rho=log(b)
        reward_r=pbvi_solver.ursa.r_action_hostile[a]
    elseif pbvi_solver.ursa.True_lambda==0
        reward_rho=log(1-b)
        reward_r=pbvi_solver.ursa.r_action_civil[a]
    end
    return reward_rho+reward_r
end

function sample_ursa(pb::PBVI_solver)
    u=URSA(pb.ursa.discount_factor,pb.ursa.Num_actions,pb.ursa.True_lambda,pb.ursa.Num_t_actions,pb.ursa.uncertain_factor,pb.ursa.N_steps,pb.ursa.N_iter)
    #now u already has emit and rewards
    dis=Categorical([0.5;0.5])
    for i=1:pb.ursa.Num_actions
        rand_ind=rand(dis)
        range=u.emit_prob_0[i,rand_ind]/u.uncertain_factor-u.emit_prob_0[i,rand_ind]*u.uncertain_factor
        u.emit_prob_0[i,rand_ind]*=u.uncertain_factor
        u.emit_prob_0[i,rand_ind]+=rand()*range
        u.emit_prob_0[i,rand_ind]=min(u.emit_prob_0[i,rand_ind],1.0)
        if rand_ind==1
            u.emit_prob_0[i,2]=1-u.emit_prob_0[i,1]
        elseif rand_ind==2
            u.emit_prob_0[i,1]=1-u.emit_prob_0[i,2]
        end
        @assert abs(u.emit_prob_0[i,1]+u.emit_prob_0[i,2]-1)<=10.0^(-5.0)
    end

    for i=1:pb.ursa.Num_actions
        rand_ind=rand(dis)
        range=u.emit_prob_1[i,rand_ind]/u.uncertain_factor-u.emit_prob_1[i,rand_ind]*u.uncertain_factor
        u.emit_prob_1[i,rand_ind]*=u.uncertain_factor
        u.emit_prob_1[i,rand_ind]+=rand()*range
        u.emit_prob_1[i,rand_ind]=min(u.emit_prob_1[i,rand_ind],1.0)
        if rand_ind==1
            u.emit_prob_1[i,2]=1-u.emit_prob_1[i,1]
        elseif rand_ind==2
            u.emit_prob_1[i,1]=1-u.emit_prob_1[i,2]
        end
        @assert abs(u.emit_prob_1[i,1]+u.emit_prob_1[i,2]-1)<=10.0^(-5.0)
    end
    return u
end

export URSA, URSAState, PBVI_solver, adversarial_p, belief_update, dot_alpha_b, entropy_ursa, get_argmax_alpha
export get_p_alpha, get_reward, get_true_reward, initialize_alpha!, policy, robust_solve!, sample_ursa
export simulate_true_reward_ursa, simulate_true_reward_ursa_adversary, solve!, ursa_target_action_sample
export get_chance_threshold, interp_value, chance_solve!
end
