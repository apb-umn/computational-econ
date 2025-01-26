using Parameters,LinearAlgebra,BasisMatrices,SparseArrays,Arpack,Roots, KrylovKit,QuantEcon, PrettyTables

using ForwardDiff,Dierckx,Plots,NLsolve,NPZ

Approx1D = Interpoland{Basis{1,Tuple{SplineParams{Array{Float64,1}}}},Array{Float64,1},BasisMatrix{Tensor,SparseArrays.SparseMatrixCSC{Float64,Int64}}}

"""
    Contains the Parameters of the Aiyagari Model
"""
@with_kw mutable struct OCModel
    #SteadyState Parameters
    σ::Float64 = 2. #Risk Aversion
    β::Float64 = 0.9412677574899786  #Discount Factor
    σ_θw::Float64 = 0.13 #standard deviation of income shock θ
    ρ_θw::Float64 = 0.966 #persistence of the productivity shocks
    N_θw::Int = 5 #number of gridpoints for the productivity shocks
    σ_ε::Float64 = 0.001 #standard deviation of taste shock ε
    Γf::Function = κ->0


    #Entrepreneur parameters
    α::Float64 = 0.3 #curvature of production function capital
    ν::Float64 = 0.3 #curvature of production function labor
    δ::Float64 = 0.1 #depreciation rate
    χ::Float64 = 5. #collateral constraint
    σ_θb::Float64 = 0.26 #standard deviation of income shock θ
    ρ_θb::Float64 = 0.95 #persistence of the productivity shocks
    N_θb::Int = 5 #number of gridpoints for the productivity shocks
    
    πθ::Matrix{Float64} = ones(N_θb*N_θw,N_θb*N_θw) #transition matrix
    a̲::Float64  = 0. #Borrowing constraint
    amax::Float64 = 200.
    Nθ::Int = N_θb*N_θw #number of gridpoints for the productivity shocks
    Na::Int = 25 #number of gridpoints for splines
    curv_interp::Float64 = 2.5 #controls spacing for interpolation
    ka::Int = 2 #Spline Order
    curv_hist::Float64 = 2. #controls spacing for histogram
    Ia::Int = 500 #number of gridpoints for histogram
    R̄::Float64 =1.02 #Equlibrium gross interest rate
    W̄::Float64 = 1.  #Equilibrium wage.
    K2Y::Float64 = 3 #target capital to output ratio
    Θ̄::Float64 = 1.5 #Level of TFP for corporate sector
    N̄::Float64 = 1. #Average labor supply

    #Helpful Objects
    a′grid::Vector{Float64} = zeros(1)#grid vector for endogenous grid
    abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}} =Basis(SplineParams(collect(LinRange(0,1,60)),0,2)) #basis for endogenous grid
    aw_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
    ab_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na)
    EΦeg::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na) #expectations of basis functions at gridpoints on potential savings
    EΦ_aeg::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na) #expectations of basis functions at gridpoints on potential savings
    ω̄::Vector{Float64} = ones(1)/1000 #masses for the stationary distribution
    Λ::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia)

    #policy functions
    V̄coefs::Vector{Float64} = zeros(1) # value function coefficients
    X̄::Vector{Float64} = zeros(1) #steady state objects

    #grids
    âgrid::Vector{Float64} = zeros(1) # gridpoints for the approximations
    āgrid::Vector{Float64} = zeros(1) # gridpoints for the distribution   
    θ::Matrix{Float64} = ones(1,1) #productivity levels
    âθ̂::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef,1) #gridpoints for the approximations 
    āθ̄::Matrix{Float64} = ones(1,1) #gridpoints for the distribution


end

"""
computeoptimalconsumption_eg(OCM::OCModel,Vcoefs::Vector{Float64})

This function takes as inputs spline coeff. for λ (marginal utility function) 
and uses Carrol's EGM to compute consumption policy function.  
"""
function computeoptimalconsumptionW_eg(OCM::OCModel,Vcoefs::Vector{Float64})
    # extract the wage and interest rate
    @unpack σ,β,θ,a̲,EΦ_aeg,a′grid,R̄,W̄ = OCM
    lθb,lθw = θ[:,1],θ[:,2];
    Sw = length(lθw);
    θw = exp.(lθw);

    EVₐ′ = reshape(EΦ_aeg*Vcoefs,:,Sw) #precomputing expectations
    EVₐ′ = max.(EVₐ′,1e-6)

    # compute consumption today implied by EE
    cEE = (β.*EVₐ′).^(-1/σ) #consumption today
    # compute asset today implied by savings and consumtion
    Implieda = (a′grid .+ cEE .- W̄.*θw')./R̄  #Implied assets today

    # now we want to figure out the consumption policy function
    # We know that for all a∈[a̲,implied_a(a′=̲a)] we have the borrowing constraint binding
    # so we check a[1,s]. If its bigger than ̲b then consumption is given by budget constraint
    # for all a≤a[1,s]. For the rest we use the interpolation using the EE consumption
      

    cf = Vector{Spline1D}(undef,Sw)#implied policy rules for each productivity
    af = Vector{Spline1D}(undef,Sw)#implied policy rules for each productivity
    for s in 1:Sw
        #with some productivities the borrowing constraint does not bind
        if issorted(Implieda[:,s])
            if Implieda[1,s] > a̲ #borrowing constraint binds
                OCM.aw_cutoff[θ[s,:]] = Implieda[1,s]
                #add extra points on the borrowing constraint for interpolation
                â = [a̲;Implieda[:,s]]
                ĉ = [R̄*a̲-a̲ + W̄*θw[s];cEE[:,s]]
                cf[s] = Spline1D(â,ĉ,k=1)
                af[s] = Spline1D(â,[a̲;a′grid],k=1)
            else
                OCM.aw_cutoff[θ[s,:]] = -Inf
                cf[s] = Spline1D(Implieda[:,s],cEE[:,s],k=1)
                af[s] = Spline1D(Implieda[:,s],a′grid,k=1)
            end
        else
            p = sortperm(Implieda[:,s])
            if Implieda[p[1],s] > a̲ #borrowing constraint binds
                OCM.aw_cutoff[θ[s,:]] = Implieda[p[1],s]
                #add extra points on the borrowing constraint for interpolation
                â = [a̲;Implieda[p,s]]
                ĉ = [R̄*a̲-a̲ + W̄*θw[s];cEE[p,s]]
                cf[s] = Spline1D(â,ĉ,k=1)
                af[s] = Spline1D(â,[a̲;a′grid[p]],k=1)
            else
                OCM.aw_cutoff[θ[s,:]] = -Inf
                cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1)
                af[s] = Spline1D(Implieda[p,s],a′grid[p],k=1)
            end
        end
    end
    return cf,af
end


"""
computeoptimalconsumption_λ(OCM::OCModel,λcoefs)

This function takes as inputs spline coeff. for λ (marginal utility function) 
and uses Carrol's EGM to compute consumption policy function.  
"""
function computeoptimalconsumptionB_eg(OCM::OCModel,Vcoefs::Vector{Float64})
    @unpack σ,β,θ,a̲,EΦ_aeg,a′grid,R̄,W̄,α,ν,δ,χ ,amax= OCM
    lθb,lθw = θ[:,1],θ[:,2];
    Sb = length(lθb);
    θb = exp.(lθb);
    Na = length(a′grid)
    # compute expectations of marginal utility at gridpoints on potential savings
    #∑π_θ(s,s′) U_c (c(a′,s′)) = ∑π_θ(s,s′)λ(a′,s′)
    #∑π_θ(s,s′)∑λcoeff^j ϕ^j(a′,s^′)=EΦeg*λcoefs
    #so EΦeg is a matrix that evaluates the basis functions on the potential savings grid and sums up across shock

    #Start by computing the firms profit if unconstrained
    r̄ = R̄-1
    n̂ = ν*(r̄+δ)/(α*W̄) #labor to capital ratio
    k = @. (W̄/(ν*θb*n̂^(ν-1)))^(1/(α+ν-1))
    πu = @. θb*k^α*(n̂*k)^ν - r̄*k - δ*k - W̄*(n̂*k)

    EVₐ′ = reshape(EΦ_aeg*Vcoefs,:,Sb) #precomputing expectations 
    EVₐ′ = max.(EVₐ′,1e-6)
    # compute consumption today implied by EE
    cEE = (β.*EVₐ′).^(-1/σ) #consumption today

    # compute asset today implied by savings and consumtion
    Implieda = (a′grid .+ cEE .- πu')/R̄  #Implied assets today
    

    #Find out where the borrowing constraint binds


    # now we want to figure out the consumption policy function
    #we'll need to adjust for the collarteral constraint binding for some points
      

    cf = Vector{Spline1D}(undef,Sb)#implied policy rules for each productivity
    af = Vector{Spline1D}(undef,Sb)
    kf = Vector{Spline1D}(undef,Sb)
    nf = Vector{Spline1D}(undef,Sb)
    yf = Vector{Spline1D}(undef,Sb)
    πf = Vector{Spline1D}(undef,Sb)

    k = ones(Na).*k'
    πb = ones(Na).*πu'
    n = k*n̂
    y = θb'.*k.^α.*n.^ν
    
    for s in 1:Sb
        mask_c = χ.*a′grid .< k[1,s] #same k unconstrained for all assets


        #Next we can compute the implied assets if the collateral constraint binds
        kc = max.(χ.*a′grid[mask_c],1e-6)
        nc = (W̄./(ν.*θb[s].*kc.^α)).^(1/(ν-1))
        πc = θb[s].*kc.^α.*nc.^ν - r̄.*kc - δ.*kc- W̄.*nc
        ξ = min.(α.*θb[s].*kc.^(α-1).*nc.^ν .- r̄ .- δ,0.99./χ) #collateral constraint multiplier # makes the equation below sensible
        cEE_c = (β.*EVₐ′[mask_c,s]./(1 .- ξ.*χ)).^(-1/σ) #consumption today
        Implieda[mask_c,s] = (a′grid[mask_c] .+ cEE_c .- πc )/R̄  #Implied assets today
        cEE[mask_c,s] = cEE_c
        OCM.ab_cutoff[θ[s,:]] = k[s]/χ
        k[mask_c,s] = kc
        n[mask_c,s] = nc
        y[mask_c,s] = θb[s].*kc.^α.*nc.^ν
        πb[mask_c,s] = πc
    #with some productivities the borrowing constraint does not bind
    a̲₋ = a̲+((amax-a̲)/Na)*1e-6
    c₋ = R̄*a̲₋-a̲
    #a̲₊ = Implieda[1,s]-((amax-a̲)/Na)
    #c₊ = R̄*a̲₊-a̲



    if issorted(Implieda[:,s])
            cf[s] = Spline1D(Implieda[:,s],cEE[:,s],k=1)
            af[s] = Spline1D(Implieda[:,s],a′grid,k=1)
            kf[s] = Spline1D(Implieda[:,s],k[:,s],k=1)
            nf[s] = Spline1D(Implieda[:,s],n[:,s],k=1)
            yf[s] = Spline1D(Implieda[:,s],y[:,s],k=1)
            πf[s] = Spline1D(Implieda[:,s],πb[:,s],k=1)


    else
        p = sortperm(Implieda[:,s])
            cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1)
            af[s] = Spline1D(Implieda[p,s],a′grid[p],k=1)
            kf[s] = Spline1D(Implieda[p,s],k[p,s],k=1)
            nf[s] = Spline1D(Implieda[p,s],n[p,s],k=1)
            yf[s] = Spline1D(Implieda[p,s],y[p,s],k=1)
            πf[s] = Spline1D(Implieda[p,s],πb[p,s],k=1)
        #end
     end
end
    return cf,af,kf,nf,yf,πf
end


"""
iterate_eg!(OCM::OCModel,luΦ,Vcoefs)

Iterates on the λ function
λcoef Φ = T (λcoef)
where T   operator takes λcoeff, computes consumption policy function using EGM and then uses 
the form of  computes the marginal utility function to compute the new λ policy function
and finally finds the new λcoef by solving the linear system     
"""
function iterate_eg!(OCM::OCModel,luΦ,Vcoefs)
    @unpack σ,β,Φ,θ,πθ,R̄,W̄,Γf,σ_ε = OCM
    S = size(θ,1)

    agrid = OCM.a′grid
    Na = length(agrid)
    #Compute optimal consumption function for current λ function stored using Φ and λcoefs
    cf_w,af_w = computeoptimalconsumptionW_eg(OCM,Vcoefs)     
    cf_b,af_b,kf,nf,yf,πf = computeoptimalconsumptionB_eg(OCM,Vcoefs)
    #compute EΦ
    c_w,c_b = zeros(Na,S),zeros(Na,S) 
    a_w,a_b = zeros(Na,S),zeros(Na,S)
    Vw,Vb = zeros(Na,S),zeros(Na,S)
    for s in 1:S
        c_w[:,s] = cf_w[s](agrid) #compute consumption at gridpoints
        c_b[:,s] = cf_b[s](agrid) #compute consumption at gridpoints
        a_w[:,s] = af_w[s](agrid) #asset choice
        a_b[:,s] = af_b[s](agrid) #asset choice
        EΦw = kron(πθ[s,:]',BasisMatrix(OCM.abasis,Direct(),a_w[:,s]).vals[1])
        EΦb = kron(πθ[s,:]',BasisMatrix(OCM.abasis,Direct(),a_b[:,s]).vals[1])
        Vw[:,s] = c_w[:,s].^(1-σ)/(1-σ) + β.*EΦw*Vcoefs
        Vb[:,s] = c_b[:,s].^(1-σ)/(1-σ) + β.*EΦb*Vcoefs
    end
    p = Γf.(Vb.-Vw)
    V = p.*Vw .+ (1 .- p).*Vb
    ptol = 1e-8
    maskp =ptol.< p .< 1-ptol
    V[maskp] .= Vw[maskp] .+ σ_ε.*log.(1 .+ exp.((Vb[maskp].-Vw[maskp])./σ_ε))
    #V = σ_ε*log.(exp.(Vw/σ_ε) .+ exp.(Vb/σ_ε))

    #now get the implied value functions
    Vf_w = [Spline1D(agrid,Vw[:,s],k=1) for s in 1:S]
    Vf_b = [Spline1D(agrid,Vb[:,s],k=1) for s in 1:S]

    # update the λcoef using the linear system Φλcoefs = λ

    Vcoefs′ = luΦ\V[:]
    diff = norm(Vcoefs.-Vcoefs′)
    Vcoefs .= Vcoefs′

    wf = (c=cf_w,a=af_w,v=Vf_w)
    bf = (c=cf_b,a=af_b,v=Vf_b,k=kf,n=nf,y=yf,π=πf)
    return diff,wf,bf
end

"""
    solve_eg!(OCM::OCModel,Vcoefs,tol=1e-8)


Solves the functional equation for λ.
"""
function solve_eg!(OCM::OCModel,Vcoefs,tol=1e-8)
    #increases stability to iterate on the time dimension a few times
    diff = 1.
    luΦ = lu(OCM.Φ)
    n = 0
    while diff > tol && n < 5000
        #println(diff)
        #then use newtons method
        diff = iterate_eg!(OCM,luΦ,Vcoefs)[1]
        n += 1
    end
    if n> 5000
        println("Did not converge: $diff")
    end
end



"""
    solvebellman(OCM::OCModel)

Helper function to obtainthe  value function using optimal consumption functions
"""
function solvebellman(OCM::OCModel)
    @unpack σ,β,Φ,θ,πθ,R̄,W̄,a′grid,cf,abasis = OCM
    S = length(θ)
    V = Vector{Interpoland}(undef,1)
    V = [Interpoland(abasis,a->((1-β)*a.+1).^(1-σ)./(1-σ)./(1-β)) for s in 1:S] #initialize with value function equal to β*a
    Vcoefs = vcat([V[s].coefs for s in 1:S]...)::Vector{Float64}
    agrid=a′grid
    Na = length(agrid)
    c = zeros(Na*S) 
    EΦ = spzeros(Na*S,Na*S)
    for s in 1:S
        for s′ in 1:S
            c[(s-1)*Na+1:s*Na] = cf[s](agrid) #compute consumption at gridpoints
            a′ = R̄*agrid .+ exp(θ[s])*W̄ .- c[(s-1)*Na+1:s*Na] #asset choice
            EΦ[(s-1)*Na+1:s*Na,(s′-1)*Na+1:s′*Na] = πθ[s,s′]*BasisMatrix(abasis,Direct(),a′).vals[1][:]
        end
    end


    Jac =  Φ .-  β.*EΦ
    #the next line solves  Jac*Vcoefs=c.^(1-σ)./(1-σ)
    Vcoefs=Jac\c.^(1-σ)./(1-σ)


    for s in 1:S
        V[s].coefs .= Vcoefs[1+(s-1)*Na:s*Na]
    end
    return V 
end



"""
    setupgrids_and_VF!(OCM::OCModel,amax,curv=1.7)

Setup grid points for the OCModel Model given parameters
"""
function setupgrids_and_VF!(OCM::OCModel)
    @unpack a̲,Na,N_θb,N_θw,ρ_θw,σ_θw,ρ_θb,σ_θb,β,σ,Ia,amax,curv_interp,curv_hist,abasis = OCM
    xvec = LinRange(0,1,Na-1).^curv_interp  #The Na -1 to to adjust for the quadratic splines
    a′gridknots = a̲ .+ (amax - a̲).*xvec #nonlinear grid knots
    
    mc = rouwenhorst(N_θw,ρ_θw,σ_θw)
    πθw = mc.p
    Θwgrid = exp.(mc.state_values)
    πθwstat = real(eigs(πθw',nev=1)[2])
    πθwstat ./= sum(πθwstat)
    
    mc = rouwenhorst(N_θb,ρ_θb,σ_θb)
    πθb = mc.p
    Θbgrid = exp.(mc.state_values)
    πθbstat = real(eigs(πθb',nev=1)[2])
    πθbstat ./= sum(πθbstat)

    πθ = OCM.πθ = kron(πθb,πθw)
    θ = [kron(ones(length(Θwgrid)),Θbgrid) kron(Θwgrid,ones(length(Θbgrid)))]
    OCM.θ = log.(θ)
    S = size(πθ,1)




    #Grid pointsfor the policy and value functions
    abasis = Basis(SplineParams(a′gridknots,0,2))
    OCM.abasis = abasis;    
    agrid = nodes(abasis)[1]
    OCM.a′grid = agrid

    #Precompute  EΦeg that is used to  compute expectations of marginal utility at gridpoints on potential savings
    #∑π_θ(s,s′) U_c (c(a\prime,s′)) = ∑π_θ(s,s′)λ(a′,s′)
    #∑π_θ(s,s′)∑λcoeff^j ϕ^j(a′,s^′)=EΦeg*λcoefs
    #so EΦeg is a matrix that evaluates the basis functions on the exogenous potential savings grid and sums up across shock

    OCM.EΦeg = kron(πθ,BasisMatrix(abasis,Direct(),OCM.a′grid).vals[1])
    OCM.EΦ_aeg = kron(πθ,BasisMatrix(abasis,Direct(),OCM.a′grid,[1]).vals[1])
    #Precompute Phi
    OCM.Φ = kron(Matrix(I,S,S),BasisMatrix(abasis,Direct()).vals[1])
    c_guess = (1-β)*agrid .+ OCM.W̄

    Vcoefs = OCM.Φ\repeat(c_guess.^(1-σ)./(1-σ)/(1-β),S)
    #Grid for distriaution
    xvec = LinRange(0,1,Ia).^curv_hist 
    āgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grides
    OCM.āθ̄ = hcat(kron(ones(S),āgrid),kron(OCM.θ,ones(Ia)))
    OCM.ω̄ = ones(2*Ia*S)/(2*Ia*S) #include occupational choice

    #cutoffs
    OCM.aw_cutoff = Dict{Vector{Float64},Float64}()
    OCM.ab_cutoff = Dict{Vector{Float64},Float64}()

    OCM.Γf = κ-> 1/(1+exp(κ/OCM.σ_ε)) #CDF for Gumbel Distribution
    return Vcoefs
end

"""
    find_stationarydistribution!(OCM::OCModel,V)

Computes the stationary distribution 
"""
function find_stationarydistribution!(OCM::OCModel,Vcoefs)
    @unpack θ,πθ,Ia,āθ̄,R̄,W̄,Γf = OCM
    S = size(θ,1)
    diff,wf,bf = iterate_eg!(OCM,lu(OCM.Φ),Vcoefs)

    ā = āθ̄[1:Ia,1] #grids are all the same for all shocks
    a′w = max.(min.(hcat([wf.a[s](ā) for s in 1:S]...),ā[end]),ā[1]) 
    a′b =  max.(min.(hcat([bf.a[s](ā) for s in 1:S]...),ā[end]),ā[1]) 
    Vw = hcat([wf.v[s](ā) for s in 1:S]...)
    Vb = hcat([bf.v[s](ā) for s in 1:S]...)
    p = Γf.(Vb.-Vw)
    
    Qsw = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ā,0,1)),Direct(),@view a′w[:,s]).vals[1]') for s in 1:S]
    Qsb = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ā,0,1)),Direct(),@view a′b[:,s]).vals[1]') for s in 1:S]
    Λtemp = hcat(Qsw...,Qsb...)
    OCM.Λ = vcat(p[:].*Λtemp,(1 .- p[:]).*Λtemp)
    OCM.ω̄ .=  real(eigsolve(OCM.Λ,OCM.ω̄ ,1)[2])[1]
    OCM.ω̄ ./= sum(OCM.ω̄)

    nb = hcat([bf.n[s](ā) for s in 1:S]...)
    n̄ =  [-exp.(āθ̄[:,2]);nb[:]]
    kb = hcat([bf.k[s](ā) for s in 1:S]...)
    k̄ =  [zeros(Ia*S);kb[:]]
    πb = hcat([bf.π[s](ā) for s in 1:S]...)
    vw = hcat([wf.v[s](ā) for s in 1:S]...)
    vb = hcat([bf.v[s](ā) for s in 1:S]...)
    v̄  = [vw[:];vb[:]]
    ā = hcat([āθ̄[:,1];āθ̄[:,1]])
    
    return n̄,k̄,ā,v̄
end

"""
calibratesteadystate_λ!(OCM::OCModel)

Solves for the steady state without aggregate shocks
"""
function calibratesteadystate!(OCM::OCModel)
    @unpack Θ̄,α,N̄,K2Y,R̄ = OCM
    

    #Compute objects from targeted moments
    Y2K = 1/K2Y
    OCM.δ = δ = α*Y2K + 1 - R̄ #matching capital to output ratio and interest rate gives depreciation rate
    K2N = (Y2K/Θ̄)^(1/(α-1)) #relationship between capital to output and capital to labor
    OCM.W̄=W̄= (1-α)*Θ̄*K2N^α
    Vcoefs = setupgrids_and_VF!(OCM)

    function βTres(x)
        OCM.β = x[1]
        Vcoefs = setupgrids_and_VF!(OCM)
        solve_eg!(OCM,Vcoefs)
        n̄,k̄,ā,v̄ = find_stationarydistribution!(OCM,Vcoefs)
        N̄ = -dot(n̄,OCM.ω̄)
        K̄ = K2N*N̄
        sum(reshape(n̄.*OCM.ω̄,:,2),dims=1)
        #find_transitionΛ_λ!(OCM,λcoefs)
        #find_ω̄!(OCM)
        #find_stationarydistribution_λ!(OCM,λcoefs)
        res = [dot(OCM.ω̄,ā) - K̄ - dot(OCM.ω̄,k̄),sum(reshape(OCM.ω̄,:,2),dims=1)[2] - 0.13]
        #OCM.X̄ = [R̄,W̄,sum(reshape(OCM.ω̄,:,2),dims=1)[2],dot(OCM.ω̄,v̄),dot(OCM.ω̄,ā)]
        println("res: ",res)
        println("β: ",x)
        return res[1]
    end
    #Q̄ = 1/R̄
    #β= fzero(βTres,.85,0.95)
    ret = nlsolve(βTres,[OCM.β],ftol=1e-2)
    #if !converged(ret)
    #    error("Could not find steady state")
    #end
    #βTres(ret.zero)
    #save_policy_functions!(OCM,Vcoefs)
    #save_Λs!(OCM)
    OCM.V̄coefs = Vcoefs
    return Vcoefs
end


"""
calibratesteadystate_λ!(OCM::OCModel)

Solves for the steady state without aggregate shocks
"""
function solvesteadystate!(OCM::OCModel)
    @unpack Θ̄,α,δ = OCM
    
    Vcoefs = OCM.V̄coefs

    function ssRes(x)
        Q = x[1]
        OCM.R̄ = R̄ = 1/Q
        Y2K = (R̄-1+δ)/α
        K2N = (Y2K/Θ̄)^(1/(α-1))
        OCM.W̄ = W̄ = (1-α)*Θ̄*K2N^α
        solve_eg!(OCM,Vcoefs)
        n̄,k̄,ā,τ̄,v̄ = find_stationarydistribution!(OCM,Vcoefs)
        N̄ = -dot(n̄,OCM.ω̄)
        K̄ = K2N*N̄
        #find_transitionΛ_λ!(OCM,λcoefs)
        #find_ω̄!(OCM)
        #find_stationarydistribution_λ!(OCM,λcoefs)
        res = [dot(OCM.ω̄,ā) - K̄ - dot(OCM.ω̄,k̄),OCM.T̄-dot(OCM.ω̄,τ̄)]
        OCM.X̄ = [R̄,W̄,OCM.T̄,sum(reshape(OCM.ω̄,:,2),dims=1)[2],dot(OCM.ω̄,v̄),dot(OCM.ω̄,ā)]
        println(res[1])
        return res[1]
    end
    #Q̄ = 1/R̄
    #β= fzero(βres,Q̄^30,Q̄)
    ret = nlsolve(ssRes,[1/OCM.R̄],ftol=1e-6)
    #if !converged(ret)
    #    error("Could not find steady state")
    #end
    ssRes(ret.zero)
    #save_policy_functions!(OCM,Vcoefs)
    #save_Λs!(OCM)
    OCM.V̄coefs = Vcoefs
    return Vcoefs
end


 """
    save_policy_functions!(OCM::OCModel)

Saves the policy functions in the OCModel object
"""

function get_policy_functions!(OCM::OCModel,Vcoefs)
    @unpack R̄,W̄,πθ,σ,Ia,Nθ,Na,curv_interp,curv_hist,a̲,amax= OCM #then unpack equilibrium objects
    diff,wf,bf = iterate_eg!(OCM,lu(OCM.Φ),Vcoefs)
    
    #save the policy functions a,n,k,λ,v
    af(θ,a,c) = c==1 ? wf.a[[θ].==eachrow(OCM.θ)][1](a) : bf.a[[θ].==eachrow(OCM.θ)][1](a)
    nf(θ,a,c) = c==1 ? -exp.(θ[2]) : bf.n[[θ].==eachrow(OCM.θ)][1](a)
    kf(θ,a,c) = c==1 ? 0 : bf.k[[θ].==eachrow(OCM.θ)][1](a)
    cf(θ,a,c) = c==1 ? wf.c[[θ].==eachrow(OCM.θ)][1](a) : bf.c[[θ].==eachrow(OCM.θ)][1](a)
    λf(θ,a,c) = OCM.R̄*cf(θ,a,c).^(-σ)
    vf(θ,a,c) = c==1 ? wf.v[[θ].==eachrow(OCM.θ)][1](a) : bf.v[[θ].==eachrow(OCM.θ)][1](a)
    πf(θ,a,c) = c==1 ? 0 : bf.π[[θ].==eachrow(OCM.θ)][1](a)
    Ibf(θ,a,c) = c==1 ? 0 : 1

    xvec = LinRange(0,1,Na-1).^curv_interp  #The Na -1 to adjust for the quadratic splines
    OCM.âgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grid for knot points
    xvec = LinRange(0,1,Ia).^curv_hist 
    OCM.āgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grids for distribution

    return [af,nf,kf,πf,Ibf,λf,vf] #return xf
end



function save_Λs!(OCM::OCModel)
    @unpack R̄,W̄,θ,πθ,σ,Ia,Nθ,Na,curv_interp,curv_hist,a̲,amax,āθ̄, cf,abasis= OCM #then unpack equilibrium objects
    S = length(θ)
    
    âθ̂ = hcat(kron(ones(S),nodes(abasis)[1]),kron(1:S,ones(length(abasis))))
    agrid = nodes(abasis)[1]
    N = length(agrid)
    
    #construct Λ_z
    ā = āθ̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](ā) for s in 1:S]...) #consumption policy
    a′ = R̄.*ā .+ W̄.*exp.(θ') .- c #create a Ia×S grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if amax is correct
    a′ = max.(min.(a′,ā[end]),ā[1])
    ābasis = Basis(SplineParams(ā,0,1))
    f(x̂) = BasisMatrix(ābasis,Direct(),[x̂]).vals[1]

    Qa_a = spzeros(Ia*S,Ia)
    for i in 1:length(a′)
        Qa_a[i,:] = ForwardDiff.derivative(f,a′[i])
    end
    Q_a = spzeros(Ia*S,Ia*S)
    for s in 1:S
        Q_a[1+(s-1)*Ia:s*Ia,:] = kron(reshape(πθ[s,:],1,:),Qa_a[1+(s-1)*Ia:s*Ia,:]) 
    end
    
    #construct Λ_z for histogram
    OCM.Λ_z = Q_a'
end


function get_grids(OCM)
    @unpack âgrid,āgrid,πθ,θ= OCM #then unpack equilibrium objects
    aknots = [âgrid]
    a_sp = nodes(SplineParams(aknots[1],0,OCM.ka)) #construct gridpoints from knots
    a_Ω = āgrid
    nθ,nsp,nΩ = size(πθ,1),length(a_sp),length(a_Ω)
    aθ_sp = hcat(kron(ones(nθ),a_sp),kron(θ,ones(nsp)))
    aθc_sp = [aθ_sp ones(size(aθ_sp,1));aθ_sp 2*ones(size(aθ_sp,1))]
    aθ_Ω = hcat(kron(ones(nθ),a_Ω),kron(θ,ones(nΩ)))
    aθc_Ω = [aθ_Ω ones(size(aθ_Ω,1));aθ_Ω 2*ones(size(aθ_Ω,1))]

    #next get kinks
    ℵ = Int[]
    #for s in 1:nθ
    #    if OCM.a_cutoff[θ[s]] > -Inf
    #        push!(ℵ,findlast(a_sp .< OCM.a_cutoff[θ[s]])+(s-1)*nsp)
    #    end
    #end 
    mask = OCM.ω̄ .> 1e-10
    println("Maximum assets: $(maximum(aθc_Ω[mask,1]))")

    return aknots,OCM.ka,aθc_sp,aθc_Ω,ℵ
end

Base.show(io::IO, model::OCModel) = print(io, """
OCModel: Parameters of the Aiyagari Model
----------------------------------------
  # Steady State Parameters
  σ        = $(model.σ)       # Risk Aversion
  β        = $(model.β)       # Discount Factor
  σ_θw     = $(model.σ_θw)    # Standard deviation of income shock θ
  ρ_θw     = $(model.ρ_θw)    # Persistence of the productivity shocks
  N_θw     = $(model.N_θw)    # Number of gridpoints for the productivity shocks
  σ_ε      = $(model.σ_ε)     # Standard deviation of taste shock ε
  Γf       = $(model.Γf)      # Function mapping differences to probabilities

  # Entrepreneur Parameters
  α        = $(model.α)       # Curvature of production function capital
  ν        = $(model.ν)       # Curvature of production function labor
  δ        = $(model.δ)       # Depreciation rate
  χ        = $(model.χ)       # Collateral constraint
  σ_θb     = $(model.σ_θb)    # Standard deviation of entrepreneur income shock θ
  ρ_θb     = $(model.ρ_θb)    # Persistence of the entrepreneur productivity shocks
  N_θb     = $(model.N_θb)    # Number of gridpoints for entrepreneur productivity shocks

  πθ       =  ...     # Transition matrix
  a̲       = $(model.a̲)      # Borrowing constraint
  amax     = $(model.amax)    # Maximum asset value
  Nθ       = $(model.Nθ)      # Total number of gridpoints for productivity shocks
  Na       = $(model.Na)      # Number of gridpoints for splines
  curv_interp = $(model.curv_interp) # Controls spacing for interpolation
  ka       = $(model.ka)      # Spline order
  curv_hist = $(model.curv_hist)    # Controls spacing for histogram
  Ia       = $(model.Ia)      # Number of gridpoints for histogram
  R̄       = $(model.R̄)       # Equilibrium gross interest rate
  W̄       = $(model.W̄)       # Equilibrium wage
  K2Y      = $(model.K2Y)     # Target capital to output ratio
  Θ̄       = ...       # Level of TFP for the corporate sector
  N̄       = $(model.N̄)       # Average labor supply

  # Helpful Objects
  a′grid   = ...   # Grid vector for endogenous grid
  abasis   = $(model.abasis)   # Basis for endogenous grid
  aw_cutoff = $(model.aw_cutoff) # Points where borrowing constraint binds (worker)
  ab_cutoff = $(model.ab_cutoff) # Points where borrowing constraint binds (entrepreneur)
  Φ        = $(model.Φ)       # Sparse basis matrix for savings grid
  EΦeg     = $(model.EΦeg)    # Expectations of basis functions on the savings grid
  EΦ_aeg   = $(model.EΦ_aeg)  # Expectations of basis functions on potential savings
  ω̄       = $(model.ω̄)       # Masses for the stationary distribution
  Λ        = $(model.Λ)       # Transition matrix

  # Policy Functions
  V̄coefs  = $(model.V̄coefs)  # Value function coefficients
  X̄       = $(model.X̄)       # Steady-state objects

  # Grids
  âgrid   = $(model.âgrid)   # Gridpoints for approximations
  āgrid   = $(model.āgrid)   # Gridpoints for the distribution
  θ        = $(model.θ)       # Productivity levels
  âθ̂     = $(model.âθ̂)     # Gridpoints for approximations
  āθ̄     = $(model.āθ̄)     # Gridpoints for the distribution
""")
