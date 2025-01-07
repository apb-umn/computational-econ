using Parameters,BasisMatrices,SparseArrays,SuiteSparse,TensorOperations



function create_array_with_one(n::Int, position::Int)
    arr = zeros(n)  # Create an array of zeros of length n
    arr[position] = 1  # Set the specified position to 1
    return arr
end


function construct_selector_matrix(n::Int64, indices::Vector)
    m = length(indices)
    sel_matrix = sparse(1:m, indices, 1, m, n)
    return sel_matrix
end




function construct_abasis(zgrid::Vector)::Basis{1, Tuple{SplineParams{Vector{Float64}}}}
    abasis = Basis(SplineParams(zgrid,0,2))
    return abasis
end
    


function construct_x̄s(abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}},xf::Vector{Function},n::Nums)
    #@unpack iz = inputs
    x̄f = Matrix{Interpoland}(undef,n.x,n.θ)
    x̄=zeros(n.x,n.sp)
    for i in 1:n.x
        x̄f[i,:] .= [Interpoland(abasis,a->xf[i](a,s)) for s in 1:n.θ]
        x̄[i,:]  = hcat([x̄f[i,s].coefs' for s in 1:n.θ]...)
    end
    return x̄
end

function construct_Φ̃s(abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}},aθ_sp::Matrix{Float64},aθ_Ω::Matrix{Float64},af::Function,πθ::Matrix{Float64},n::Nums)
    a_sp = unique(aθ_sp[:,1])
    θ    = unique(aθ_sp[:,2])
    a_Ω  = unique(aθ_Ω[:,1])

    N = length(a_sp)

    Φ̃ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_sp).vals[1])'
    Φ̃ₐ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_sp,[1]).vals[1])'
    Φ̃ᵉ = spzeros(N*n.θ,N*n.θ)
    Φ̃ᵉₐ = spzeros(N*n.θ,N*n.θ)
    
    for s in 1:n.θ
        for s′ in 1:n.θ
            #b′ = R̄*bgrid .+ ϵ[s]*W̄ .- cf[s](bgrid) #asset choice
            a′ = af(a_sp,θ[s])
            Φ̃ᵉ[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πθ[s,s′]*BasisMatrix(abasis,Direct(),a′).vals[1]
            Φ̃ᵉₐ[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πθ[s,s′]*BasisMatrix(abasis,Direct(),a′,[1]).vals[1]
        end
    end
    #Recall our First order code assumes these are transposed
    Φ̃ᵉ = Φ̃ᵉ'
    Φ̃ᵉₐ = (Φ̃ᵉₐ)'  

    Φ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_Ω).vals[1])' #note transponse again
    Φₐ =kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_Ω,[1]).vals[1])' #note transponse again
    return Φ̃,Φ̃ₐ,Φ̃ᵉ,Φ̃ᵉₐ,Φ,Φₐ
end 


