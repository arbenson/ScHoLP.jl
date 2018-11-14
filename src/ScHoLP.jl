module ScHoLP

using Base.Threads
using Combinatorics
using DataStructures
using DelimitedFiles
using Krylov
using LinearAlgebra
using LinearOperators
using Printf
using SparseArrays
using StatsBase

# This needs to go first
include("util.jl")

# Computations of various dataset statistics
include("simplicial_closure_probs.jl")
include("summary_statistics.jl")
include("temporal_asynchronicity.jl")

# Triangle prediction
include("walk_scores.jl")
include("local_scores.jl")
include("simplicial_ppr_scores.jl")

# Lifecycle analysis
include("lifecycles.jl")

end # module
