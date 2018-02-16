export HONData, SpIntMat, SpFltMat

"""
SpIntMat
--------

const SpIntMat = SparseMatrixCSC{Float64,Int64}
"""
const SpIntMat = SparseMatrixCSC{Int64,Int64}

"""
SpFltMat
--------

const SpFltMat = SparseMatrixCSC{Float64,Int64}
"""
const SpFltMat = SparseMatrixCSC{Float64,Int64}

"""
HONData
-------

Data structure for a temporal higher-order network.

Each dataset consists of three integer vectors: simplices, nverts, and times. 
- The simplices is a contiguous vector of nodes comprising the simplices. 
- The nverts vector contains the number of vertices within each simplex. 
- The times vector contains the timestamps of the simplices (same length as nverts).

For example, consider a dataset consisting of three simplices:

    1. {1, 2, 3} at time 10
    2. {2, 4} at time 15.
    3. {1, 3, 4, 5} at time 21.

Then the data structure would be  
- simplices = [1, 2, 3, 2, 4, 1, 3, 4, 5]
- nverts = [3, 2, 4]
- times = [10, 15, 21]
There is an additional name variable attached to the dataset.
"""
immutable HONData
    simplices::Vector{Int64}
    nverts::Vector{Int64}
    times::Vector{Int64}
    name::String
end

sorted_tuple(a::Int64, b::Int64, c::Int64) =
    NTuple{3, Int64}(sort([a, b, c], alg=InsertionSort))
sorted_tuple(a::Int64, b::Int64, c::Int64, d::Int64) =
    NTuple{4, Int64}(sort([a, b, c, d], alg=InsertionSort))

function read_txt_data(dataset::String)
    read(filename::String) = convert(Vector{Int64}, readdlm(filename, Int64)[:, 1])
    return HONData(read("data/$(dataset)/$(dataset)-simplices.txt"),
                   read("data/$(dataset)/$(dataset)-nverts.txt"),
                   read("data/$(dataset)/$(dataset)-times.txt"),
                   dataset)
end

function read_node_labels(dataset::String)
    if dataset[end-2:end] == "-25"
        dataset = dataset[1:end-3]
    end
    labels = Vector{String}()
    open("data/$(dataset)/$(dataset)-node-labels.txt") do f
        for line in eachline(f)
            push!(labels, join(split(line)[2:end], " "))
        end
    end
    return labels
end

function read_simplex_labels(dataset::String)
    if dataset[end-2:end] == "-25"
        dataset = dataset[1:end-3]
    end
    labels = Vector{String}()
    open("data/$(dataset)/$(dataset)-simplex-labels.txt") do f
        for line in eachline(f)
            push!(labels, join(split(line)[2:end], " "))
        end
    end
    return labels
end

function read_closure_stats(dataset::AbstractString, simplex_size::Int64, initial_cutoff::Int64=100)
    keys = []
    probs, nsamples, nclosed = Float64[], Int64[], Int64[]
    data = readdlm("output/$(dataset)-$(simplex_size)-node-closures.txt")
    if initial_cutoff < 100
        data = readdlm("output/$(dataset)-$(simplex_size)-node-closures-$(initial_cutoff).txt")
    end
    for row_ind in 1:size(data, 1)
        row = convert(Vector{Int64}, data[row_ind, :])
        push!(keys, (row[1:simplex_size]...))
        push!(nsamples, row[end - 1])
        push!(nclosed, row[end])
    end
    return (keys, nsamples, nclosed)
end

function bipartite_graph(simplices::Vector{Int64}, nverts::Vector{Int64})
    if length(simplices) == 0
        return convert(SpIntMat, sparse([], [], []))
    end
    I, J = Int64[], Int64[]
    curr_ind = 1
    for (simplex_ind, nv) in enumerate(nverts)
        for vert in simplices[curr_ind:(curr_ind + nv - 1)]
            push!(I, vert)
            push!(J, simplex_ind)
        end
        curr_ind += nv
    end
    return convert(SpIntMat, sparse(I, J, ones(length(I)), maximum(simplices), length(nverts)))
end

function basic_matrices(simplices::Vector{Int64}, nverts::Vector{Int64})
    A = bipartite_graph(simplices, nverts)
    At = A'
    B = A * At
    B -= spdiagm(diag(B))  # projected graph (no diagonal)
    return (A, At, B)
end
basic_matrices(dataset::HONData) =
    basic_matrices(dataset.simplices, dataset.nverts)

nz_row_inds(A::SpIntMat, ind::Int64) = A.rowval[A.colptr[ind]:(A.colptr[ind + 1] - 1)]
nz_row_inds(A::SpFltMat, ind::Int64) = A.rowval[A.colptr[ind]:(A.colptr[ind + 1] - 1)]
nz_row_vals(A::SpIntMat, ind::Int64) = A.nzval[A.colptr[ind]:(A.colptr[ind + 1] - 1)]
nz_row_vals(A::SpFltMat, ind::Int64) = A.nzval[A.colptr[ind]:(A.colptr[ind + 1] - 1)]

function triangle_closed(A::SpIntMat, At::SpIntMat, order::Vector{Int64},
                         i::Int64, j::Int64, k::Int64)
    ind, nbr1, nbr2 = sort([i, j, k], by=(v -> order[v]), alg=InsertionSort)
    # Search all simplices of least common vertex
    for simplex_id in nz_row_inds(At, ind)
        if A[nbr1, simplex_id] > 0 && A[nbr2, simplex_id] > 0
            return true
        end
    end
    return false
end

function tetrahedron_closed(A::SpIntMat, At::SpIntMat, order::Vector{Int64},
                            i::Int64, j::Int64, k::Int64, l::Int64)
    ind, nbr1, nbr2, nbr3 = sort([i, j, k, l], by=(v -> order[v]), alg=InsertionSort)
    # Search all simplices of least common vertex
    for simplex_id in nz_row_inds(At, ind)
        if A[nbr1, simplex_id] > 0 && A[nbr2, simplex_id] > 0 && A[nbr3, simplex_id] > 0
            return true
        end
    end
    return false
end

function neighbors(B::SpIntMat, order::Vector{Int64}, node::Int64)
    node_order = order[node]
    return filter(nbr -> order[nbr] > node_order, nz_row_inds(B, node))
end

neighbor_pairs(B::SpIntMat, order::Vector{Int64}, node::Int64) =
    combinations(neighbors(B, order, node), 2)

# Ordering of nodes by the number of simplices in which they appear
function simplex_degree_order(At::SpIntMat)
    n = size(At, 2)
    simplex_order = zeros(Int64, n)
    simplex_order[sortperm(vec(sum(At, 1)))] = collect(1:n)
    return simplex_order
end

# Ordering of nodes by their degree
function proj_graph_degree_order(B::SpIntMat)
    n = size(B, 1)
    triangle_order = zeros(Int64, n)
    triangle_order[sortperm(vec(sum(spones(B), 1)))] = collect(1:n)
    return triangle_order
end

function num_open_closed_triangles(A::SpIntMat, At::SpIntMat, B::SpIntMat)
    simp_order = simplex_degree_order(At)
    tri_order = proj_graph_degree_order(B)
    n = size(B, 2)
    counts = zeros(Int64, 2, Threads.nthreads())
    Threads.@threads for i = 1:n
        for (j, k) in neighbor_pairs(B, tri_order, i)
            if B[j, k] > 0
                tid = Threads.threadid()
                closed = triangle_closed(A, At, simp_order, i, j, k)
                counts[1 + closed, tid] += 1
            end
        end
    end
    return (sum(counts, 2)...)
end

# Get the subset of data in interval [start_time, end_time]
function window_data(start_time::Int64, end_time::Int64, simplices::Vector{Int64},
                     nverts::Vector{Int64}, times::Vector{Int64})
    curr_ind = 1
    window_simplices, window_nverts, window_times = Int64[], Int64[], Int64[]
    for (nv, time) in zip(nverts, times)
        end_ind = curr_ind + nv - 1
        if time >= start_time && time <= end_time
            push!(window_nverts, nv)
            push!(window_times, time)
            append!(window_simplices, simplices[curr_ind:end_ind])
        end
        curr_ind += nv
    end
    return (window_simplices, window_nverts, window_times)
end

# Split data by timestamps into quantiles specified by percentile1 and
# percentile2. Returns a 4-tuple (old_simps, old_nverts, new_simps, new_nverts),
# where (old_simps, old_nverts) are the data in the quantile [0, percentile1]
# and (new_simps, new_nverts) are the data in the quantile (percentile1, percentile2].
function split_data(simplices::Vector{Int64}, nverts::Vector{Int64},
                    times::Vector{Int64}, percentile1::Int64,
                    percentile2::Int64)
    assert(percentile1 <= percentile2)
    cutoff(prcntl::Int64) = convert(Int64, round(percentile(times, prcntl)))

    cutoff1 = cutoff(percentile1)
    old_simps, old_nverts =
        window_data(minimum(times), cutoff1, simplices, nverts, times)[1:2]

    cutoff2 = (percentile2 == 100) ? (maximum(times) + 1) : cutoff(percentile2)
    new_simps, new_nverts =
        window_data(cutoff1 + 1, cutoff2, simplices, nverts, times)[1:2]

    return old_simps, old_nverts, new_simps, new_nverts
end


function backbone(simplices::Vector{Int64}, nverts::Vector{Int64},
                  times::Vector{Int64})
    # backbone data
    bb_simplices, bb_nverts, bb_times = Int64[], Int64[], Int64[]
    
    # contains for all simplices
    max_size = maximum(nverts)
    all_simplices = Vector{Set}(max_size)
    for i in 1:max_size; all_simplices[i] = Set{Any}(); end

    curr_ind = 1
    for (nvert, time) in zip(nverts, times)
        simplex = simplices[curr_ind:(curr_ind + nvert - 1)]
        nvert = length(simplex)
        # Add to data if we have not seen it yet
        if !(simplex in all_simplices[nvert])
            push!(all_simplices[nvert], simplex)
            # add to backbone
            append!(bb_simplices, simplex)
            push!(bb_nverts, nvert)
            push!(bb_times, time)
        end
        curr_ind += nvert
    end

    return (bb_simplices, bb_nverts, bb_times)
end

function data_size_cutoff(simplices::Vector{Int64}, nverts::Vector{Int64},
                          times::Vector{Int64}, min_simplex_size::Int64,
                          max_simplex_size::Int64)
    new_simplices = Int64[]
    new_nverts = Int64[]
    new_times = Int64[]    
    curr_ind = 1
    for (nv, time) in zip(nverts, times)
        if min_simplex_size <= nv <= max_simplex_size
            append!(new_simplices, simplices[curr_ind:(curr_ind + nv - 1)])
            push!(new_nverts, nv)
            push!(new_times, time)
        end
        curr_ind += nv
    end
    return (new_simplices, new_nverts, new_times)
end

function data_size_cutoff(simplices::Vector{Int64}, nverts::Vector{Int64},
                          times::Vector{Int64}, cutoff::Int64)
    return data_size_cutoff(simplices, nverts, times, 0, cutoff)
end

# Get a configuration
function configuration(simplices::Vector{Int64}, nverts::Vector{Int64})
    while true
        config = shuffle(simplices)
        valid_config = true
        curr_ind = 1
        for nv in nverts
            simplex = config[curr_ind:(curr_ind + nv - 1)]
            if length(unique(simplex)) != nv
                # Invalid configuration --> start over
                valid_config = false
                break
            end
            curr_ind += nv
        end
        if valid_config; return config; end
    end
end

# Get a configuration that preserves the number of k-vertex simplices that every
# vertex participates in.
function configuration_sizes_preserved(simplices::Vector{Int64},
                                       nverts::Vector{Int64})
    config = zeros(Int64, length(simplices))
    app = copy(nverts)
    unshift!(app, 1)
    capp = cumsum(app)
    for val in unique(nverts)
        # Get the simplices with this number of vertices
        inds = Int64[]
        cnt = 0
        for ind in find(nverts .== val)
            append!(inds, capp[ind]:(capp[ind] + val  - 1))
            cnt += 1
        end
        # TODO(arbenson): this isn't creating a simple graph necessarily
        config[inds] = shuffle(simplices[inds])
    end
    return config
end

# This is just a convenient wrapper around all of the formatting parameters for
# making plots.
function all_datasets_plot_params()
    green  = "#1b9e77"
    orange = "#d95f02"
    purple = "#7570b3"
    plot_params = [["coauth-DBLP-25",            "coauth-DBLP",            "x", green],
                   ["coauth-MAG-Geology-25",     "coauth-MAG-Geology",     "x", orange],
                   ["coauth-MAG-History-25",     "coauth-MAG-History",     "x", purple],
                   ["music-rap-genius-25",       "music-rap-genius",       "v", green],
                   ["tags-stack-overflow",       "tags-stack-overflow",    "s", green],
                   ["tags-math-sx",              "tags-math-sx",           "s", orange],
                   ["tags-ask-ubuntu",           "tags-ask-ubuntu",        "s", purple],
                   ["threads-stack-overflow-25", "threads-stack-overflow", "o", green],
                   ["threads-math-sx",           "threads-math-sx",        "o", orange],
                   ["threads-ask-ubuntu",        "threads-ask-ubuntu",     "o", purple],
                   ["NDC-substances-25",         "NDC-substances",         "<", green],
                   ["NDC-classes-25",            "NDC-classes",            "<", orange],
                   ["DAWN",                      "DAWN",                   "p", green],
                   ["congress-bills-25",         "congress-bills",         "*", green],
                   ["congress-committees-25",    "congress-committees",    "*", orange],
                   ["email-Eu-25",               "email-Eu",               "P", green],
                   ["email-Enron-25",            "email-Enron",            "P", orange],
                   ["contact-high-school",       "contact-high-school",    "d", green],
                   ["contact-primary-school",    "contact-primary-school", "d", orange],
                   ]
    return plot_params
end
