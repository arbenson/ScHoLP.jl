"""
summary_statistics
------------------------

summary_statistics(dataset::HONData)

Computes several statistics about the dataset.

dataset::HONData
    The dataset.
"""
function summary_statistics(dataset::HONData)
    function _summary_statistics(simplices::Vector{Int64}, nverts::Vector{Int64}, times::Vector{Int64})
        if length(nverts) == 0
            return ("0,0,0,0,0,0.0,0.0,0,0,0,0,0.0,0.0,0.0,0.0,0,0")
        end

        A, At, B = basic_matrices(simplices, nverts)
        no, nc = num_open_closed_triangles(A, At, B)
        num_nodes = sum(sum(At, 1) .> 0)  # note: includes 1-node simplices
        density = nnz(B) / (num_nodes^2 - num_nodes)
        str1 = @sprintf("%d,%d,%d,%d,%d,%f,%e,%d,%d",
                        num_nodes, length(nverts), nnz(A), mean(nverts),
                        maximum(nverts), mean(nonzeros(B)), density, nc, no)

        # Backbone dataset
        bb_simplices, bb_nverts, bb_times = backbone(simplices, nverts, times)
        (A, At, C) = basic_matrices(bb_simplices, bb_nverts)
        str2 = @sprintf("%d,%d,%f,%f",
                        length(bb_nverts), nnz(A), mean(bb_nverts), mean(nonzeros(C)))

        # Random configuration
        config = configuration_sizes_preserved(bb_simplices, bb_nverts)
        (A, At, B) = basic_matrices(config, bb_nverts)
        num_nodes = sum(sum(At, 1) .> 0)
        density = nnz(B) / (num_nodes^2 - num_nodes)        
        no, nc = num_open_closed_triangles(A, At, B)
        str3 = @sprintf("%f,%e,%d,%d",
                        mean(nonzeros(B)), density, nc, no)
        return "$str1,$str2,$str3"
    end

    dataset_name = dataset.name
    open("$(dataset_name)-statistics.csv", "w") do f
        write(f, "dataset,")
        write(f, "nnodes,nsimps,nconns,meansimpsize,maxsimpsize,meanprojweight,projdensity,nclosedtri,nopentri,")
        write(f, "nbbnodes,nbbconns,meanbbsimpsize,meanbbprojweight,")
        write(f, "meanbbconfigweight,bbconfigdensity,nbbconfigclosedtri,nbbconfigopentri\n")
        simplices, nverts, times = dataset.simplices, dataset.nverts, dataset.times
        stats = _summary_statistics(simplices, nverts, times)
        write(f, "$(dataset_name),$stats\n")

        # Get the same statistics but for the dataset restricted to 3-node
        # simplices.
        csimplices, cnverts, ctimes = data_size_cutoff(simplices, nverts, times, 3, 3)
        stats = _summary_statistics(csimplices, cnverts, ctimes)
        write(f, "$(dataset_name)-3-3,$stats\n")
    end
end

"""
summary_statistics
------------------------

summary_statistics(dataset::String)

Computes several statistics about the dataset.

dataset::String
    The dataset name.
"""
summary_statistics(dataset::String) = summary_statistics(read_txt_data(dataset))

"""
basic_summary_statistics
------------------------

basic_summary_statistics(dataset::String)

Prints some basic summary statistics of the dataset.

dataset::HONData
    The dataset.
"""
function basic_summary_statistics(dataset::HONData)
    simplices, nverts, times = dataset.simplices, dataset.nverts, dataset.times
    bb_simplices, bb_nverts, bb_times = backbone(simplices, nverts, times)
    A, At, B = basic_matrices(simplices, nverts)
    num_nodes = sum(sum(At, 1) .> 0)
    num_edges = nnz(B) / 2
    num_simps = length(nverts)
    num_bb_simps = length(bb_nverts)
    println("dataset & # nodes & # edges in proj. graph & # simplices & # unique simplices")
    println(@sprintf("%s & %d & %d & %d & %d", dataset.name, num_nodes, num_edges, num_simps, num_bb_simps))
end

"""
basic_summary_statistics
------------------------

basic_summary_statistics(dataset::String)

Prints some basic summary statistics of the dataset.

dataset::String
    The dataset name.
"""
basic_summary_statistics(dataset::String) =
    basic_summary_statistics(read_txt_data(dataset))
