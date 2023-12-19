using Pkg
Pkg.activate("d:/julialy") # PATH
using LinearAlgebra
using FFTW
using DataStructures
using MAT
using CairoMakie
using MakiePublication

function read_structure_detail(filename)
    mat = matopen(filename)
    M = read(mat, "M") #M
    edges = read(mat, "edges") #a,b,c
    angles = read(mat, "angles") #α,β,γ
    wfields = read(mat, "wfields") #
    ϕfields = read(mat, "component") #
    close(mat)
    return[ϕfields, wfields, edges, angles, M]
end

struct Hexagonal2D
end
struct Orthorhombic
end
struct Tetragonal
end
struct Cubic
end

function candidates(::Hexagonal2D)
    return ["HEX"]
end

function candidates(::Orthorhombic)
    return ["SPC", "O70", "HCP_orthogonal"]
end

function candidates(::Tetragonal)
    return ["Sigma"]
end

function candidates(::Cubic)
    return ["LAM", "LAMs", "SC", "BCC", "FCC", "GYR", "A15", "SPC", "SG", "DD", "PL"]
end

function FFT_new(input, weight::Array; threshold=1, max_index=12, standard_M=nothing)
    if isa(input, String) 
        data, M = read_structure_detail(input)[1], read_structure_detail(input)[5]
    elseif isa(input, NoncyclicChainSCFT)
        data = input.ϕfields
        M = Scattering.shape(unitcell(input)).M
    end

    if standard_M != nothing
        M = standard_M
    end

    length(data) == 3 ? arrayd = data[1]*weight[1] + data[2]*weight[2] + data[3]*weight[3] : arrayd = data[1]*weight[1] + data[2]*weight[2]

    fft2d = false
    if length(size(arrayd)) == 2
        n1, n2 = size(arrayd)
        n3 = 1
        fft2d = true
    elseif length(size(arrayd)) == 3
        n1, n2, n3 = size(arrayd)
    end

    F_density_matrix=abs2.(fft(arrayd))
    output = Array{Any}(undef, n1*n2*n3, 4)
    k=1

    for i = 1:size(output, 1)
        output[i, 1:2] = [0.0, 0.0]
        output[i, 3:4] = [[] for j = 1:2]
    end

    m1 = n1 >= max_index ? max_index : n1
    m2 = n2 >= max_index ? max_index : n2
    
    if n3 != 1
        m3 = n3 >= max_index ? max_index : n3
    elseif n3 == 1
        m3 = 1
    end
    
    for i = 0:m1-1, j = 0:m2-1, l = 0:m3-1

        if fft2d 
            x,y = M * [i,j]
            z=0 
        elseif !fft2d 
            x,y,z = M * [i,j,l]
        end

        index = sqrt(x^2 + y^2 + z^2)
        value = F_density_matrix[i+1,j+1,l+1]
        hkl = [i,j,l]
        found = false

        for m = 1:k-1
            if abs.(output[m,1]-index) <= 0.001*output[m,1]
                output[m,2] += value
                push!(output[m,3],hkl)
                push!(output[m,4],value)
                found = true
                break
            end
        end
        
        if !found
            output[k,1] = index
            output[k,2] = value
            push!(output[k,3],hkl)
            push!(output[k,4],value)
            k += 1
        end
    end
    
    for i = 1:size(output, 1)
        if log10(output[i,2]) < threshold
            output[i,2] = 0.0
        end
    end

    output = output[2:k-1,:] #第一个是000
    sorted_indices = sortperm(output[:,1])
    output = output[sorted_indices,:]
    
    for i in 1:size(output,1)
        sum_val = sum(output[i,4])
        contributions = [(x/sum_val) for x in output[i,4]]
        output[i,4] = [contributions[p] for p in 1:length(contributions)]
    end
    
    return output
end

function stage1(input, reference; system=Cubic(), Nt=10)
    candidate_list = candidates(system)
    stage1_dict = Dict()

    for phase in candidate_list
        match = 0
        phase_pattern = reference[phase][1]
        M = reference[phase][2]
        input_pattern = FFT_new(input, [1.0, 0.0, 0.0]; standard_M=M)[:, 2:3]
        threshold = size(phase_pattern, 1) > Nt ? Nt : size(phase_pattern, 1)
        for i in 1:threshold
            input_label = log10(input_pattern[i, 1]) > 1 ? 1 : 0
            input_set = input_pattern[i, 2]
            reference_label, reference_set = phase_pattern[i, :]
            if input_label == reference_label && reference_label == 1 && issubset(input_set, reference_set)
                match += 1
            elseif input_label == reference_label && reference_label == 0
                match += 1
            else
                break
            end
        end
        stage1_dict[phase] = match
    end

    return sort(collect(stage1_dict), by = x -> x[2], rev = true)
end

function stage2(input, reference; system=Cubic(), limit=20) # referce = phase_library
    candidate_list = candidates(system)
    stage2_dict = Dict()

    for phase in candidate_list
        pattern = reference[phase][1]
        M = reference[phase][2]
        indices = Int[]
        output = FFT_new(input, [1.0, 0.0, 0.0]; standard_M=M)
        for i in 1:size(output,1)
            if output[i,2] > 0.0
                push!(indices, i)
            end
        end
        q_exp = output[indices, 1]

        q_theorical = []
        q_exp2 = copy(q_exp)
    
        if size(pattern,1) < limit
            limit = size(pattern,1)
        end
        
        for i in 1:limit
            if pattern[i,1] > 0.0
                if size(M) == (2,2)
                    q = √(sum((M * pattern[i,2][1][1:2]).^2))
                elseif size(M) == (3,3)
                    q = √(sum((M * pattern[i,2][1]).^2))
                end

                push!(q_theorical, q)
            end
        end
        

        if length(q_exp2) >= length(q_theorical)
            q_exp2 = q_exp2[1:length(q_theorical)]
        else
            q_theorical = q_theorical[1:length(q_exp2)]
        end

        q3 = abs.(q_exp2.^(-1) - q_theorical.^(-1))
        norm = √(sum(q3.^2))/length(q3)
        stage2_dict[phase] = norm
    end
    return sort(collect(stage2_dict), by = x -> x[2], rev = false)
end

function modified_variance(array)
    var = 0.0
    number = 0
    average = sum(array)/length(array)
    for i in array
        if i > average
            var += (i/average-1.0)^2
            number += 1
        end
    end
    return var/number
end