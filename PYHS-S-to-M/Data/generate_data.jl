using Plots, Dierckx, DelimitedFiles, Printf
include("PercusYevick.jl")
include("ModeCouplingTimeScheme.jl")
cd(@__DIR__)

Nk = 100
k_array = collect(LinRange(0.2, 39.8, Nk))
open("k_array.txt", "w") do io
    writedlm(io, k_array)
end

η_crit = 0.5159121289383616

Nη = 10^5
σ_η = 0.04
η_array = σ_η * randn(Nη) .+ η_crit

println("Generated $Nη values of the volume fraction betweel η = $(minimum(η_array)) and η = $(maximum(η_array))")
p1 = plot()
p2 = plot()

Nt = 1000
i = 0
for η in η_array
    i += 1
    log_time_sample_array = rand(Nt).*20 .- 10
    Sk = find_analytical_S_k.(k_array, η)
    t_array, ϕ = find_intermediate_scattering_function(η)
    log_time_array = log10.(t_array)
    ϕ = ϕ[19, :]
    iterpolation_spline = Spline1D(log_time_array, ϕ)
    ϕ_sample_array = iterpolation_spline.(log_time_sample_array)

    if i % 100 == 0
        scatter!(p1, log_time_sample_array, ϕ_sample_array, ms=0.4, color=:black)
        plot!(p2, log_time_array, ϕ)
    end

    open(@sprintf("Sk_data\\Sk_eta_%.10f.txt",η), "w") do io
        writedlm(io, Sk)
    end
    
    open(@sprintf("phi_data\\phi_eta_%.10f.txt",η), "w") do io
        writedlm(io, [log_time_sample_array ϕ_sample_array])
    end
end
plot!(p1, legend=false)
plot!(p2, legend=false)
display(p1)
display(p2)
savefig(p1, "phi_randomly_sampled.png")
savefig(p2, "phi.png")