using LoopVectorization, Plots, Tullio

function find_analytical_C_r(r, η)
    """
        Finds the direct correlation function given by the 
        analytical percus yevick solution of the Ornstein Zernike 
        equation for hard spheres for a given volume fraction η on the coordinates r
        in units of one over the diameter of the particles
    """ 
    C = -(1 - η)^-4 * ((1 + 2η)^2 .- 6η * (1 + η/2)^2*r .+ 1/2 * η*(1 + 2η)^2 * r.^3)
    C[r.>1] .= 0
    return C
end

function find_analytical_C_k(k, η)
    """
        Finds the fourier transform of the direct correlation function given by the 
        analytical percus yevick solution of the Ornstein Zernike 
        equation for hard spheres for a given volume fraction η on the coordinates r
        in units of one over the diameter of the particles
    """ 
    A = -(1 - η)^-4 *(1 + 2η)^2
    B = (1 - η)^-4*  6η*(1 + η/2)^2
    D = -(1 - η)^-4 * 1/2 * η*(1 + 2η)^2
    Cₖ = @. 4π/k^6 * (24*D - 2*B * k^2 - (24*D - 2 * (B + 6*D) * k^2 + (A + B + D) * k^4) * cos(k) + k * (-24*D + (A + 2*B + 4*D) * k^2) * sin(k))
    return Cₖ
end


function find_analytical_S_k(k, η)
    """
        Finds the static structure factor given by the 
        analytical percus yevick solution of the Ornstein Zernike 
        equation for hard spheres for a given volume fraction η on the coordinates r
        in units of one over the diameter of the particles
    """ 
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

function plotCk(ρ)
    k = collect(LinRange(0, 60, 1000))
    η = π/6 * ρ
    Ck = find_analytical_C_k(k, η)
    p = plot(k, find_analytical_C_k(k, η))
    return p
end

function plotSk(ρ)
    k = collect(LinRange(0, 60, 1000))
    η = π/6 * ρ
    Ck = find_analytical_C_k(k, η)
    p = plot(k, find_analytical_S_k(k, η))
    return p
end

# plotCk(0.9)


