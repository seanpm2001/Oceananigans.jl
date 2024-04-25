#####
##### Centered advection scheme
#####

"""
    struct Centered{N, FT, XT, YT, ZT, CA} <: AbstractCenteredAdvectionScheme{N, FT}

Centered reconstruction scheme.
"""
struct Centered{N, FT, XT, YT, ZT, CA, D} <: AbstractCenteredAdvectionScheme{N, D, FT}
    "coefficient for Centered reconstruction on stretched ``x``-faces" 
    coeff_xᶠᵃᵃ :: XT
    "coefficient for Centered reconstruction on stretched ``x``-centers"
    coeff_xᶜᵃᵃ :: XT
    "coefficient for Centered reconstruction on stretched ``y``-faces"
    coeff_yᵃᶠᵃ :: YT
    "coefficient for Centered reconstruction on stretched ``y``-centers"
    coeff_yᵃᶜᵃ :: YT
    "coefficient for Centered reconstruction on stretched ``z``-faces"
    coeff_zᵃᵃᶠ :: ZT
    "coefficient for Centered reconstruction on stretched ``z``-centers"
    coeff_zᵃᵃᶜ :: ZT

    "advection scheme used near boundaries"
    buffer_scheme :: CA

    function Centered{N, FT, D}(coeff_xᶠᵃᵃ::XT, coeff_xᶜᵃᵃ::XT,
                                coeff_yᵃᶠᵃ::YT, coeff_yᵃᶜᵃ::YT, 
                                coeff_zᵃᵃᶠ::ZT, coeff_zᵃᵃᶜ::ZT,
                                buffer_scheme::CA) where {N, FT, XT, YT, ZT, CA, D}

        return new{N, FT, XT, YT, ZT, CA, D}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, 
                                             coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, 
                                             coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ,
                                             buffer_scheme)
    end
end

function Centered(FT::DataType = Float64; 
                  grid = nothing, 
                  order = 2,
                  divergent_branches = true) 

    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N  = Int(order ÷ 2)
    if N > 1 
        coefficients  = Tuple(nothing for i in 1:6)
        # Stretched coefficient seem to be more unstable that constant spacing ones for centered reconstruction.
        # Some tests are needed to verify why this is the case (and if it is expected). We keep constant coefficients for the moment
        # coefficients = compute_reconstruction_coefficients(grid, FT, :Centered; order)
        buffer_scheme = Centered(FT; grid, order = order - 2)
    else
        coefficients    = Tuple(nothing for i in 1:6)
        buffer_scheme = nothing
    end
    return Centered{N, FT, divergent_branches}(coefficients..., buffer_scheme)
end

const    DivergentCentered{N, FT, XT, YT, ZT, CA} = Centered{N, FT, XT, YT, ZT, CA, true}  where {N, FT, XT, YT, ZT, CA}
const NonDivergentCentered{N, FT, XT, YT, ZT, CA} = Centered{N, FT, XT, YT, ZT, CA, false} where {N, FT, XT, YT, ZT, CA}

Base.summary(a::Centered{N}) where N = string("Centered reconstruction order ", N*2)

Base.show(io::IO, a::Centered{N, FT, XT, YT, ZT}) where {N, FT, XT, YT, ZT} =
    print(io, summary(a), " \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.buffer_scheme), "\n",
              " Directions:", "\n",
              "    ├── X $(XT == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(YT == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(ZT == Nothing ? "regular" : "stretched")" )


Adapt.adapt_structure(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                    Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                    Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                    Adapt.adapt(to, scheme.buffer_scheme))

on_architecture(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(on_architecture(to, scheme.coeff_xᶠᵃᵃ), on_architecture(to, scheme.coeff_xᶜᵃᵃ),
                    on_architecture(to, scheme.coeff_yᵃᶠᵃ), on_architecture(to, scheme.coeff_yᵃᶜᵃ),
                    on_architecture(to, scheme.coeff_zᵃᵃᶠ), on_architecture(to, scheme.coeff_zᵃᵃᶜ),
                    on_architecture(to, scheme.buffer_scheme))

# Useful aliases
Centered(grid, FT::DataType=Float64; kwargs...) = Centered(FT; grid, kwargs...)

CenteredSecondOrder(grid=nothing, FT::DataType=Float64) = Centered(grid, FT; order=2)
CenteredFourthOrder(grid=nothing, FT::DataType=Float64) = Centered(grid, FT; order=4)

const ACAS = AbstractCenteredAdvectionScheme

# left and right biased for Centered reconstruction are just symmetric!
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::ACAS, c, args...) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c, args...)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::ACAS, c, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c, args...)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::ACAS, c, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::ACAS, c, args...) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c, args...)
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::ACAS, c, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c, args...)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::ACAS, c, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c, args...)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::ACAS, u, args...) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::ACAS, v, args...) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::ACAS, w, args...) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::ACAS, u, args...) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::ACAS, v, args...) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::ACAS, w, args...) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w, args...)

# uniform centered reconstruction
for buffer in advection_buffers
    @eval begin
        @inline inner_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, <:Nothing}, ψ, idx, loc, args...)           where FT = @inbounds $(calc_reconstruction_stencil(buffer, :symmetric, :x, false))
        @inline inner_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, <:Nothing}, ψ::Function, idx, loc, args...) where FT = @inbounds $(calc_reconstruction_stencil(buffer, :symmetric, :x,  true))
    
        @inline inner_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, <:Nothing}, ψ, idx, loc, args...)           where {FT, XT} = @inbounds $(calc_reconstruction_stencil(buffer, :symmetric, :y, false))
        @inline inner_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT} = @inbounds $(calc_reconstruction_stencil(buffer, :symmetric, :y,  true))
    
        @inline inner_symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, YT, <:Nothing}, ψ, idx, loc, args...)           where {FT, XT, YT} = @inbounds $(calc_reconstruction_stencil(buffer, :symmetric, :z, false))
        @inline inner_symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, YT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT, YT} = @inbounds $(calc_reconstruction_stencil(buffer, :symmetric, :z,  true))
    end
end

# stretched centered reconstruction
for (dir, ξ, val) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ), (:x, :y, :z), (1, 2, 3))
    stencil = Symbol(:inner_symmetric_interpolate_, dir)

    for buffer in advection_buffers
        @eval begin
            @inline $stencil(i, j, k, grid, scheme::Centered{$buffer, FT}, ψ, idx, loc, args...)           where FT = @inbounds sum($(reconstruction_stencil(buffer, :symmetric, ξ, false)) .* retrieve_coeff(scheme, Val($val), idx, loc))
            @inline $stencil(i, j, k, grid, scheme::Centered{$buffer, FT}, ψ::Function, idx, loc, args...) where FT = @inbounds sum($(reconstruction_stencil(buffer, :symmetric, ξ,  true)) .* retrieve_coeff(scheme, Val($val), idx, loc))
        end
    end
end

# Retrieve precomputed coefficients 
@inline retrieve_coeff(scheme::Centered, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[i] 
