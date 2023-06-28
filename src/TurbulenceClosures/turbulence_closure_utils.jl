using Oceananigans.Operators

const PossibleDiffusivity = Union{Number, Function, DiscreteDiffusionFunction, AbstractArray}

tracer_diffusivities(tracers, κ::PossibleDiffusivity) = with_tracers(tracers, NamedTuple(), (tracers, init) -> κ)
tracer_diffusivities(tracers, ::Nothing) = nothing

function tracer_diffusivities(tracers, κ::NamedTuple)

    all(name ∈ propertynames(κ) for name in tracers) ||
        throw(ArgumentError("Tracer diffusivities or diffusivity parameters must either be a constants
                            or a `NamedTuple` with a value for every tracer!"))

    return κ
end

convert_diffusivity(FT, κ::Number; kw...) = convert(FT, κ)

function convert_diffusivity(FT, κ; discrete_form=false, loc=(nothing, nothing, nothing), parameters=nothing)
    discrete_form && return DiscreteDiffusionFunction(κ; loc, parameters)
    return κ
end
    
function convert_diffusivity(FT, κ::NamedTuple; discrete_form=false, loc=(nothing, nothing, nothing), parameters=nothing)
    κ_names = propertynames(κ)
    return NamedTuple{κ_names}(Tuple(convert_diffusivity(FT, κi; discrete_form, loc, parameters) for κi in κ))
end

@kernel function calculate_nonlinear_viscosity!(νₑ, grid, closure, buoyancy, velocities, tracers) 
    i, j, k = @index(Global, NTuple)
    @inbounds νₑ[i, j, k] = calc_nonlinear_νᶜᶜᶜ(i, j, k, grid, closure, buoyancy, velocities, tracers)
end

@kernel function calculate_nonlinear_tracer_diffusivity!(κₑ, grid, closure, tracer, tracer_index, U)
    i, j, k = @index(Global, NTuple)
    @inbounds κₑ[i, j, k] = calc_nonlinear_κᶜᶜᶜ(i, j, k, grid, closure, tracer, tracer_index, U)
end

# extend κ kernel to compute also the boundaries
@inline function κ_kernel_size(grid, ::AbstractTurbulenceClosure{TD, B}) where{TD, B}
    Nx, Ny, Nz = size(grid)
    Tx, Ty, Tz = topology(grid)

    Ax = Tx == Flat ? Nx : Nx + 2B 
    Ay = Ty == Flat ? Ny : Ny + 2B 
    Az = Tz == Flat ? Nz : Nz + 2B

    return (Ax, Ay, Az)
end

@inline function κ_kernel_offsets(grid, ::AbstractTurbulenceClosure{TD, B}) where{TD, B}
    Tx, Ty, Tz = topology(grid)

    Ax = Tx == Flat ? 0 : - B
    Ay = Ty == Flat ? 0 : - B 
    Az = Tz == Flat ? 0 : - B

    return (Ax, Ay, Az)
end
