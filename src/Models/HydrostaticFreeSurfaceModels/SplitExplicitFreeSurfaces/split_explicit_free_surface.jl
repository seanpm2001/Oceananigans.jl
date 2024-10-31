"""
    struct SplitExplicitFreeSurface

The split-explicit free surface solver.

$(FIELDS)
"""
struct SplitExplicitFreeSurface{H, U, M, FT, K , S, T} <: AbstractFreeSurface{H, FT}
    η :: H
    barotropic_velocities :: U # A namedtuple with U, V 
    filtered_state :: M # A namedtuple with η, U, V averaged throughout the substepping
    gravitational_acceleration :: FT
    kernel_parameters :: K
    substepping :: S  # Either `FixedSubstepNumber` or `FixedTimeStepSize`
    timestepper :: T # redesigned to contain all auxiliary field and settings necessary to the particular timestepping
end

"""
    SplitExplicitFreeSurface(grid = nothing;
                             gravitational_acceleration = g_Earth,
                             substeps = nothing,
                             cfl = nothing,
                             fixed_Δt = nothing,
                             averaging_kernel = averaging_shape_function,
                             timestepper = ForwardBackwardScheme())

Return a `SplitExplicitFreeSurface` representing an explicit time discretization
of a free surface dynamics with `gravitational_acceleration`.

Keyword Arguments
=================

- `gravitational_acceleration`: the gravitational acceleration (default: `g_Earth`)

- `substeps`: The number of substeps that divide the range `(t, t + 2Δt)`, where `Δt` is the baroclinic
              timestep. Note that some averaging functions do not require substepping until `2Δt`.
              The number of substeps is reduced automatically to the last index of `averaging_kernel`
              for which `averaging_kernel > 0`.

- `cfl`: If set then the number of `substeps` are computed based on the advective timescale imposed from
         the barotropic gravity-wave speed that corresponds to depth `grid.Lz`. If `fixed_Δt` is provided,
         then the number of `substeps` adapts to maintain an exact `cfl`. If not, the effective cfl will
         always be lower than the specified `cfl` provided that the baroclinic time step satisfies
         `Δt_baroclinic < fixed_Δt`.

!!! info "Needed keyword arguments"
    Either `substeps` _or_ `cfl` need to be prescribed.
    
    When `cfl` is prescribed then `grid` is also required as a positional argument.

- `fixed_Δt`: The maximum baroclinic timestep allowed. If `fixed_Δt` is a `nothing` and a cfl is provided,
              then the number of substeps will be computed on the fly from the baroclinic time step to
              maintain a constant cfl.

- `averaging_kernel`: A function of `τ` used to average the barotropic transport `U` and the free surface
                      `η` within the barotropic advancement. `τ` is the fractional substep going from 0 to 2
                      with the baroclinic time step `t + Δt` located at `τ = 1`. The `averaging_kernel`
                      function should be centered at `τ = 1`, that is, ``∑ (aₘ m / M) = 1``, where the
                      the summation occurs for ``m = 1, ..., M_*``. Here, ``m = 0`` and ``m = M`` correspond
                      to the two consecutive baroclinic timesteps between which the barotropic timestepping
                      occurs and ``M_*`` corresponds to the last barotropic time step for which the
                      `averaging_kernel > 0`. By default, the averaging kernel described by [Shchepetkin2005](@citet)
                      is used.

- `timestepper`: Time stepping scheme used for the barotropic advancement. Choose one of:
  * `ForwardBackwardScheme()` (default): `η = f(U)`   then `U = f(η)`,
  * `AdamsBashforth3Scheme()`: `η = f(U, Uᵐ⁻¹, Uᵐ⁻²)` then `U = f(η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)`.

References
==========

Shchepetkin, A. F., & McWilliams, J. C. (2005). The regional oceanic modeling system (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model. Ocean Modelling, 9(4), 347-404.
"""
function SplitExplicitFreeSurface(grid = nothing;
                                  gravitational_acceleration = g_Earth,
                                  substeps = nothing,
                                  cfl = nothing,
                                  fixed_Δt = nothing,
                                  averaging_kernel = averaging_shape_function,
                                  timestepper = ForwardBackwardScheme())

    if !isnothing(grid)
        FT = eltype(grid)
    else
        # this is a fallback and only used via the outer constructor,
        # in case no grid is provided; when afterwards the free surfade
        # is materialized via materialize_free_surface
        # FT becomes eltype(grid)
        FT = Float64
    end

    gravitational_acceleration = convert(FT, gravitational_acceleration)
    substepping = split_explicit_substepping(cfl, substeps, fixed_Δt, grid, averaging_kernel, gravitational_acceleration)
    
    kernel_parameters = :xy

    return SplitExplicitFreeSurface(nothing,
                                    nothing,
                                    nothing,
                                    gravitational_acceleration,
                                    kernel_parameters,
                                    substepping,
                                    timestepper)
end

# Simplest case: we have the substeps and the averaging kernel
function split_explicit_substepping(::Nothing, substeps, fixed_Δt, grid, averaging_kernel, gravitational_acceleration)
    FT = eltype(gravitational_acceleration)
    fractional_step_size, averaging_weights = weights_from_substeps(FT, substeps, averaging_kernel)
    return FixedSubstepNumber(fractional_step_size, averaging_weights)
end

# The substeps are calculated dynamically when a cfl without a fixed_Δt is provided
function split_explicit_substepping(cfl, ::Nothing, ::Nothing, grid, averaging_kernel, gravitational_acceleration)  
    if isnothing(grid)
        throw(ArgumentError(string("Need to provide the grid to calculate the barotropic substeps from the cfl. ",
                                    "For example, SplitExplicitFreeSurface(grid, cfl=0.7, ...)")))
    end
    cfl = convert(eltype(grid), cfl)

    return FixedTimeStepSize(grid; cfl, averaging_kernel)
end

# The number of substeps are calculated based on the cfl an2d the fixed_Δt
function split_explicit_substepping(cfl, ::Nothing, fixed_Δt, grid, averaging_kernel, gravitational_acceleration)
    
    substepping = split_explicit_substepping(cfl, nothing, nothing, grid, averaging_kernel, gravitational_acceleration)    
    substeps    = ceil(Int, 2 * fixed_Δt / substepping.Δt_barotropic)
    substepping = split_explicit_substepping(nothing, substeps, nothing, grid, averaging_kernel, gravitational_acceleration)        
    
    return substepping
end

# TODO: When open boundary conditions are online
# We need to calculate the barotropic boundary conditions 
# from the baroclinic boundary conditions by integrating the BC upwards
@inline  west_barotropic_bc(baroclinic_velocity) = baroclinic_velocity.boundary_conditions.west
@inline  east_barotropic_bc(baroclinic_velocity) = baroclinic_velocity.boundary_conditions.east
@inline south_barotropic_bc(baroclinic_velocity) = baroclinic_velocity.boundary_conditions.south
@inline north_barotropic_bc(baroclinic_velocity) = baroclinic_velocity.boundary_conditions.north

@inline barotropic_bc(baroclinic_velocity) = FieldBoundaryConditions(
    west = west_barotropic_bc(baroclinic_velocity),
    east = east_barotropic_bc(baroclinic_velocity),
    south = south_barotropic_bc(baroclinic_velocity),
    north = north_barotropic_bc(baroclinic_velocity),
    top = nothing,
    bottom = nothing
)

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::SplitExplicitFreeSurface, velocities, grid)

    η = free_surface_displacement_field(velocities, free_surface, grid)
    η̅ = free_surface_displacement_field(velocities, free_surface, grid)

    u_baroclinic = velocities.u
    v_baroclinic = velocities.v

    u_bc = barotropic_bc(u_baroclinic)
    v_bc = barotropic_bc(v_baroclinic)

    U = Field{Center, Center, Nothing}(grid, boundary_conditions = u_bc)
    V = Field{Center, Center, Nothing}(grid, boundary_conditions = v_bc)

    U̅ = Field{Center, Center, Nothing}(grid, boundary_conditions = u_bc)
    V̅ = Field{Center, Center, Nothing}(grid, boundary_conditions = v_bc)

    filtered_state = (η = η̅, U = U̅, V = V̅)
    barotropic_velocities = (U = U, V = V)

    gravitational_acceleration = convert(eltype(grid), free_surface.gravitational_acceleration)
    timestepper = materialize_timestepper(free_surface.timestepper, grid, free_surface, velocities, u_bc, v_bc)

    return SplitExplicitFreeSurface(η,
                                    barotropic_velocities,
                                    filtered_state,
                                    gravitational_acceleration,
                                    free_surface.kernel_parameters,
                                    free_surface.substepping,
                                    timestepper)
end

# (p = 2, q = 4, r = 0.18927) minimize dispersion error from Shchepetkin and McWilliams (2005): https://doi.org/10.1016/j.ocemod.2004.08.002 
@inline function averaging_shape_function(τ::FT; p = 2, q = 4, r = FT(0.18927)) where FT
    τ₀ = (p + 2) * (p + q + 2) / (p + 1) / (p + q + 1)

    return (τ / τ₀)^p * (1 - (τ / τ₀)^q) - r * (τ / τ₀)
end

@inline   cosine_averaging_kernel(τ::FT) where FT = τ ≥ 0.5 && τ ≤ 1.5 ? convert(FT, 1 + cos(2π * (τ - 1))) : zero(FT)
@inline constant_averaging_kernel(τ::FT) where FT = convert(FT, 1)

""" An internal type for the `SplitExplicitFreeSurface` that allows substepping with
a fixed `Δt_barotropic` based on a CFL condition """
struct FixedTimeStepSize{B, F}
    Δt_barotropic    :: B
    averaging_kernel :: F
end

""" An internal type for the `SplitExplicitFreeSurface` that allows substepping with
a fixed number of substeps with time step size of `fractional_step_size * Δt_baroclinic` """
struct FixedSubstepNumber{B, F}
    fractional_step_size :: B
    averaging_weights    :: F
end

function FixedTimeStepSize(grid;
                           cfl = 0.7,
                           averaging_kernel = averaging_shape_function,
                           gravitational_acceleration = g_Earth)

    FT = eltype(grid)

    Δx⁻² = topology(grid)[1] == Flat ? 0 : 1 / minimum_xspacing(grid)^2
    Δy⁻² = topology(grid)[2] == Flat ? 0 : 1 / minimum_yspacing(grid)^2
    Δs   = sqrt(1 / (Δx⁻² + Δy⁻²))

    wave_speed = sqrt(gravitational_acceleration * grid.Lz)

    Δt_barotropic = convert(FT, cfl * Δs / wave_speed)

    return FixedTimeStepSize(Δt_barotropic, averaging_kernel)
end

@inline function weights_from_substeps(FT, substeps, averaging_kernel)

    τᶠ = range(FT(0), FT(2), length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]

    averaging_weights = map(averaging_kernel, τᶠ[2:end])
    idx = searchsortedlast(averaging_weights, 0, rev=true)
    substeps = idx

    averaging_weights = averaging_weights[1:idx]
    averaging_weights ./= sum(averaging_weights)

    return Δτ, tuple(averaging_weights...)
end

# Convenience Functions for grabbing free surface
free_surface(free_surface::SplitExplicitFreeSurface) = free_surface.η

# extend
@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)

Base.summary(s::FixedTimeStepSize)  = string("Barotropic time step equal to $(prettytime(s.Δt_barotropic))")
Base.summary(s::FixedSubstepNumber) = string("Barotropic fractional step equal to $(s.fractional_step_size) times the baroclinic step")

Base.summary(sefs::SplitExplicitFreeSurface) = string("SplitExplicitFreeSurface with $(summary(sefs.settings.substepping))")

Base.show(io::IO, sefs::SplitExplicitFreeSurface) = print(io, "$(summary(sefs))\n")

# Adapt
Adapt.adapt_structure(to, free_surface::SplitExplicitFreeSurface) =
    SplitExplicitFreeSurface(Adapt.adapt(to, free_surface.η), nothing, nothing,
                             free_surface.gravitational_acceleration, nothing)

for Type in (:SplitExplicitFreeSurface,
             :AdamsBashforth3Scheme,
             :FixedTimeStepSize,
             :FixedSubstepNumber)
    
    @eval begin
        function on_architecture(to, fs::$Type)
            args = Tuple(on_architecture(to, prop) for prop in propertynames(fs))
            return $Type(args...)
        end
    end
end