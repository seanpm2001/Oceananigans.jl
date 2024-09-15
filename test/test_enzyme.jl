include("dependencies_for_runtests.jl")

using Enzyme
using EnzymeCore
# Required presently
Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Oceananigans.Fields: FunctionField
using Oceananigans: architecture
using KernelAbstractions

using InteractiveUtils

@inline function tracer_flux(x, y, t, c, p)
  c₀ = p.surface_tracer_concentration
  u★ = p.piston_velocity
  return - u★ * (c₀ - c)
end

@inline function apply_z_bottom_bc!(Gc, loc, bottom_flux, i, j, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, j, 1] += Oceananigans.BoundaryConditions.getbc(bottom_flux, i, j, grid, args...) * Oceananigans.AbstractOperations.Az(i, j, 1, grid, LX, LY, Oceananigans.AbstractOperations.flip(LZ)) / Oceananigans.Operators.volume(i, j, 1, grid, LX, LY, LZ)
    return nothing
end

@inline function getbc(bc::Oceananigans.BoundaryConditions.ZBoundaryFunction{LX, LY, S}, i::Integer, j::Integer,
                       grid::AbstractGrid, model_fields) where {LX, LY, S}
    cbf = bc.condition
    k, k′ = Oceananigans.BoundaryConditions.domain_boundary_indices(S(), grid.Nz)
    args = Oceananigans.BoundaryConditions.user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    X = (-1.5707963267948966, -1.5707963267948966)
    return tracer_flux(X..., nothing, args...)
end

@inline function apply_z_top_bc!(Gc, top_flux, i, j, grid, model_fields)
    @inbounds Gc[i, j, grid.Nz] *= getbc(top_flux, i, j, grid, model_fields)
    return nothing
end

@kernel function _apply_z_bcs!(Gc, grid, top_bc, model_fields)
    i, j = @index(Global, NTuple)
    apply_z_top_bc!(Gc, top_bc,    i, j, grid, model_fields)
end

function set_initial_condition!(model, amplitude)
    tracers = model.tracers
    
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    ci(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    
    phi = getproperty(tracers, :c)
   
    Oceananigans.set!(phi, ci)
    
   Gⁿ = model.timestepper.Gⁿ

    Gc = Gⁿ[:c]

    grid = Gc.grid
    launch!(CPU(), grid, :xy, _apply_z_bcs!, Gc,  grid, phi.boundary_conditions.top, fields(model))

    return nothing
end


@testset "Enzyme on advection and diffusion WITH flux boundary condition" begin
    Nx = Ny = 2
    Nz = 8

    Lx = Ly = L = 2π
    Lz = 1

    x = y = (-L/2, L/2)
    z = (-Lz/2, Lz/2)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)
    diffusion = VerticalScalarDiffusivity(κ=0.1)

    parameters = (surface_tracer_concentration = 1,
                  piston_velocity = 0.1)

    top_c_bc = FluxBoundaryCondition(tracer_flux, field_dependencies=:c; parameters)
    c_bcs = FieldBoundaryConditions(top=top_c_bc)

    # TODO:
    # 1. Make the velocity fields evolve
    # 2. Add surface fluxes
    # 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracers = :c,
                                        buoyancy = nothing,
                                        boundary_conditions = (; c=c_bcs),
                                        closure = diffusion)

    amplitude2 = Ref(1.0)

    cf(x, y, z) = amplitude2[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)

    amplitude = 1.0
    κ = 1.0
    dmodel = Enzyme.make_zero(model)
    
    dc²_dκ = autodiff(Enzyme.Reverse,
                      set_initial_condition!,
                      Duplicated(model, dmodel),
                      Const(amplitude))
    
end
