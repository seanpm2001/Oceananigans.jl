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

@inline function tracer_flux(c, p)
  c₀ = p.surface_tracer_concentration
  u★ = p.piston_velocity
  return - u★ * (c₀ - c)
end

@inline function tracer_flux2(x, y, t, c, p)
  c₀ = p.surface_tracer_concentration
  u★ = p.piston_velocity
  return - u★ * (c₀ - c)
end

@inline function mygetbc(bc::Oceananigans.BoundaryConditions.ZBoundaryFunction{LX, LY, S}, grid::AbstractGrid, model_fields) where {LX, LY, S}
    cbf = bc.condition
    
    pop = cbf.field_dependencies_interp
    idx = cbf.field_dependencies_indices
    field_args = @inbounds (pop[1](1, 1, 1, grid, model_fields[idx[1]]),)
    args = (field_args...,)

    return tracer_flux(args..., cbf.parameters)
end

@kernel function my_apply_z_bcs!(Gc, grid, top_flux, model_fields)
    i, j = @index(Global, NTuple)
    @inbounds Gc[i, j, grid.Nz] *= mygetbc(top_flux, grid, model_fields)
    nothing
end

function set_initial_condition!(model, grid, top)

    Gⁿ = model.timestepper.Gⁿ

    Gc = Gⁿ[:c]

    launch!(CPU(), grid, :xy, my_apply_z_bcs!, Gc,  grid, top, fields(model))

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

    top_c_bc = FluxBoundaryCondition(tracer_flux2, field_dependencies=:c; parameters)
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

    amplitude = Ref(1.0)
    ci(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    Oceananigans.set!(getproperty(model.tracers, :c), ci)
    
    κ = 1.0
    dmodel = Enzyme.make_zero(model)
    
    # set_initial_condition!(deepcopy(model), grid)

    dc²_dκ = autodiff(Enzyme.Reverse,
                      set_initial_condition!,
                      Duplicated(model, dmodel),
                      Const(grid),
		      Const(getproperty(model.tracers, :c).boundary_conditions.top) # phi = getproperty(model.tracers, :c)
		      )
    
end
