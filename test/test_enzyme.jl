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

@inline function getbc(bc::Oceananigans.BoundaryConditions.ZBoundaryFunction{LX, LY, S}, i::Integer, j::Integer,
                       grid::AbstractGrid) where {LX, LY, S}
    cbf = bc.condition
    k, k′ = Oceananigans.BoundaryConditions.domain_boundary_indices(S(), grid.Nz)
    # args = Oceananigans.BoundaryConditions.user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    #@show args
    X = Oceananigans.BoundaryConditions.z_boundary_node(i, j, k′, grid, LX(), LY())
    #@show X
    # args = (9.561340652234675e-48, (; surface_tracer_concentration = 1, piston_velocity = 0.1))
    # X = (-1.5707963267948966, -1.5707963267948966)
    return tracer_flux(X..., nothing, 9.561340652234675e-48, (; surface_tracer_concentration = 1, piston_velocity = 0.1))
end

@inline function apply_z_top_bc!(Gc, top_flux, i, j, grid, args...)
    # LX, LY, LZ = loc
    @show @which getbc(top_flux, i, j, grid, args...) 
    @inbounds Gc[i, j, grid.Nz] *= getbc(top_flux, i, j, grid, args...) # * Oceananigans.AbstractOperations.Az(i, j, grid.Nz+1, grid, LX, LY, Oceananigans.AbstractOperations.flip(LZ)) / Oceananigans.Operators.volume(i, j, grid.Nz, grid, LX, LY, LZ)
    return nothing
end

@kernel function _apply_z_bcs!(Gc, grid, top_bc, args)
    i, j = @index(Global, NTuple)
    # Oceananigans.BoundaryConditions.apply_z_bottom_bc!(Gc, loc, bottom_bc, i, j, grid, args...)
    # Oceananigans.BoundaryConditions.
    apply_z_top_bc!(Gc, top_bc,    i, j, grid, args...)
end

myapply_z_bcs!(Gc, grid::AbstractGrid, c, bottom_bc, top_bc, arch::AbstractArchitecture, args...) =
    launch!(arch, grid, :xy, _apply_z_bcs!, Gc, Oceananigans.instantiated_location(Gc), grid, bottom_bc, top_bc, Tuple(args))

myapply_z_bcs!(Gc, c, args...) = myapply_z_bcs!(Gc, Gc.grid, c, c.boundary_conditions.bottom, c.boundary_conditions.top, args...)

function set_initial_condition!(model, amplitude)
    tracers = model.tracers
    
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    ci(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    
    phi = getproperty(tracers, :c)
   
    Oceananigans.set!(phi, ci)
    
    Gⁿ = model.timestepper.Gⁿ

    Gc = Gⁿ[:c]
    # myapply_z_bcs!(Gc, Gc.grid, phi, phi.boundary_conditions.bottom, phi.boundary_conditions.top, arch, args...)
    # myapply_z_bcs!(Gc, phi, arch, model.clock, fields(model))
    
    grid = Gc.grid
    launch!(CPU(), grid, :xy, _apply_z_bcs!, Gc, grid, phi.boundary_conditions.top, ())

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
    
    # set_initial_condition!(deepcopy(model), amplitude)

    dc²_dκ = autodiff(Enzyme.Reverse,
                      set_initial_condition!,
                      Duplicated(model, dmodel),
                      Const(amplitude))
    
end
