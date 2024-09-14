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

@inline function getbc(bc::Oceananigans.BoundaryConditions.ZBoundaryFunction{LX, LY, S}, i::Integer, j::Integer) where {LX, LY, S}
    #cbf = bc.condition
    #k, k′ = Oceananigans.BoundaryConditions.domain_boundary_indices(S(), grid.Nz)
    # args = Oceananigans.BoundaryConditions.user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    #@show args
    # X = Oceananigans.BoundaryConditions.z_boundary_node(i, j, k′, grid, LX(), LY())
    #@show X
    # args = (9.561340652234675e-48, (; surface_tracer_concentration = 1, piston_velocity = 0.1))
    X = (-1.5707963267948966, -1.5707963267948966)
    return tracer_flux(X..., nothing, 9.561340652234675e-48, (; surface_tracer_concentration = 1, piston_velocity = 0.1))
end

@kernel function _apply_z_bcs!(top_bc)
    i, j = @index(Global, NTuple)
    
    @show @which getbc(top_bc, i, j) 
end

function set_initial_condition!(phi, Gc)
    grid = Gc.grid
    launch!(CPU(), grid, :xy, _apply_z_bcs!, phi.top)
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
    ci(x, y, z) = amplitude2[] * 0.09 # exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    Oceananigans.set!(getproperty(model.tracers, :c), ci)

    amplitude = 1.0
    κ = 1.0
    dmodel = Enzyme.make_zero(model)
    
    phi = getproperty(model.tracers, :c).boundary_conditions
    dphi = getproperty(dmodel.tracers, :c).boundary_conditions
    
    Gc = model.timestepper.Gⁿ[:c]
    dGc = dmodel.timestepper.Gⁿ[:c]
    
    # set_initial_condition!(deepcopy(model), amplitude)

    dc²_dκ = autodiff(Enzyme.Reverse,
                      set_initial_condition!,
		      Duplicated(phi, dphi),
		      Duplicated(Gc, dGc)
		      )
    
end
