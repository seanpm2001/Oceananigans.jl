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


const maximum_diffusivity = 100

apply_z_bcs!(Gc, c, args...) = apply_z_bcs!(Gc, Gc.grid, c, c.boundary_conditions.bottom, c.boundary_conditions.top, args...)

apply_z_bcs!(Gc, grid::AbstractGrid, c, bottom_bc, top_bc, arch::AbstractArchitecture, args...) =
    launch!(arch, grid, :xy, _apply_z_bcs!, Gc, instantiated_location(Gc), grid, bottom_bc, top_bc, Tuple(args))


@inline function apply_z_bottom_bc!(Gc, loc, bottom_flux, i, j, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, j, 1] += getbc(bottom_flux, i, j, grid, args...) * Az(i, j, 1, grid, LX, LY, flip(LZ)) / volume(i, j, 1, grid, LX, LY, LZ)
    return nothing
end

@inline function apply_z_top_bc!(Gc, loc, top_flux, i, j, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, j, grid.Nz] -= getbc(top_flux, i, j, grid, args...) * Az(i, j, grid.Nz+1, grid, LX, LY, flip(LZ)) / volume(i, j, grid.Nz, grid, LX, LY, LZ)
    return nothing
end

@kernel function _apply_z_bcs!(Gc, loc, grid, bottom_bc, top_bc, args)
    i, j = @index(Global, NTuple)
    apply_z_bottom_bc!(Gc, loc, bottom_bc, i, j, grid, args...)
       apply_z_top_bc!(Gc, loc, top_bc,    i, j, grid, args...)
end

function set_initial_condition!(model, amplitude)
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    ci(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    
    ϕ = getproperty(model.tracers, :c)
   
    # @apply_regionally set!(ϕ,ci)
    #apply_regionally!(set!, ϕ, ci)
    Oceananigans.set!(ϕ, ci)
    
    # kernel_parameters = tuple(Oceananigans.Models.HydrostaticFreeSurfaceModels.interior_tendency_kernel_parameters(model.grid))
    
   Gⁿ = model.timestepper.Gⁿ
 arch = model.architecture
 # velocities = model.velocities
 # free_surface = model.free_surface
 tracers = model.tracers
 args = (model.clock,
 fields(model),
 model.closure,
 model.buoyancy)

        #Oceananigans.Models.HydrostaticFreeSurfaceModels.apply_z_bcs!(Gⁿ[:c], tracers[:c], arch, args)
	apply_z_bcs!(Gⁿ[:c], tracers[:c], arch, args)

    return nothing
end

using InteractiveUtils

@testset "Enzyme on advection and diffusion WITH flux boundary condition" begin
    Nx = Ny = 64
    Nz = 8

    Lx = Ly = L = 2π
    Lz = 1

    x = y = (-L/2, L/2)
    z = (-Lz/2, Lz/2)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)
    diffusion = VerticalScalarDiffusivity(κ=0.1)

    @inline function tracer_flux(x, y, t, c, p)
        c₀ = p.surface_tracer_concentration
        u★ = p.piston_velocity
        return - u★ * (c₀ - c)
    end

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

    # This has a "width" of 0.1
    cf(x, y, z) = amplitude2[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    # @show @which set!(model, cf)

    # Now for real
    amplitude = 1.0
    κ = 1.0
    dmodel = Enzyme.make_zero(model)
    # set_diffusivity!(dmodel, 0)
    
    dc²_dκ = autodiff(Enzyme.Reverse,
                      set_initial_condition!,
                      Duplicated(model, dmodel),
                      Const(amplitude))
    
end
