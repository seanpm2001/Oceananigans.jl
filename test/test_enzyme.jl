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

function set_initial_condition!(model, amplitude)
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    ci(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    
    ϕ = getproperty(model.tracers, :c)
   
    # @apply_regionally set!(ϕ,ci)
    #apply_regionally!(set!, ϕ, ci)
    Oceananigans.set!(ϕ, ci)
    
    kernel_parameters = tuple(Oceananigans.Models.HydrostaticFreeSurfaceModels.interior_tendency_kernel_parameters(model.grid))
    
   Gⁿ = model.timestepper.Gⁿ
 arch = model.architecture
 velocities = model.velocities
 free_surface = model.free_surface
 tracers = model.tracers
 args = (model.clock,
 fields(model),
 model.closure,
 model.buoyancy)

    # Velocity fields
    for i in (:u, :v)
        Oceananigans.Models.HydrostaticFreeSurfaceModels.apply_flux_bcs!(Gⁿ[i], velocities[i], arch, args)
    end

    # Free surface
    Oceananigans.Models.HydrostaticFreeSurfaceModels.apply_flux_bcs!(Gⁿ.η, Oceananigans.Models.HydrostaticFreeSurfaceModels.displacement(free_surface), arch, args)

    # Tracer fields
    for i in propertynames(tracers)
        Oceananigans.Models.HydrostaticFreeSurfaceModels.apply_flux_bcs!(Gⁿ[i], tracers[i], arch, args)
    end



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
