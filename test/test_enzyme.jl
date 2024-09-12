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

# myinitialize!(model) = Oceananigans.Models.HydrostaticFreeSurfaceModels.initialize_free_surface!(model.free_surface, model.grid, model.velocities)

myupdate_state!(model, callbacks=[]; compute_tendencies = true) =
    myupdate_state!(model, model.grid, callbacks; compute_tendencies)

function myupdate_state!(model, grid, callbacks; compute_tendencies = true)
    # Oceananigans.Models.HydrostaticFreeSurfaceModels.mask_immersed_model_fields!(model, grid)

    # Update possible FieldTimeSeries used in the model
    # Oceananigans.Models.HydrostaticFreeSurfaceModels.update_model_field_time_series!(model, model.clock)

    # Oceananigans.Models.HydrostaticFreeSurfaceModels.fill_halo_regions!(Oceananigans.Models.HydrostaticFreeSurfaceModels.prognostic_fields(model), model.clock, fields(model); async = true)
    # Oceananigans.Models.HydrostaticFreeSurfaceModels.replace_horizontal_vector_halos!(model.velocities, model.grid)
    # Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_auxiliaries!(model)

    # Oceananigans.Models.HydrostaticFreeSurfaceModels.fill_halo_regions!(model.diffusivity_fields; only_local_halos = true)

    # Oceananigans.Models.HydrostaticFreeSurfaceModels.update_biogeochemical_state!(model.biogeochemistry, model)

    Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_tendencies!(model, callbacks)

    return nothing
end

function mycompute_tendencies!(model, callbacks)

    kernel_parameters = tuple(Oceananigans.Models.HydrostaticFreeSurfaceModels.interior_tendency_kernel_parameters(model.grid))

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
     Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters;
                                                             active_cells_map = active_interior_map(model.grid))

     Oceananigans.Models.HydrostaticFreeSurfaceModels.complete_communication_and_compute_boundary!(model, model.grid, model.architecture)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
     Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                         model.architecture,
                                                         model.velocities,
                                                         model.free_surface,
                                                         model.tracers,
                                                         model.clock,
                                                         fields(model),
                                                         model.closure,
                                                         model.buoyancy)

    #for callback in callbacks
    #    callback.callsite isa TendencyCallsite && callback(model)
    #end

     Oceananigans.Models.HydrostaticFreeSurfaceModels.update_tendencies!(model.biogeochemistry, model)

    return nothing
end

function set_initial_condition!(model, amplitude)
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    ci(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    
    ϕ = getproperty(model.tracers, :c)
   
    # @apply_regionally set!(ϕ,ci)
    #apply_regionally!(set!, ϕ, ci)
    Oceananigans.set!(ϕ, ci)

    # Oceananigans.initialize!(model)
    # Oceananigans.
    # myupdate_state!(model)
    mycompute_tendencies!(model, ()) # callbacks)

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
