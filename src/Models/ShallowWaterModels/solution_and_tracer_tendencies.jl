using Oceananigans.Advection
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ

# bathymetry (hB) is assumed to be a negative value equal to - depth.
@inline free_surface(i, j, k, grid, h, hB) = @inbounds h[i, j, k] + hB[i, j, k]

@inline x_pressure_gradient(i, j, k, grid, g, h, hB, ::ConservativeFormulation) = g * ℑxᶠᵃᵃ(i, j, k, grid, h) * ∂xᶠᶜᶜ(i, j, k, grid, free_surface, h, hB)
@inline y_pressure_gradient(i, j, k, grid, g, h, hB, ::ConservativeFormulation) = g * ℑyᵃᶠᵃ(i, j, k, grid, h) * ∂yᶜᶠᶜ(i, j, k, grid, free_surface, h, hB)

@inline x_pressure_gradient(i, j, k, grid, g, h, hB, ::VectorInvariantFormulation) = g * ∂xᶠᶜᶜ(i, j, k, grid, free_surface, h, hB)
@inline y_pressure_gradient(i, j, k, grid, g, h, hB, ::VectorInvariantFormulation) = g * ∂yᶜᶠᶜ(i, j, k, grid, free_surface, h, hB)

"""
Compute the tendency for the x-directional transport, uh
"""
@inline function uh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      velocities,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock,
                                      formulation)

    g = gravitational_acceleration

    model_fields = shallow_water_fields(velocities, tracers, solution, formulation)

    return ( - div_mom_u(i, j, k, grid, advection, solution, formulation)
             - x_pressure_gradient(i, j, k, grid, g, solution.h, bathymetry, formulation)
             - x_f_cross_U(i, j, k, grid, coriolis, solution)
             - sw_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, formulation)
             + forcings[1](i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the y-directional transport, vh.
"""
@inline function vh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      velocities,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock,
                                      formulation)

     g = gravitational_acceleration

     model_fields = shallow_water_fields(velocities, tracers, solution, formulation)

    return ( - div_mom_v(i, j, k, grid, advection, solution, formulation)
             - y_pressure_gradient(i, j, k, grid, g, solution.h, bathymetry, formulation)
             - y_f_cross_U(i, j, k, grid, coriolis, solution)
             - sw_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, formulation)
             + forcings[2](i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the height, h.
"""
@inline function h_solution_tendency(i, j, k, grid,
                                     gravitational_acceleration,
                                     advection,
                                     coriolis,
                                     closure,
                                     solution,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock,
                                     formulation)

    return ( - div_Uh(i, j, k, grid, advection, solution, formulation)
             + forcings.h(i, j, k, grid, clock, merge(solution, tracers)))
end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 closure,
                                 solution,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock,
                                 formulation) where tracer_index

    @inbounds c = tracers[tracer_index]

    return ( - div_Uc(i, j, k, grid, advection, solution, c, formulation) 
             + c_div_U(i, j, k, grid, solution, c, formulation)         
             + forcing(i, j, k, grid, clock, merge(solution, tracers)) 
            )
end
