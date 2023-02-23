# We assume here that top/bottom BC are always synched (no partitioning in z)

function recompute_boundary_tendencies(model)
    grid = model.grid
    arch = architecture(grid)

    # What shall we do with w, p and κ???

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    size_x = (Hx, Ny, Nz)
    size_y = (Nx, Hy, Nz-2Hz)

    offsetᴸx = (0,  0,  0)
    offsetᴸy = (0,  0,  Hz)
    offsetᴿx = (Nx-Hx, 0,      0)
    offsetᴿy = (0,     Ny-Hy, Hz)

    sizes   = (size_x,     size_y,   size_x,   size_y)
    offsets = (offsetᴸx, offsetᴸy, offsetᴿx, offsetᴿy)

    u_immersed_bc = immersed_boundary_condition(model.velocities.u)
    v_immersed_bc = immersed_boundary_condition(model.velocities.v)

    start_momentum_kernel_args = (grid,
                                  model.advection.momentum,
                                  model.coriolis,
                                  model.closure)

    end_momentum_kernel_args = (model.velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.diffusivity_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.forcing,
                                model.clock)

    u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args...)
    v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args...)
    
    for (kernel_size, kernel_offsets) in zip(sizes, offsets)
        launch!(arch, grid, kernel_size,
                calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, kernel_offsets, u_kernel_args...)
    
        launch!(arch, grid, kernel_size,
                calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, kernel_offsets, v_kernel_args...)
    end

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds c_tendency = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection = model.advection[tracer_name]
        @inbounds c_forcing = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        c_kernel_function, closure, diffusivity_fields = tracer_tendency_kernel_function(model,
                                                                                         Val(tracer_name),
                                                                                         model.closure,
                                                                                         model.diffusivity_fields)

        args = (c_kernel_function,
                grid,
                Val(tracer_index),
                c_advection,
                closure,
                c_immersed_bc,
                model.buoyancy,
                model.velocities,
                model.free_surface,
                model.tracers,
                top_tracer_bcs,
                diffusivity_fields,
                model.auxiliary_fields,
                c_forcing,
                model.clock)

        for (kernel_size, kernel_offsets) in zip(sizes, offsets)
            launch!(arch, grid, kernel_size, calculate_hydrostatic_free_surface_Gc!, c_tendency, kernel_offsets, args...)
        end
    end
end