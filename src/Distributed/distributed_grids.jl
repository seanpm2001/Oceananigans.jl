using MPI
using Oceananigans.Utils: getname
using Oceananigans.Grids: topology, size, halo_size, architecture, pop_flat_elements
using Oceananigans.Grids: validate_rectilinear_grid_args, validate_lat_lon_grid_args
using Oceananigans.Grids: generate_coordinate, with_precomputed_metrics
using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z
using Oceananigans.Grids: R_Earth

using Oceananigans.ImmersedBoundaries

import Oceananigans.Grids: RectilinearGrid, LatitudeLongitudeGrid, with_halo

const DistributedGrid{FT, TX, TY, TZ} = AbstractGrid{FT, TX, TY, TZ, <:MultiArch}
const DistributedRectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ} =
    RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, <:MultiArch} where {FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ}
const DistributedLatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ} = 
    LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, <:MultiArch} where {FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ}

const DistributedImmersedBoundaryGrid = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:DistributedGrid, I, M, <:MultiArch} where {FT, TX, TY, TZ, I, M}

"""
    RectilinearGrid(arch::MultiArch, FT=Float64; kw...)

Return the rank-local portion of `RectilinearGrid` on `arch`itecture.
"""
function RectilinearGrid(arch::MultiArch, 
                         FT::DataType = Float64;
                         size,
                         x = nothing,
                         y = nothing,
                         z = nothing,
                         halo = nothing,
                         extent = nothing,
                         topology = (Periodic, Periodic, Bounded))

    TX, TY, TZ, size, halo, x, y, z =
        validate_rectilinear_grid_args(topology, size, halo, FT, extent, x, y, z)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    ri, rj, rk = arch.local_index
    Rx, Ry, Rz = arch.ranks

    TX = insert_connected_topology(TX, Rx, ri)
    TY = insert_connected_topology(TY, Ry, rj)
    TZ = insert_connected_topology(TZ, Rz, rk)

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)

    # Local sizes are denoted with lowercase `n`
    nx, ny, nz = local_size = Nx÷Rx, Ny÷Ry, Nz÷Rz

    xl = partition(x, nx, Rx, ri)
    yl = partition(y, ny, Ry, rj)
    zl = partition(z, nz, Rz, rk)

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, topology[1], nx, Hx, xl, child_architecture(arch))
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, topology[2], ny, Hy, yl, child_architecture(arch))
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topology[3], nz, Hz, zl, child_architecture(arch))

    architecture = MultiArch(child_architecture(arch), topology = topology, ranks = arch.ranks, communicator = arch.communicator)

    return RectilinearGrid{TX, TY, TZ}(architecture,
                                       nx, ny, nz,
                                       Hx, Hy, Hz,
                                       Lx, Ly, Lz,
                                       Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                       Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ,
                                       Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

function LatitudeLongitudeGrid(arch::MultiArch,
                               FT::DataType = Float64; 
                               precompute_metrics = false,
                               size,
                               latitude,
                               longitude,
                               z,           
                               topology = nothing,           
                               radius = R_Earth,
                               halo = (1, 1, 1))

    Nλ, Nφ, Nz, Hλ, Hφ, Hz, latitude, longitude, z, topology, precompute_metrics =
        validate_lat_lon_grid_args(FT, latitude, longitude, z, size, halo, topology, precompute_metrics)
    
    ri, rj, rk = arch.local_index
    Rx, Ry, Rz = arch.ranks

    TX = insert_connected_topology(topology[1], Rx, ri)
    TY = insert_connected_topology(topology[2], Ry, rj)
    TZ = insert_connected_topology(topology[3], Rz, rk)

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nλ / Rx)
    @assert isinteger(Nφ / Ry)
    @assert isinteger(Nz / Rz)

    nλ, nφ, nz = local_size = Nλ÷Rx, Nφ÷Ry, Nz÷Rz

    λl = partition(longitude, nλ, Rx, ri)
    φl = partition(latitude,  nφ, Ry, rj)
    zl = partition(z,         nz, Rz, rk)

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX, nλ, Hλ, λl, arch.child_architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, TY, nφ, Hφ, φl, arch.child_architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, nz, Hz, zl, arch.child_architecture)

    architecture = MultiArch(child_architecture(arch), topology = topology, ranks = arch.ranks, communicator = arch.communicator)

    preliminary_grid = LatitudeLongitudeGrid{TX, TY, TZ}(architecture,
                                                         nλ, nφ, nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                         (nothing for i=1:10)..., radius)

    return !precompute_metrics ? preliminary_grid : with_precomputed_metrics(preliminary_grid)
end

function reconstruct_global_grid(grid::DistributedRectilinearGrid)

    arch = grid.architecture
    ri, rj, rk = arch.local_index

    Rx, Ry, Rz = R = arch.ranks

    nx, ny, nz = n = size(grid)
    Hx, Hy, Hz = H = halo_size(grid)
    Nx, Ny, Nz = n .* R

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, Rx, ri, arch.communicator)
    TY = reconstruct_global_topology(TY, Ry, rj, arch.communicator)
    TZ = reconstruct_global_topology(TZ, Rz, rk, arch.communicator)

    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    xG = assemble(x, nx, Rx, ri, arch)
    yG = assemble(y, ny, Ry, rj, arch)
    zG = assemble(z, nz, Rz, rk, arch)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, TX, Nx, Hx, xG, child_arch)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, TY, Ny, Hy, yG, child_arch)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, Nz, Hz, zG, child_arch)

    return RectilinearGrid{TX, TY, TZ}(child_arch,
                                       Nx, Ny, Nz,
                                       Hx, Hy, Hz,
                                       Lx, Ly, Lz,
                                       Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                       Δyᵃᶠᵃ, Δyᵃᶜᵃ, yᵃᶠᵃ, yᵃᶜᵃ,
                                       Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

function reconstruct_global_grid(grid::DistributedLatitudeLongitudeGrid)

    arch = grid.architecture
    i, j, k = arch.local_index

    Rx, Ry, Rz = R = arch.ranks

    nλ, nφ, nz = n = size(grid)
    Hλ, Hφ, Hz = H = halo_size(grid)
    Nλ, Nφ, Nz = n .* R

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, Rx, ri, arch.communicator)
    TY = reconstruct_global_topology(TY, Ry, rj, arch.communicator)
    TZ = reconstruct_global_topology(TZ, Rz, rk, arch.communicator)

    λ = cpu_face_constructor_x(grid)
    φ = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    λG = assemble(λ, nλ, Rx, i, arch)
    φG = assemble(φ, nφ, Ry, j, arch)
    zG = assemble(z, nz, Rz, k, arch)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX, Nλ, Hλ, λG, child_arch)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, TY, Nφ, Hφ, φG, child_arch)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, Nz, Hz, zG, child_arch)

    preliminary_grid = LatitudeLongitudeGrid{TX, TY, TZ}(child_arch,
                                                         Nλ, Nφ, Nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                         (nothing for i=1:10)..., radius)

    return !precompute_metrics ? preliminary_grid : with_precomputed_metrics(preliminary_grid)
end

function reconstruct_global_grid(grid::ImmersedBoundaryGrid)
    arch      = grid.architecture
    local_ib  = grid.immersed_boundary    
    global_ug = reconstruct_global_grid(grid.underlying_grid)
    global_ib = getname(local_ib)(reconstruct_global_array(arch, local_ib.bottom_height, size(grid)))
    return ImmersedBoundaryGrid(global_ug, global_ib)
end

function with_halo(new_halo, grid::DistributedRectilinearGrid) 
    new_grid = with_halo(new_halo, reconstruct_global_grid(grid))    
    return scatter_local_grids(architecture(grid), new_grid)
end

function with_halo(new_halo, grid::DistributedLatitudeLongitudeGrid) 
    new_grid = with_halo(new_halo, reconstruct_global_grid(grid))    
    return scatter_local_grids(architecture(grid), new_grid)
end

function with_halo(new_halo, grid::DistributedImmersedBoundaryGrid)
    new_grid = with_halo(new_halo, reconstruct_global_grid(grid))    
    return scatter_local_grids(architecture(grid), new_grid)
end

function scatter_grid_properties(global_grid)
    # Pull out face grid constructors
    x = cpu_face_constructor_x(global_grid)
    y = cpu_face_constructor_y(global_grid)
    z = cpu_face_constructor_z(global_grid)

    topo = topology(global_grid)
    sz   = pop_flat_elements(size(global_grid), topo)
    halo = pop_flat_elements(halo_size(global_grid), topo)

    return x, y, z, topo, sz, halo
end

function scatter_local_grids(arch::MultiArch, global_grid::RectilinearGrid)
    x, y, z, topo, sz, halo = scatter_grid_properties(global_grid)
    return RectilinearGrid(arch, eltype(global_grid); size=sz, x=x, y=y, z=z, halo=halo, topology=topo)
end

function scatter_local_grids(arch::MultiArch, global_grid::LatitudeLongitudeGrid)
    x, y, z, topo, sz, halo = scatter_grid_properties(global_grid)
    return LatitudeLongitudeGrid(arch, eltype(global_grid); size=sz, longitude=x, latitude=y, z=z, halo=halo, topology=topo)
end

function scatter_local_grids(arch::MultiArch, global_grid::ImmersedBoundaryGrid)
    ib   = global_grid.immersed_boundary
    ug   = global_grid.underlying_grid

    local_ug = scatter_local_grids(arch, ug)
    local_ib = getname(ib)(partition_global_array(arch, ib.bottom_height, size(global_grid)))
    
    return ImmersedBoundaryGrid(local_ug, local_ib)
end

insert_connected_topology(T, R, r) = T

insert_connected_topology(::Type{Bounded}, R, r) = ifelse(R == 1,  Bounded,
                                                   ifelse(r == 1, RightConnected,
                                                   ifelse(r == R, LeftConnected,
                                                   FullyConnected)))

insert_connected_topology(::Type{Periodic}, R, r) = ifelse(R == 1, Periodic, FullyConnected)

function reconstruct_global_topology(T, R, r, comm)
    if R == 1
        return T
    end
    
    topologies = zeros(R)
    if T == FullyConnected
        topologies[r] = 1
    end

    MPI.Allreduce!(topologies, +, comm)

    if sum(topologies) == R
        return Periodic
    else
        return Bounded
    end
end
