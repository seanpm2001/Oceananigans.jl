using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices

using DocStringExtensions
import Oceananigans.Fields: replace_horizontal_velocity_halos!

struct CubedSpherePartition{M, P} <: AbstractPartition
    div :: Int
     Rx :: M
     Ry :: P

    CubedSpherePartition(div, Rx::M, Ry::P) where {M, P} = new{M, P}(div, Rx, Ry)
end

"""
    CubedSpherePartition(; R = 1)

Return a cubed sphere partition with `R` partitions in each horizontal dimension of each
panel of the sphere.
"""
function CubedSpherePartition(; R = 1)
    # CubedSpherePartitions must have Rx = Ry
    Rx = Ry = R

    if R isa Number
        div = 6 * R^2
    else
        div = sum(R .* R)
    end

    @assert mod(div, 6) == 0 "Total number of regions (div = $div) must be a multiple of 6 for a cubed sphere partition."

    return CubedSpherePartition(div, Rx, Ry)
end

const RegularCubedSpherePartition  = CubedSpherePartition{<:Number, <:Number}
const XRegularCubedSpherePartition = CubedSpherePartition{<:Number}
const YRegularCubedSpherePartition = CubedSpherePartition{<:Any, <:Number}

Base.length(p::CubedSpherePartition) = p.div

"""
utilities to get the index of the panel, the index within the panel, and the global index
"""

@inline div_per_panel(panel_idx, partition::RegularCubedSpherePartition)  = partition.Rx            * partition.Ry
@inline div_per_panel(panel_idx, partition::XRegularCubedSpherePartition) = partition.Rx            * partition.Ry[panel_idx]
@inline div_per_panel(panel_idx, partition::YRegularCubedSpherePartition) = partition.Rx[panel_idx] * partition.Ry

@inline Rx(panel_idx, partition::XRegularCubedSpherePartition) = partition.Rx    
@inline Rx(panel_idx, partition::CubedSpherePartition)         = partition.Rx[panel_idx]

@inline Ry(panel_idx, partition::YRegularCubedSpherePartition) = partition.Ry    
@inline Ry(panel_idx, partition::CubedSpherePartition)         = partition.Ry[panel_idx]

@inline panel_index(r, partition)         = (r - 1) ÷ div_per_panel(r, partition) + 1
@inline intra_panel_index(r, partition)   = mod(r - 1, div_per_panel(r, partition)) + 1
@inline intra_panel_index_x(r, partition) = mod(intra_panel_index(r, partition) - 1, Rx(r, partition)) + 1
@inline intra_panel_index_y(r, partition) = (intra_panel_index(r, partition) - 1) ÷ Rx(r, partition) + 1

@inline rank_from_panel_idx(pᵢ, pⱼ, panel_idx, partition::CubedSpherePartition) =
            (panel_idx - 1) * div_per_panel(panel_idx, partition) + Rx(panel_idx, partition) * (pⱼ - 1) + pᵢ

@inline function region_corners(r, p::CubedSpherePartition)
    pᵢ = intra_panel_index_x(r, p)
    pⱼ = intra_panel_index_y(r, p)

    bottom_left  = pᵢ == 1    && pⱼ == 1    ? true : false
    bottom_right = pᵢ == p.Rx && pⱼ == 1    ? true : false
    top_left     = pᵢ == 1    && pⱼ == p.Ry ? true : false
    top_right    = pᵢ == p.Rx && pⱼ == p.Ry ? true : false

    return (; bottom_left, bottom_right, top_left, top_right)
end

@inline function region_edge(r, p::CubedSpherePartition)
    pᵢ = intra_panel_index_x(r, p)
    pⱼ = intra_panel_index_y(r, p)

    west  = pᵢ == 1    ? true : false 
    east  = pᵢ == p.Rx ? true : false
    south = pⱼ == 1    ? true : false
    north = pⱼ == p.Ry ? true : false

    return (; west, east, south, north)
end

#####
##### Boundary-specific Utils
#####

"""
    struct CubedSphereConnectivity{S, FS}

The connectivity among various regions for a cubed sphere grid. Parameters
`S` and `FS` denote the sides of the current region and the region from which
the boundary condition is coming from respectively.

$(TYPEDFIELDS)
"""
struct CubedSphereConnectivity{S <: AbstractRegionSide, FS <: AbstractRegionSide} <: AbstractConnectivity 
    "the current region rank"
            rank :: Int
    "the region from which boundary condition comes from"
       from_rank :: Int
    "the current region side"
            side :: S
    "the side of the region from which boundary condition comes from"
       from_side :: FS

    @doc """
        CubedSphereConnectivity(rank, from_rank, side, from_side)

    Return a `CubedSphereConnectivity`: `from_rank :: Int` → `rank :: Int` and
    `from_side :: AbstractRegionSide` → `side :: AbstractRegionSide`.

    Example
    =======

    A connectivity that implies that the boundary condition for the
    east side of region 1 comes from the west side of region 2 is:

    ```jldoctest cubedsphereconnectivity
    julia> using Oceananigans

    julia> using Oceananigans.MultiRegion: CubedSphereConnectivity, East, West, North, South

    julia> CubedSphereConnectivity(1, 2, East(), West())
    CubedSphereConnectivity{East, West}(1, 2, East(), West())
    ```

    A connectivity that implies that the boundary condition for the
    north side of region 1 comes from the east side of region 3 is 

    ```jldoctest cubedsphereconnectivity
    julia> CubedSphereConnectivity(1, 3, North(), East())
    CubedSphereConnectivity{North, East}(1, 3, North(), East())
    ```
    """
    CubedSphereConnectivity(rank, from_rank, side, from_side) = new{typeof(side), typeof(from_side)}(rank, from_rank, side, from_side)
end

function inject_west_boundary(region, p::CubedSpherePartition, global_bc)
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pᵢ == 1
        if mod(pidx, 2) == 0
            from_side  = East()
            from_panel = pidx - 1
            from_pᵢ    = Rx(from_panel, p)
            from_pⱼ    = pⱼ
        else
            from_side  = North()
            from_panel = mod(pidx + 3, 6) + 1
            from_pᵢ    = Rx(from_panel, p) - pⱼ + 1
            from_pⱼ    = Ry(from_panel, p)
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = East()
        from_rank = rank_from_panel_idx(pᵢ - 1, pⱼ, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, West(), from_side))
end

function inject_east_boundary(region, p::CubedSpherePartition, global_bc) 
 
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pᵢ == p.Rx
        if mod(pidx, 2) != 0
            from_side  = West()
            from_panel = pidx + 1
            from_pᵢ    = 1
            from_pⱼ    = pⱼ
        else
            from_side  = South()
            from_panel = mod(pidx + 1, 6) + 1
            from_pᵢ    = Rx(from_panel, p) - pⱼ + 1
            from_pⱼ    = 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = West()
        from_rank = rank_from_panel_idx(pᵢ + 1, pⱼ, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, East(), from_side))
end

function inject_south_boundary(region, p::CubedSpherePartition, global_bc)
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pⱼ == 1
        if mod(pidx, 2) != 0
            from_side  = North()
            from_panel = mod(pidx + 4, 6) + 1
            from_pᵢ    = pᵢ
            from_pⱼ    = Ry(from_panel, p)
        else
            from_side  = East()
            from_panel = mod(pidx + 3, 6) + 1
            from_pᵢ    = Rx(from_panel, p)
            from_pⱼ    = Ry(from_panel, p) - pᵢ + 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = North()
        from_rank = rank_from_panel_idx(pᵢ, pⱼ - 1, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, South(), from_side))
end

function inject_north_boundary(region, p::CubedSpherePartition, global_bc)
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pⱼ == p.Ry
        if mod(pidx, 2) == 0
            from_side  = South()
            from_panel = mod(pidx, 6) + 1
            from_pᵢ    = pᵢ
            from_pⱼ    = 1
        else    
            from_side  = West()
            from_panel = mod(pidx + 1, 6) + 1
            from_pᵢ    = 1
            from_pⱼ    = Rx(from_panel, p) - pᵢ + 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = South()
        from_rank = rank_from_panel_idx(pᵢ, pⱼ + 1, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, North(), from_side))
end

"Trivial connectivities are East ↔ West, North ↔ South. Anything else is referred to as non-trivial."
const NonTrivialConnectivity = Union{CubedSphereConnectivity{East, South}, CubedSphereConnectivity{East, North},
                                     CubedSphereConnectivity{West, South}, CubedSphereConnectivity{West, North},
                                     CubedSphereConnectivity{South, East}, CubedSphereConnectivity{South, West},
                                     CubedSphereConnectivity{North, East}, CubedSphereConnectivity{North, West}}

@inline flip_west_and_east_indices(buff, conn) = buff
@inline flip_west_and_east_indices(buff, ::NonTrivialConnectivity) = reverse(permutedims(buff, (2, 1, 3)), dims = 2)

@inline flip_south_and_north_indices(buff, conn) = buff
@inline flip_south_and_north_indices(buff, ::NonTrivialConnectivity) = reverse(permutedims(buff, (2, 1, 3)), dims = 1)

function Base.summary(p::CubedSpherePartition)
    region_str = p.Rx * p.Ry > 1 ? "regions" : "region"

    return "CubedSpherePartition with ($(p.Rx * p.Ry) $(region_str) in each panel)"
end

replace_horizontal_velocity_halos!(::PrescribedVelocityFields, ::OrthogonalSphericalShellGrid; signed=true) = nothing

function replace_horizontal_velocity_halos!(velocities, grid::OrthogonalSphericalShellGrid; signed=true)
    u, v, _ = velocities

    ubuff = u.boundary_buffers
    vbuff = v.boundary_buffers

    conn_west  = u.boundary_conditions.west.condition.from_side
    conn_east  = u.boundary_conditions.east.condition.from_side
    conn_south = u.boundary_conditions.south.condition.from_side
    conn_north = u.boundary_conditions.north.condition.from_side

    Hx, Hy, _ = halo_size(u.grid)
    Nx, Ny, _ = size(grid)

     replace_west_u_halos!(u, vbuff, Nx, Hx, conn_west; signed)
     replace_east_u_halos!(u, vbuff, Nx, Hx, conn_east; signed)
    replace_south_u_halos!(u, vbuff, Ny, Hy, conn_south; signed)
    replace_north_u_halos!(u, vbuff, Ny, Hy, conn_north; signed)

     replace_west_v_halos!(v, ubuff, Nx, Hx, conn_west; signed)
     replace_east_v_halos!(v, ubuff, Nx, Hx, conn_east; signed)
    replace_south_v_halos!(v, ubuff, Ny, Hy, conn_south; signed)
    replace_north_v_halos!(v, ubuff, Ny, Hy, conn_north; signed)

    return nothing
end

for vel in (:u, :v), dir in (:east, :west, :north, :south)
    @eval $(Symbol(:replace_, dir, :_, vel, :_halos!))(u, buff, N, H, conn; signed=true) = nothing
end

function replace_west_u_halos!(u, vbuff, N, H, ::North; signed)
    view(u, -H+1:0,  :, :) .= vbuff.west.recv
    return nothing
end

function replace_west_v_halos!(v, ubuff, N, H, ::North; signed)
    view(v, -H+1:0,  :, :) .= ubuff.west.recv
    if signed
        view(v, -H+1:0,  :, :) .*= -1
    end
    return nothing
end

function replace_east_u_halos!(u, vbuff, N, H, ::South; signed)
    view(u, N+1:N+H, :, :) .= vbuff.east.recv
    return nothing
end

function replace_east_v_halos!(v, ubuff, N, H, ::South; signed)
    view(v, N+1:N+H, :, :) .= ubuff.east.recv
    if signed
        view(v, N+1:N+H, :, :) .*= -1
    end
    return nothing
end

function replace_south_u_halos!(u, vbuff, N, H, ::East; signed)
     view(u, :, -H+1:0, :)  .= vbuff.south.recv
     if signed
        view(u, :, -H+1:0, :)  .*= -1
     end
     return nothing
end

function replace_south_v_halos!(v, ubuff, N, H, ::East; signed)
     view(v, :, -H+1:0, :)  .= + ubuff.south.recv
     return nothing
end

function replace_north_u_halos!(u, vbuff, N, H, ::West; signed)
     view(u, :, N+1:N+H, :) .= vbuff.north.recv
     if signed
        view(u, :, N+1:N+H, :) .*= -1
     end
     return nothing
end

function replace_north_v_halos!(v, ubuff, N, H, ::West; signed)
     view(v, :, N+1:N+H, :) .= + ubuff.north.recv
     return nothing
end

Base.show(io::IO, p::CubedSpherePartition) =
    print(io, summary(p), "\n",
          "├── Rx: ", p.Rx, "\n",
          "├── Ry: ", p.Ry, "\n",
          "└── div: ", p.div)