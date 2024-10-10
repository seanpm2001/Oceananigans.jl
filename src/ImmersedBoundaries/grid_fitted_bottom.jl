using Adapt
using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Grids: total_size
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.BoundaryConditions: FBC
using Printf

import Oceananigans.TurbulenceClosures: bottom_height

#####
##### GridFittedBottom (2.5D immersed boundary with modified bottom height)
#####

abstract type AbstractGridFittedBottom{H} <: AbstractGridFittedBoundary end

# To enable comparison with PartialCellBottom in the limiting case that
# fractional cell height is 1.0.
struct InterfaceImmersedCondition end
struct CenterImmersedCondition end

struct GridFittedBottom{H, I} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    immersed_condition :: I
end

GridFittedBottom(bottom_height) = GridFittedBottom(bottom_height, CenterImmersedCondition())

Base.summary(::CenterImmersedCondition) = "CenterImmersedCondition"
Base.summary(::InterfaceImmersedCondition) = "InterfaceImmersedCondition"

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

"""
    GridFittedBottom(bottom_height)

Return a bottom immersed boundary.

Keyword Arguments
=================

* `bottom_height`: an array or function that gives the height of the
              bottom in absolute ``z`` coordinates.

* `immersed_condition`: Determine whether the part of the domain that is immersed are all the cell centers that lie below
        `bottom_height` (`CenterImmersedCondition()`; default) or all the cell faces that lie below `bottom_height` (`InterfaceImmersedCondition()`). 
        The only purpose of `immersed_condition` to allow `GridFittedBottom` and `PartialCellBottom` to have the same behavior when the
        minimum fractional cell height for partial cells is set to 0.
"""

function Base.summary(ib::GridFittedBottom)
    zmax  = maximum(ib.bottom_height)
    zmin  = minimum(ib.bottom_height)
    zmean = mean(ib.bottom_height)

    summary1 = "GridFittedBottom("

    summary2 = string("mean(z)=", prettysummary(zmean),
                      ", min(z)=", prettysummary(zmin),
                      ", max(z)=", prettysummary(zmax))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::GridFittedBottom{<:Function}) = @sprintf("GridFittedBottom(%s)", ib.bottom_height)

function Base.show(io::IO, ib::GridFittedBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
end

on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(on_architecture(arch, ib.bottom_height), ib.immersed_condition)

function on_architecture(arch, ib::GridFittedBottom{<:Field})
    architecture(ib.bottom_height) == arch && return ib
    arch_grid = on_architecture(arch, ib.bottom_height.grid)
    new_bottom_height = Field{Center, Center, Nothing}(arch_grid)
    set!(new_bottom_height, ib.bottom_height)
    fill_halo_regions!(new_bottom_height)
    return GridFittedBottom(new_bottom_height, ib.immersed_condition)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom_height), adapt(to, ib.immersed_condition))

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary (`ib`).

Computes `ib.bottom_height` and wraps it in a Field. `ib.bottom_height` is the z-coordinate of top-most interface
of the last ``immersed`` cell in the column.
"""
function ImmersedBoundaryGrid(grid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    @apply_regionally correct_bottom_height!(bottom_field, grid, ib)
    fill_halo_regions!(bottom_field)
    new_ib = GridFittedBottom(bottom_field)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
end

correct_bottom_height!(bottom_field, grid, ib) = 
    launch!(architecture(grid), grid, :xy, _correct_bottom_height!, bottom_field, grid, ib)

@kernel function _correct_bottom_height!(bottom_field, grid, ib::GridFittedBottom)
    i, j = @index(Global, NTuple)
    zb = @inbounds bottom_field[i, j, 1]
    condition = ib.immersed_condition
    for k in 1:grid.Nz
        z⁺ = znode(i, j, k+1, grid, c, c, f)
        z  = znode(i, j, k,   grid, c, c, c)
        bottom_cell = ifelse(condition isa CenterImmersedCondition, z ≤ zb, z⁺ ≤ zb)
        @inbounds bottom_field[i, j, 1] = ifelse(bottom_cell, z⁺, zb)
    end
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom)
    z  = znode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    return z ≤ zb
end

@inline bottom_height(i, j, ibg::GFBIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]

#####
##### Bottom height
#####

const AGFBIB = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline column_heightᶜᶜᵃ(i, j, k, ibg::AGFBIB) = @inbounds znode(i, j, ibg.Nz+1, ibg, c, c, f) - ibg.immersed_boundary.bottom_height[i, j, 1] 
@inline column_heightᶜᶠᵃ(i, j, k, ibg::AGFBIB) = min(column_heightᶜᶜᵃ(i, j-1, k, ibg), column_heightᶜᶜᵃ(i, j, k, ibg))
@inline column_heightᶠᶜᵃ(i, j, k, ibg::AGFBIB) = min(column_heightᶜᶜᵃ(i-1, j, k, ibg), column_heightᶜᶜᵃ(i, j, k, ibg))
@inline column_heightᶠᶠᵃ(i, j, k, ibg::AGFBIB) = min(column_heightᶠᶜᵃ(i, j-1, k, ibg), column_heightᶠᶜᵃ(i, j, k, ibg))

# Make sure column_height works for horizontally-Flat topologies.
XFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}
YFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline column_heightᶠᶜᵃ(i, j, k, ibg::XFlatAGFIBG) = column_heightᶜᶜᵃ(i, j, k, ibg)
@inline column_heightᶜᶠᵃ(i, j, k, ibg::YFlatAGFIBG) = column_heightᶜᶜᵃ(i, j, k, ibg)
@inline column_heightᶠᶠᵃ(i, j, k, ibg::XFlatAGFIBG) = column_heightᶜᶠᵃ(i, j, k, ibg)
@inline column_heightᶠᶠᵃ(i, j, k, ibg::YFlatAGFIBG) = column_heightᶠᶜᵃ(i, j, k, ibg)
