struct CATKEVerticalDiffusivity{TD, CL, FT, TKE} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    mixing_length :: CL
    turbulent_kinetic_energy_equation :: TKE
    maximum_tracer_diffusivity :: FT
    maximum_tke_diffusivity :: FT
    maximum_viscosity :: FT
    minimum_turbulent_kinetic_energy :: FT
    minimum_convective_buoyancy_flux :: FT
    negative_turbulent_kinetic_energy_damping_time_scale :: FT
end

function CATKEVerticalDiffusivity{TD}(mixing_length::CL,
                                      turbulent_kinetic_energy_equation::TKE,
                                      maximum_tracer_diffusivity::FT,
                                      maximum_tke_diffusivity::FT,
                                      maximum_viscosity::FT,
                                      minimum_turbulent_kinetic_energy::FT,
                                      minimum_convective_buoyancy_flux::FT,
                                      negative_turbulent_kinetic_energy_damping_time_scale::FT) where {TD, CL, TKE, FT}

    return CATKEVerticalDiffusivity{TD, CL, FT, TKE}(mixing_length,
                                                     turbulent_kinetic_energy_equation,
                                                     maximum_tracer_diffusivity,
                                                     maximum_tke_diffusivity,
                                                     maximum_viscosity,
                                                     minimum_turbulent_kinetic_energy,
                                                     minimum_convective_buoyancy_flux,
                                                     negative_turbulent_kinetic_energy_damping_time_scale)
end

CATKEVerticalDiffusivity(FT::DataType; kw...) =
    CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

const CATKEVD{TD} = CATKEVerticalDiffusivity{TD} where TD
const CATKEVDArray{TD} = AbstractArray{<:CATKEVD{TD}} where TD
const FlavorOfCATKE{TD} = Union{CATKEVD{TD}, CATKEVDArray{TD}} where TD

"""
    CATKEVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                             FT = Float64;]
                             mixing_length = CATKEMixingLength(),
                             turbulent_kinetic_energy_equation = CATKEEquation(),
                             maximum_tracer_diffusivity = Inf,
                             maximum_tke_diffusivity = Inf,
                             maximum_viscosity = Inf,
                             minimum_turbulent_kinetic_energy = 1e-15,
                             minimum_convective_buoyancy_flux = 1e-15,
                             negative_turbulent_kinetic_energy_damping_time_scale = 1minute)

Return the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).

!!! note "CATKE vertical diffusivity"
    `CATKEVerticalDiffusivity` is a relatively new turbulence closure. The default
    values for its free parameters are obtained from calibration against large eddy
    simulations. For more details please refer to [Wagner23catke](@cite).

    Use with caution and report any issues with the physics at https://github.com/CliMA/Oceananigans.jl/issues.

Arguments
=========

- `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`;
                         default `VerticallyImplicitTimeDiscretization()`.

- `FT`: Float type; default `Float64`.


Keyword arguments
=================

- `maximum_diffusivity`: Maximum value for tracer, momentum, and TKE diffusivities.
                        Used to clip the diffusivity when/if CATKE predicts
                        diffusivities that are too large.
                        Default: `Inf`.

- `minimum_turbulent_kinetic_energy`: Minimum value for the turbulent kinetic energy.
                                    Can be used to model the presence "background" TKE
                                    levels due to, for example, mixing by breaking internal waves.
                                    Default: 0.

- `negative_turbulent_kinetic_energy_damping_time_scale`: Damping time-scale for spurious negative values of TKE,
                                                        typically generated by oscillatory errors associated
                                                        with TKE advection.
                                                        Default: 1 minute.

Note that for numerical stability, it is recommended to either have a relative short
`negative_turbulent_kinetic_energy_damping_time_scale` or a reasonable
`minimum_turbulent_kinetic_energy`, or both.
"""
function CATKEVerticalDiffusivity(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                  FT = Float64;
                                  mixing_length = CATKEMixingLength(),
                                  turbulent_kinetic_energy_equation = CATKEEquation(),
                                  maximum_tracer_diffusivity = Inf,
                                  maximum_tke_diffusivity = Inf,
                                  maximum_viscosity = Inf,
                                  minimum_turbulent_kinetic_energy = 1e-15,
                                  minimum_convective_buoyancy_flux = 1e-15,
                                  negative_turbulent_kinetic_energy_damping_time_scale = 1.0) where TD

    mixing_length = convert_eltype(FT, mixing_length)
    turbulent_kinetic_energy_equation = convert_eltype(FT, turbulent_kinetic_energy_equation)

    return CATKEVerticalDiffusivity{TD}(mixing_length,
                                        turbulent_kinetic_energy_equation,
                                        convert(FT, maximum_tracer_diffusivity),
                                        convert(FT, maximum_tke_diffusivity),
                                        convert(FT, maximum_viscosity),
                                        convert(FT, minimum_turbulent_kinetic_energy),
                                        convert(FT, minimum_convective_buoyancy_flux),
                                        convert(FT, negative_turbulent_kinetic_energy_damping_time_scale))
end

function with_tracers(tracer_names, closure::FlavorOfCATKE)
    msg = "Tracers must contain :e to represent turbulent kinetic energy " *
          "for `CATKEVerticalDiffusivity`."

    :e ∈ tracer_names || throw(ArgumentError(msg))

    return closure
end


#####
##### Show
#####

function Base.summary(closure::CATKEVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("CATKEVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, closure::FlavorOfCATKE)
    print(io, summary(closure))
    print(io, '\n')
    print(io, "├── maximum_tracer_diffusivity: ", prettysummary(closure.maximum_tracer_diffusivity), '\n',
              "├── maximum_tke_diffusivity: ", prettysummary(closure.maximum_tke_diffusivity), '\n',
              "├── maximum_viscosity: ", prettysummary(closure.maximum_viscosity), '\n',
              "├── minimum_turbulent_kinetic_energy: ", prettysummary(closure.minimum_turbulent_kinetic_energy), '\n',
              "├── negative_turbulent_kinetic_energy_damping_time_scale: ", prettysummary(closure.negative_turbulent_kinetic_energy_damping_time_scale), '\n',
              "├── minimum_convective_buoyancy_flux: ", prettysummary(closure.minimum_convective_buoyancy_flux), '\n',
              "├── mixing_length: ", prettysummary(closure.mixing_length), '\n',
              "│   ├── Cˢ:   ", prettysummary(closure.mixing_length.Cˢ), '\n',
              "│   ├── Cᵇ:   ", prettysummary(closure.mixing_length.Cᵇ), '\n',
              "│   ├── Cᶜc:  ", prettysummary(closure.mixing_length.Cᶜc), '\n',
              "│   ├── Cᶜe:  ", prettysummary(closure.mixing_length.Cᶜe), '\n',
              "│   ├── Cᵉc:  ", prettysummary(closure.mixing_length.Cᵉc), '\n',
              "│   ├── Cᵉe:  ", prettysummary(closure.mixing_length.Cᵉe), '\n',
              "│   ├── Cˡᵒu: ", prettysummary(closure.mixing_length.Cˡᵒu), '\n',
              "│   ├── Cˡᵒc: ", prettysummary(closure.mixing_length.Cˡᵒc), '\n',
              "│   ├── Cˡᵒe: ", prettysummary(closure.mixing_length.Cˡᵒe), '\n',
              "│   ├── Cʰⁱu: ", prettysummary(closure.mixing_length.Cʰⁱu), '\n',
              "│   ├── Cʰⁱc: ", prettysummary(closure.mixing_length.Cʰⁱc), '\n',
              "│   ├── Cʰⁱe: ", prettysummary(closure.mixing_length.Cʰⁱe), '\n',
              "│   ├── CRiᵟ: ", prettysummary(closure.mixing_length.CRiᵟ), '\n',
              "│   └── CRi⁰: ", prettysummary(closure.mixing_length.CRi⁰), '\n',
              "└── turbulent_kinetic_energy_equation: ", prettysummary(closure.turbulent_kinetic_energy_equation), '\n',
              "    ├── CˡᵒD: ", prettysummary(closure.turbulent_kinetic_energy_equation.CˡᵒD),  '\n',
              "    ├── CʰⁱD: ", prettysummary(closure.turbulent_kinetic_energy_equation.CʰⁱD),  '\n',
              "    ├── CᶜD:  ", prettysummary(closure.turbulent_kinetic_energy_equation.CᶜD),  '\n',
              "    ├── CᵉD:  ", prettysummary(closure.turbulent_kinetic_energy_equation.CᵉD),  '\n',
              "    ├── Cᵂu★: ", prettysummary(closure.turbulent_kinetic_energy_equation.Cᵂu★), '\n',
              "    ├── CᵂwΔ: ", prettysummary(closure.turbulent_kinetic_energy_equation.CᵂwΔ), '\n',
              "    └── Cᵂϵ:  ", prettysummary(closure.turbulent_kinetic_energy_equation.Cᵂϵ))
end

