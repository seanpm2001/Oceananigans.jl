using CUDA

using Oceananigans.Fields: AbstractField, compute_at!
using Oceananigans.LagrangianParticleTracking: LagrangianParticles
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: architecture, device

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

fetch_output(output, model) = output(model)

function fetch_output(field::AbstractField, model)
    compute_at!(field, time(model))
    field_masking_event = mask_immersed_field!(field)
    wait(device(architecture(field)), field_masking_event)

    return parent(field)
end

function fetch_output(lagrangian_particles::LagrangianParticles, model)
    particle_properties = lagrangian_particles.properties
    names = propertynames(particle_properties)
    return NamedTuple{names}([getproperty(particle_properties, name) for name in names])
end

convert_output(output, writer) = output

function convert_output(output::AbstractArray, writer)
    if architecture(output) isa GPU
        output_array = writer.array_type(undef, size(output)...)
        copyto!(output_array, output)
    else
        output_array = convert(writer.array_type, output)
    end

    return output_array
end

# Need to broadcast manually because of https://github.com/JuliaLang/julia/issues/30836
convert_output(outputs::NamedTuple, writer) =
    NamedTuple(name => convert_output(outputs[name], writer) for name in keys(outputs))

"""
    fetch_and_convert_output(output, model, writer)

Computes and converts output to be written to disk. This function is called
every time when writing to disk.
"""
function fetch_and_convert_output(output, model, writer)
    fetched = fetch_output(output, model)
    return convert_output(fetched, writer)
end
