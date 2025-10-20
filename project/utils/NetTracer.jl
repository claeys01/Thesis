module NetTrace

# Reusable, version-stable network tracer for Flux models.
# - Works with Chain and structs exposing `.layers` (e.g., your Encoder/Decoder).
# - Uses Flux.outputsize for Conv/ConvTranspose output shape math (robust across versions).
# - Prints a concise per-layer summary and returns (y, rows) for programmatic use.

using Flux

export net_summary, quick_overview

# --- helpers -----------------------------------------------------------------

layer_name(l) = string(nameof(typeof(l)))

# Accept either a Chain, or a struct with a `.layers` field that is a Chain
get_chain(m) = m isa Chain ? m : getfield(m, :layers)

# Conv/ConvTranspose metadata using Flux.outputsize (no NNlib internals)
function conv_meta(x, l::Flux.Conv)
    outsz = Flux.outputsize(l, size(x))  # (Hout, Wout, Cout, N)
    return (; kind="Conv",
            kernel=size(l.weight)[1:2],
            cin=size(l.weight, 3),
            cout=size(l.weight, 4),
            stride=l.stride, pad=l.pad, dilation=l.dilation,
            out_hw=outsz[1:2])
end

function conv_meta(x, l::Flux.ConvTranspose)
    outsz = Flux.outputsize(l, size(x))  # (Hout, Wout, Cout, N) for transposed convs too
    # ConvTranspose weight dims are (kH,kW,Cout,Cin)
    return (; kind="ConvTranspose",
            kernel=size(l.weight)[1:2],
            cin=size(l.weight, 4),
            cout=size(l.weight, 3),
            stride=l.stride, pad=l.pad, dilation=l.dilation,
            out_hw=outsz[1:2])
end

conv_meta(_, _) = nothing

# crude device hint without hard CUDA dependency
_devhint(x) = occursin("CuArray", string(typeof(x))) ? "GPU" : "CPU"

# --- core tracer -------------------------------------------------------------
"""
    net_summary(model, x; prefix="", check_nan=false, io=stdout, assert_hw=false)

Recursively prints a per-layer I/O summary and returns `(y, rows)` where
`y` is the forward output and `rows` is a vector of named tuples:
`(idx, layer, in, out, meta, nan_or_inf)`.

- `check_nan=true` does a (cheap) NaN/Inf scan after each layer (convert to Array).
- `assert_hw=true` asserts that any Conv/ConvTranspose has positive output H×W.
- `io` controls where the summary is printed (defaults to `stdout`).
"""
function net_summary(model, x; prefix="", check_nan=false, io=stdout, assert_hw=false)
    m = get_chain(model)
    y = x
    rows = NamedTuple[]
    for (i, layer) in enumerate(m)
        in_sz = size(y)
        println(io, "$prefix[$i] $(layer_name(layer))  in=$(in_sz)  dev=$(_devhint(y))")

        info = conv_meta(y, layer)
        if info !== nothing
            println(io, "$prefix     $(info.kind): kernel=$(info.kernel)  $(info.cin)=>$(info.cout)  stride=$(info.stride) pad=$(info.pad) dil=$(info.dilation)")
            println(io, "$prefix     expected out H×W = $(info.out_hw)")
            if assert_hw
                oh, ow = info.out_hw
                @assert oh ≥ 1 && ow ≥ 1 "Invalid output spatial size ($oh,$ow) for $(layer_name(layer))"
            end
        end

        # Recurse into nested Chains; otherwise apply layer
        y = layer isa Chain ?
            (net_summary(layer, y; prefix=prefix*"    ", check_nan=check_nan, io=io, assert_hw=assert_hw))[1] :
            layer(y)

        out_sz = size(y)
        println(io, "$prefix     out=$(out_sz)\n")

        bad = false
        if check_nan
            # best-effort, skip if huge (conversion may be expensive)
            try
                bad = any(isnan, Array(y)) || any(isinf, Array(y))
                bad && println(io, "$prefix     WARNING: NaN/Inf after this layer!")
            catch
                # ignore if conversion too costly/unsupported
            end
        end

        push!(rows, (; idx=i, layer=layer_name(layer), in=in_sz, out=out_sz, meta=info, nan_or_inf=bad))
    end
    return y, rows
end

"""
    quick_overview(model, x; check_nan=false, assert_hw=false)

Convenience wrapper for models that are either a `Chain` or a struct with a
`.layers` field. Prints to stdout. Returns `(y, rows)`.
"""
quick_overview(model, x; check_nan=false, assert_hw=false) =
    net_summary(model, x; check_nan=check_nan, assert_hw=assert_hw)

end # module
