"""
This module contains utility routines used by Sparspak90.
See the comments below in the interface section of the module.
"""
module SpkUtilities

_BIGGY() = typemax(Float64)

"""
    extend(v::Vector, newlen::Integer, flagval=zero(eltype(v)))

Change the size (smaller or larger).    
- The contents of the vectors / arrays are preserved.
- The changes in size may be positive or negative.
- flagval is the initialization value of the new parts of the arrays.
    If it is absent,  the default is zero (for numerical arrays).
"""
function extend(v::Vector, newlen::Integer, flagval=zero(eltype(v)))
    len = length(v)
    v = resize!(v, newlen)
    if newlen > len
        for i in len+1:newlen
            v[i] = flagval
        end
    end
    return v
end

end # module 


