"""
This module contains utility routines used by Sparspak90.
See the comments below in the interface section of the module.
"""
module SpkUtilities

_BIGGY(T) = typemax(T)


#     __extend(v::Vector, newlen::Integer, flagval=zero(eltype(v)))
# Change the size (smaller or larger).    
# - The contents of the vectors / arrays are preserved.
# - The changes in size may be positive or negative.
# - flagval is the initialization value of the new parts of the arrays.
#     If it is absent,  the default is zero (for numerical arrays).
function __extend(v::Vector, newlen::Integer, flagval=zero(eltype(v)))
    len = length(v)
    v = resize!(v, newlen)
    if newlen > len
        for i in len+1:newlen
            v[i] = flagval
        end
    end
    return v
end


#     __extend(v::Matrix, newrow::Integer, newcol::Integer, flagval=zero(eltype(v)))
# Change the size of a matrix (smaller or larger).
function __extend(v::Matrix, newrow::Integer, newcol::Integer, flagval=zero(eltype(v)))
    tempv = fill(zero(eltype(v)), newrow, newcol)
    lencol = size(v, 2)
    lenrow = size(v, 1); 

    r = min(lenrow, newrow);
    c = min(lencol, newcol); 
    tempv[1:r, 1:c] .= @view v[1:r, 1:c]
    if (r < newrow)  tempv[r + 1:newrow, 1:newcol] .= flagval end
    if (c < newcol)  tempv[1:newrow, c + 1:newcol] .= flagval end

    return tempv
end

end # module 


