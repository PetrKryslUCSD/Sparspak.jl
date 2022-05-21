#
# """ This module contains a collection of subroutines for finding and
# manipulating elimination trees, finding (weighted) postorderings of
#  them, and related functions.
# """
#
module SpkETree

mutable struct ETree{IT}
    nv::IT
    parent::Vector{IT}
end

"""
"""
function ETree(nv::IT) where {IT}
    parent = fill(zero(IT), nv)
    return ETree(nv, parent)
end

end