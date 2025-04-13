# Grid objects contain a representation of an h by k rectangular grid.
# h - the number of rows ("height")
# k is the number of columns ("kolumns").
#  v is an h by k integer array

module  SpkGrid

mutable struct Grid{IT}
    h::IT
    k::IT
    v::Matrix{IT}
end

"""
    Grid(h::IT, k::IT) where {IT}

Construct a grid with a given number of spacings.
"""
function Grid(h::IT, k::IT) where {IT}
    n = h * k; 
    v = zeros(IT, h, k)
    for i in 1:h
        for j in 1:k
            v[i, j] = k * (i - 1) + j
        end
    end
    return Grid(h, k, v)
end

end 







