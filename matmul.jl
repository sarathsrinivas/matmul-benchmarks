using LoopVectorization

const CACHE = 180

function matmul_turbo!(C, A, B, cr=axes(C,1),cc=axes(C,2), ac=axes(A,2))
    @turbo for i in cr, j in cc
                Cij = zero(eltype(C))
                for k in ac
                    Cij += A[i,k] * B[k,j]
                end
                C[i,j] += Cij
            end
end

function quadrapule(ar)
    r  = ar[1]
    c = ar[2]

    fr = first(r)
    er = last(r)
    mr = (fr + er) >> 1

    fc = first(c)
    ec = last(c)
    mc = (fc + ec) >> 1

    ar11 = (fr:mr, fc:mc)
    ar21 = ((1+mr):er, fc:mc)
    ar12 = (fr:mr, (1+mc):ec)
    ar22 = ((1+mr):er, (1+mc):ec)

    return ar11, ar21, ar12, ar22
end


function matmul_dq!(C,A,B,cr=axes(C), ar=axes(A), br=axes(B))
                    #ct = Array{eltype(C)}(undef,CACHE,CACHE), at=similar(ct), bt=similar(ct))

    if length(cr[1]) < CACHE
        matmul_turbo!(C,A,B,cr[1],cr[2],ar[2])
        return nothing
    end

    ar11, ar21, ar12, ar22 = quadrapule(ar)
    br11, br21, br12, br22 = quadrapule(br)
    cr11, cr21, cr12, cr22 = quadrapule(cr)

    matmul_dq!(C,A,B,cr11,ar11,br11)
    matmul_dq!(C,A,B,cr11,ar12,br21)

    matmul_dq!(C,A,B,cr21,ar21,br11)
    matmul_dq!(C,A,B,cr21,ar22,br21)

    matmul_dq!(C,A,B,cr12,ar11,br12)
    matmul_dq!(C,A,B,cr12,ar12,br22)

    matmul_dq!(C,A,B,cr22,ar21,br12)
    matmul_dq!(C,A,B,cr22,ar22,br22)

end
