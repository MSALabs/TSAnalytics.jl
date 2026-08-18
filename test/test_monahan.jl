@testset "partrans/invpartrans (R tanh convention, exact hand-verified)" begin
    # handoff/stage-4.2-monahan-handoff.md section 3/6: independently
    # re-derived (not just transcribed) via a from-scratch Python
    # transcription of R's actual arima.c source in this session.
    @test isapprox(partrans([0.5, -0.3, 0.2]),
                    [0.654235633768312, -0.4090939097636928, 0.197375320224904]; atol=1e-10)

    # every p from 0 to 5, and a case with all-zero raw params (fixed point)
    cases = Dict(
        Float64[]                       => Float64[],
        [0.1]                           => [0.09966799462495582],
        [0.5, -0.3]                     => [0.5967377136001258, -0.2913126124515909],
        [0.5, -0.3, 0.2, 0.1]           => [0.6345636314130373, -0.3683203401642629,
                                             0.13216896659502933, 0.09966799462495582],
        [0.9, -0.8, 0.7, -0.6, 0.5]     => [2.1660236934637527, -2.802613380291481,
                                             2.4433745003962937, -1.4233181262372339,
                                             0.46211715726000974],
        [0.0, 0.0, 0.0]                 => [0.0, 0.0, 0.0],
        [-1.5, 2.0, -0.5, 1.2]          => [0.7981794088802965, 0.1578585962623975,
                                             -0.8063609149901247, 0.8336546070121552],
    )
    for (raw, expected) in cases
        result = partrans(raw)
        @test isapprox(result, expected; atol=1e-8)
        # round-trip recovers the original raw parameters, for every p tried
        if !isempty(raw)
            @test isapprox(invpartrans(result), raw; atol=1e-6)
        end
    end

    # p=0 and p=1 edge cases explicitly (p=1 needs no Durbin-Levinson recursion)
    @test partrans(Float64[]) == Float64[]
    @test invpartrans(Float64[]) == Float64[]
    @test isapprox(partrans([0.5]), [tanh(0.5)]; atol=1e-12)
    @test isapprox(invpartrans([tanh(0.5)]), [0.5]; atol=1e-10)

    # Round-trip, for many random raw vectors across p=1..6. NOTE: only the
    # intermediate PACF values (tanh(raw)) are individually bounded in
    # (-1,1) -- the FINAL AR/MA coefficients after the Durbin-Levinson
    # recursion are not (e.g. partrans([0.9,-0.8,0.7,-0.6,0.5])'s first
    # component is ~2.17), since Monahan's transform guarantees
    # *stationarity* (characteristic roots outside the unit circle), a
    # weaker condition than "each coefficient's magnitude < 1". The one
    # invariant that IS always true: `new[p]` is never touched by the
    # recursion (every iteration only rewrites positions `1:j-1`), so the
    # last coefficient always equals `tanh(raw[end])` exactly.
    Random.seed!(42)
    for p in 1:6, trial in 1:5
        raw = 4 .* (rand(p) .- 0.5)  # spread beyond (-1,1) to actually exercise tanh's squashing
        result = partrans(raw)
        @test isapprox(result[end], tanh(raw[end]); atol=1e-12)
        @test isapprox(invpartrans(result), raw; atol=1e-6)
    end

    # container-agnostic: any AbstractVector{<:Real}, not just Vector{Float64}
    @test isapprox(partrans(Float32[0.5, -0.3, 0.2]),
                    [0.654235633768312, -0.4090939097636928, 0.197375320224904]; atol=1e-5)
    @test isapprox(partrans(0.1:0.1:0.3), partrans([0.1, 0.2, 0.3]); atol=1e-12)
end
