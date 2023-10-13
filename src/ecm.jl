
@variables t
D = Differential(t)

@mtkmodel ECM begin
    @parameters begin
        Q = 94.0
        R0 = 0.5e-3
        R1 = 0.5e-3
        τ1 = 120
    end
    @variables begin
        i(t)
        v(t)
        vr(t)
        v1(t)
        ocv(t)
        soc(t)
    end
    @structural_parameters begin
        focv
        fi
    end
    @equations begin
        D(soc) ~ i / (Q * 3600.0)
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1) # check sign
        vr ~ i * R0
        ocv ~ focv(soc)
        i ~ fi(t)
        v ~ ocv + vr + v1
    end
end

function build_loss_function(prob, model, df)
    loss(x) = begin
        @unpack u0, p = x
        newprob = remake(prob; u0, p)
        sol = solve(newprob, saveat=df.t)
        v = sol[model.v]
        sum(abs2, v - df.v)
    end
end

