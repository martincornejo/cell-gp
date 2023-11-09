## 
function read_basytec(file; kwargs...)
    columns = [
        "Time[h]",
        "DataSet",
        "t-Step[h]",
        "Line",
        "Command",
        "U[V]",
        "I[A]",
        "Ah[Ah]",
        "Ah-Step",
        "Wh[Wh]",
        "Wh-Step",
        "T1[°C]",
        "Cyc-Count",
        "Count",
        "State",
    ]
    return CSV.File(file; header=columns, delim='\t', comment="~", kwargs...) |> DataFrame
end

## cell id
function get_cell_id(file)
    pattern = r"(MF\d+|LGL\d+)"
    match_result = match(pattern, file)
    return match_result |> first
end


## capacity
function calc_capa_cccv(df; line=(21, 22))
    cc, cv = line

    df_cc = filter(:Line => ∈(cc), df)
    df_cv = filter(:Line => ∈(cv), df)

    cap_cc = df_cc[end, "Ah-Step"] |> abs
    cap_cv = df_cv[end, "Ah-Step"] |> abs
    return cap_cc + cap_cv
end

function calc_capa_cc(df; line=21)
    df_cc = filter(:Line => ∈(line), df)
    cap_cc = df_cc[end, "Ah-Step"] |> abs
    return cap_cc
end


## OCV
function calc_cocv(df)
    dfc = filter(:Line => ∈(29), df)
    cap = dfc[end, "Ah-Step"] |> abs
    v = dfc[:, "U[V]"]
    s = dfc[:, "Ah-Step"]
    return LinearInterpolation(v, s), cap
end

function calc_docv(df)
    dfc = filter(:Line => ∈(31), df)
    cap = dfc[end, "Ah-Step"] |> abs
    v = reverse(dfc[:, "U[V]"])
    s = reverse(dfc[:, "Ah-Step"] .+ cap)
    return LinearInterpolation(v, s), cap
end

function calc_pocv(df)
    ocv_c, cap_c = calc_cocv(df)
    ocv_d, cap_d = calc_docv(df)

    cap = min(cap_c, cap_d)
    ocv = c -> (ocv_c(c) + ocv_d(c)) / 2
    return ocv, cap
end

function fresh_focv()
    file = "data/check-ups/2098LG_INR21700-M50L_SammyLGL13818NewFullCU.txt"
    df = read_basytec(file)
    ocv, cap = calc_pocv(df)
    focv = soc -> ocv(soc * cap)
    return focv
end


## profile
function load_profile(df)
    df = filter(:Line => ∈(35), df)

    df[:, :t] = (df[:, "Time[h]"] .- df[begin, "Time[h]"]) * 3600 # hours -> seconds

    i = ConstantInterpolation(df[:, "I[A]"], df.t)
    v = ConstantInterpolation(df[:, "U[V]"], df.t)
    s = ConstantInterpolation(df[:, "Ah-Step"], df.t)
    T = ConstantInterpolation(df[:, "T1[°C]"], df.t)

    return (; i, s, v, T)
end

function sample_dataset(data, tt)
    return DataFrame(
        :t => tt,
        :i => data.i(tt),
        :v => data.v(tt),
        :s => data.s(tt),
        :T => data.T(tt)
    )
end
