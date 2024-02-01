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

function load_data(files)
    data = Dict{Symbol,DataFrame}()
    for file in files
        id = get_cell_id(file) |> Symbol
        df = read_basytec(file)
        data[id] = df
    end
    return data
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
    # file = "data/check-ups/2098LG_INR21700-M50L_SammyLGL13818NewFullCU.txt"
    file = "data/check-ups/2097LG_INR21700-M50L_SammyLGL09107NewFullCU.txt"
    df = read_basytec(file)
    ocv, cap = calc_pocv(df)
    focv = soc -> ocv(soc * cap)
    return focv
end

## internal resistance
function calc_rint(df; timestep=1)
    cycles = 9
    timestep = timestep / 3600 # seconds

    line = 49 # 51
    i = 4.8 * 0.3

    # line = 53 # 55
    # i = 4.8 * 2 // 3

    resistances = Float64[]
    for cycle in 1:cycles
        # initial_voltage
        df2 = copy(df)
        filter!(:Line => ∈(line), df2)
        filter!(:Count => ==(cycle), df2)
        # idx = isapprox.(df2[:, "I[A]"], 0.0, atol=1e-3) |> findfirst
        # @show idx
        init_time = df2[begin, "Time[h]"]
        init_voltage = df2[begin, "U[V]"]

        idx2 = findfirst(>(init_time + timestep), df[:, "Time[h]"])
        voltage = df[idx2, "U[V]"]

        r = abs(voltage - init_voltage) / i
        append!(resistances, r)
    end
    return resistances
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


## plots
function plot_checkup_profile(df)
    fig = Figure(size=(600, 200), fontsize=11, figure_padding=5)
    gl = GridLayout(fig[1, 1])
    ax1 = Axis(gl[1, 1])
    ax2 = Axis(gl[2, 1])

    t = 1:60:size(df, 1) # 1 sample per minute
    lines!(ax1, df[t, "Time[h]"], df[t, "U[V]"], color=Cycled(1))
    lines!(ax2, df[t, "Time[h]"], df[t, "I[A]"], color=Cycled(2))
    xlims!(ax1, (df[begin, "Time[h]"], df[end, "Time[h]"]))
    xlims!(ax2, (df[begin, "Time[h]"], df[end, "Time[h]"]))

    ax1.ylabel = "Voltage (V)"
    ax2.ylabel = "Current (A)"
    ax2.xlabel = "Time (h)"

    hidexdecorations!(ax1, grid=false, ticks=false)
    rowgap!(gl, 5)
    fig
end

function plot_ocvs(data)
    s = 0:0.001:1.0
    colormap = :dense
    colorrange = (0.6, 1.0)

    fig = Figure(size=(350, 400), fontsize=11, figure_padding=7)
    gl = GridLayout(fig[1, 1])

    Colorbar(gl[1, 1]; colorrange, colormap, label="SOH", vertical=false) #, flipaxis=false)
    ax1 = Axis(gl[2, 1]; ylabel="OCV (V)") # mean pOCV
    ax2 = Axis(gl[3, 1]; xlabel="SOC", ylabel="δV (mV)")
    rowgap!(gl, 6)

    # ax1 = Axis(fig[1, 1]; ylabel="OCV (V)") # mean pOCV
    # ax2 = Axis(fig[2, 1]; xlabel="SOC", ylabel="δV (mV)")
    # Colorbar(fig[:, 2]; colorrange, colormap, label="SOH") #, vertical=false, flipaxis=false)

    # s2 = 0.5:0.001:1.0
    # ax3 = Axis(fig, bbox = BBox(350, 475, 290, 370))

    focv = fresh_focv()

    for (id, df) in data
        # mean ocv
        pocv, cap = calc_pocv(df)
        c = s .* cap
        lines!(ax1, s, pocv(c); color=cap / 4.9, colorrange, colormap)

        # c2 = s2 .* cap
        # lines!(ax3, s2, pocv(c2); color=cap / 4.9, colorrange, colormap)

        # degradation
        if id != :LGL09107 # :LGL13818
            δv = (pocv(c) - focv(s)) * 1e3 # mV
            lines!(ax2, s, δv; color=cap / 4.9, colorrange, colormap)
        end
    end

    hidexdecorations!(ax1, grid=false, ticks=false)
    return fig
end

function plot_rints(data; timestep=1)
    soc = 0.9:-0.1:0.1
    colormap = :dense
    colorrange = (0.6, 1.0)

    fig = Figure(size=(359, 350), fontsize=11)
    ax = Axis(fig[1, 1]; xlabel="SOC", ylabel="Resistance (mΩ)")
    Colorbar(fig[1, 2]; colorrange, colormap, label="SOH")

    for (id, df) in data
        soh = calc_capa_cc(df) / 4.9
        r = calc_rint(df; timestep) * 1e3 # mΩ

        scatter!(ax, soc, r; color=soh, colorrange, colormap)
    end

    return fig
end

fig = plot_ocvs(files)


save("figs/pOCV.pdf", fig, pt_per_unit=1)

