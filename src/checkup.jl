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


## plots
function plot_checkup_profile(file)
    df = read_basytec(file)

    # fig = Figure(size=(1200, 400))
    # fig = Figure(size=(900, 300))
    fig = Figure(size=(600, 200), fontsize=11, figure_padding=5)
    gl = GridLayout(fig[1,1])
    # fig = Figure(size=(515, 172), fontsize=11)
    ax1 = Axis(gl[1, 1])
    ax2 = Axis(gl[2, 1])
    n = nrow(df)
    lines!(ax1, df[1:60:n, "Time[h]"], df[1:60:n, "U[V]"], color=Cycled(1))
    lines!(ax2, df[1:60:n, "Time[h]"], df[1:60:n, "I[A]"], color=Cycled(2))
    xlims!(ax1, (df[begin, "Time[h]"], df[end, "Time[h]"]))
    xlims!(ax2, (df[begin, "Time[h]"], df[end, "Time[h]"]))
    ax1.ylabel = "Voltage (V)"
    ax2.ylabel = "Current (A)"
    ax2.xlabel = "Time (h)"
    hidexdecorations!(ax1, grid=false, ticks=false)
    rowgap!(gl, 5)
    fig
end

function plot_ocvs(files)
    s = 0:0.001:1.0
    ylabel = "Voltage in V"
    xlabel = "SOC in p.u."
    colormap = :dense
    colorrange = (0.6, 1.0)

    fig1 = Figure(size=(800, 400))
    ax1 = Axis(fig1[1, 1]; xlabel, ylabel, title="Charge pOCV")
    ax2 = Axis(fig1[1, 2]; xlabel, title="Discharge pOCV")
    linkyaxes!(ax1, ax2)
    hideydecorations!(ax2, ticks=false, grid=false)

    fig2 = Figure(size=(900, 400))
    ax3 = Axis(fig2[1, 1]; xlabel, ylabel, title="Mean pOCV")
    ax4 = Axis(fig2[1, 2]; xlabel, title="OCV Hystheresis")

    fig3 = Figure(size=(900, 400))
    ax5 = Axis(fig3[1, 1]; xlabel, ylabel, title="Mean pOCV")
    ax6 = Axis(fig3[1, 2]; xlabel, title="Deviation")

    focv = fresh_focv()

    for file in files
        df = read_basytec(file)

        # charge ocv
        ocv_c, cap_c = calc_cocv(df)
        c = s .* cap_c
        lines!(ax1, s, ocv_c(c); color=cap_c / 4.9, colorrange, colormap)

        # discharge ocv
        ocv_d, cap_d = calc_docv(df)
        c = s .* cap_d
        lines!(ax2, s, ocv_d(c); color=cap_d / 4.9, colorrange, colormap)

        # mean ocv
        pocv, cap = calc_pocv(df)
        c = s .* cap
        lines!(ax3, s, pocv(c); color=cap / 4.9, colorrange, colormap)
        lines!(ax5, s, pocv(c); color=cap / 4.9, colorrange, colormap)

        # hyst
        lines!(ax4, s, pocv(c) - ocv_c(c); color=cap / 4.9, colorrange, colormap)

        # degradation
        if !occursin("LGL13818", file)
            lines!(ax6, s, pocv(c) - focv(s); color=cap / 4.9, colorrange, colormap)
        end
    end

    Colorbar(fig1[:, 3]; colorrange, colormap, label="SOH")
    Colorbar(fig2[:, 3]; colorrange, colormap, label="SOH")
    Colorbar(fig3[:, 3]; colorrange, colormap, label="SOH")
    return fig1, fig2, fig3
end

