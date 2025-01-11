# Data-driven model enhancement of late-life lithium-ion batteries
Accompanying code to the paper *Data-driven model parametrization of late-life lithium-ion batteries*.

## Dataset
To get started, you will need to download the dataset used in this research. You can download the dataset from the following link:

[https://zenodo.org/doi/10.5281/zenodo.13353324](https://zenodo.org/doi/10.5281/zenodo.13353324)

Once downloaded, please place the dataset in a local directory named `data/` within the root of this repository.

## Setup
1. **Install Julia**: To run the code, you will need to have Julia installed on your machine. We recommend using `juliaup` for easy installation and management of Julia versions: https://julialang.org/downloads/

2. **Create the project environment**: Navigate to the project directory and start Julia by typing `julia` in your terminal. Activate the local environment and install all  the required packages by running:
    ```julia
    julia> using Pkg
    julia> Pkg.activate(".")
    julia> Pkg.instantiate()
    ```

4. **Run the code**: The Jupyter notebook `main.ipynb` generates and displays the results as outlined in the paper. There are two alternatives to open and execute a Jupyter notebook in Julia:
- *Using Jupyter:*
    - Install the [IJulia](https://github.com/JuliaLang/IJulia.jl) package by running `using Pkg; Pkg.add("IJulia")` in the Julia REPL (make sure you have the local environment activated).
    - Launch Jupyter with `using IJulia; notebook()`.
    - Open `main.ipynb` and start executing cells.
- *Using Visual Studio Code (VSCode)*:
    - Install the [Julia extension](https://code.visualstudio.com/docs/languages/julia) in VSCode.
    - Open `main.ipynb` in VSCode.
    - Select the Julia kernel from the top right corner and start executing cells (make sure you have selected the local environemnt).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
