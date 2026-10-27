using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Distributions
using Plots
using Random
using PyCall
using CSV
using DataFrames
using Statistics
using JSON


const OBS_NOISE = 0.1       #szum użyty do WYGENEROWANIA y_obs ==> jak bardzo dane są naprawdę zaszumione
const OBS_REL_UNC = 0.1     #niepewność względna zakładana w Gamma ==> jak bardzo EKI wierzy w dane (docelowo zastąpione przez niepewność obserwabli exp)
const J = 100               #wielkość zespołu (ensemble'a)
const N_ITER = 50           #maksymalna liczba iteracji algorytmu
const DATA_SEED = 105       #seed szumu obserwabli (docelowo znika z kodu)
const EKI_SEED = 106        #seed zespołu początkowego i szumu w EKI

rng_data = MersenneTwister(DATA_SEED)
rng_eki  = MersenneTwister(EKI_SEED)

dir_name = "noise=$(OBS_NOISE)_unc=$(OBS_REL_UNC)_J=$(J)_iter_max=$(N_ITER)_dseed=$(DATA_SEED)_eseed=$(EKI_SEED)"
mkpath(dir_name)

#------------------------------------------------------------
#         WCZYTYWANIE INFORMACJI Z PLIKÓW ZEWNĘTRZNYCH
#------------------------------------------------------------
#------ parametry ------
config = JSON.parsefile("config_file.json")
fixed_params = config["fixed_parameters"]
est_params_config = config["estimated_parameters"]
est_params_names = collect(keys(est_params_config))     #nazwy parametrów które mają być fitowane

#------ model ------
pushfirst!(PyVector(pyimport("sys")."path"), "")
py_model = pyimport("generuj_dane_zaszumione")      #nazwa pliku z modelem w pythonie; plik musi być w tym samym folderze co ten skrypt

#------------------------------------------------------------
#                        FORWARD FUNCTION
#------------------------------------------------------------
function model_wrapper(x::AbstractVector)
    #inicjalizowanie słownika parametrów ==> wpierw dodajemy parametry, które mają być stałe
    params = copy(fixed_params)

    #uzupełniamy słownik o parametry, które mają być estymowane
    for (i, name) in enumerate(est_params_names)
        params[name] = x[i]
    end

    #wywołanie modelu "nakarmionego" danymi parametrami
    #!UWAGA: tutaj docelowo trzeba nie zapomnieć o podmienieniu funkcji na tą właściwą modelowi NK
    return convert(Vector{Float64}, py_model.simple_linear_equations(params))
end

function G_ensemble(u_ensemble)
    N_ens = size(u_ensemble, 2)
    sample_output = model_wrapper(u_ensemble[:, 1])
    out_dim = length(sample_output)
    
    G = zeros(out_dim, N_ens) 
    for i in 1:N_ens
        G[:, i] = model_wrapper(u_ensemble[:, i])
    end
    return G
end

#------------------------------------------------------------------------------------------------------------------------
#         GENEROWANIE ZASZUMIONEGO WEKTORA OBSERWABLI EKSPERYMENTALNYCH ==> TO DOCELOWO ZNIKA Z KODU
#------------------------------------------------------------------------------------------------------------------------
#!zastąpione to zostanie importowaniem pliku w którym są wyznaczone wartości poszczególnych obserwabli wraz z niepewnościami
#UWAGA: ten blok musi być PO definicji `model_wrapper` (w skrypcie uruchamianym od zera funkcja wywołana przed definicją => UndefVarError)

#wartości nominalne tylko dla parametrów ESTYMOWANYCH (x1 jest stały i jest w config_file.json); podane po nazwach, więc kolejność kluczy w słowniku
#z pliku JSON nie ma znaczenia; `X_NOMINAL` ma dokładnie tyle elementów co `est_params_names` i w tej samej kolejności
NOMINAL = Dict("x2" => 9.0, "x3" => 8.0, "x4" => 7.0, "x5" => 6.0, "x6" => 5.0, "x7" => 4.0, "x8" => 3.0, "x9" => 2.0, "x10" => 1.0)
X_NOMINAL = [NOMINAL[nm] for nm in est_params_names]       #bez `const`: w trwającej sesji REPL ponowne uruchomienie skryptu zmieniającego const daje ostrzeżenie o redefinicji

y_nominal = model_wrapper(X_NOMINAL)
y_obs = y_nominal .* (1.0 .+ OBS_NOISE .* randn(rng_data, length(y_nominal)))
Gamma = Diagonal((OBS_REL_UNC .* abs.(y_obs)).^2)

#POMYSŁ: przy pomocy macierzy Gamma możemy wprowadzić powiązanie między różnymi obserwablami dla konkretnych eksperymentów (tj. pokazać, że obserwabla
#x i y pochodzą z tego samego eksperymentu); trzeba to zrobić poprzez dodanie wartości kowariancji poza diagonalą (je można wyliczyć); rzucam ten pomysł tutaj 
#żeby nie został w innym miejscu usunięty

#------------------------------------------------------------
#                   INICJALIZACJA EKI
#------------------------------------------------------------
#zaczynamy od wylosowania parametów, od których startujemy
#!UWAGA: docelowo tutaj wrzucamy mechanizm Latin Cube Sample (jest do tego dedykowana biblioteka); teraz jest losowanie z rozkładu jednostajnego
#zrobione po to, by można testować, czy poprawnie działa importowanie przedziałów z pliku `config_file`

#UWAGA: w ParameterDistributions rozkład podany w `Parameterized(...)` jest zawsze w przestrzeni NIEograniczonej, a `constraint` mapuje go na fizyczną
#(dla `bounded_below(0.0)` jest to ϕ = exp(u)). `Parameterized(Uniform(0, 20))` z `bounded_below(0.0)` dawałoby więc ϕ = exp(u) ∈ (1, ~5e8), a nie (0, 20).
#Dlatego zespół początkowy losujemy sami w przestrzeni FIZYCZNEJ z przedziałów z config_file i dopiero przeliczamy na przestrzeń nieograniczoną;
#tak samo będzie można podstawić Latin Hypercube Sampling (wystarczy podmienić linię z `rand`).
#Rozkład w `prior` służy tu więc tylko jako nośnik ograniczenia `bounded_below(0.0)` (transformacja exp/log); nie jest z niego losowany zespół.

prior_distributions = [Dict(
    "name" => name,
    "distribution" => Parameterized(Normal(0.0, 1.0)),      #placeholder (patrz wyżej): nie jest używany do losowania
    "constraint" => bounded_below(0.0)          #nakłada warunek, że parametry muszą być większe od 0; biblioteka EKI na czas optymalizacji nałoży w tle
    )                                           #transformację logarytmiczną, co zabezpieczy parametry przed przyjmowaniem wartości ujemnych ==> do modelu
    for name in est_params_names]               #będą jednak wysyłane spowrotem jako fizyczne wartości a nie ich logarytmiczne transformaty

prior = ParameterDistribution(prior_distributions)

#zespół początkowy w przestrzeni fizycznej: równomiernie z [min, max] dla każdego parametru (kolejność wierszy = est_params_names)
p_min = [Float64(est_params_config[nm]["min"]) for nm in est_params_names]
p_max = [Float64(est_params_config[nm]["max"]) for nm in est_params_names]
phi_init = p_min .+ (p_max .- p_min) .* rand(rng_eki, length(est_params_names), J)       #N_par x J

#EKP pracuje w przestrzeni nieograniczonej (log), więc przeliczamy kolumna po kolumnie
initial_ensemble = reduce(hcat, [transform_constrained_to_unconstrained(prior, phi_init[:, j]) for j in 1:J])
ekp = EnsembleKalmanProcess(initial_ensemble, y_obs, Gamma, Inversion())

#------------------------------------------------------------
#                   FUNKCJE POMOCNICZE
#------------------------------------------------------------
#`member_rmse`: 
#funkcja służąca do wyznaczenia RMSE osobno dla każdego członka zespołu (bibliotego zwraca tylko wartość średnią po członkach, a w iteracjach,
#przynajmniej na wczesnym etapie, chcemy mieć możliwość śledzenia, czy RMSE w obrębie jednego zespołu zbiega do jednej wartości. Wyznaczane w tej funkcji RMSE
#jest znormalizowane przez macierz Gamma, jest więc w założeniu miarą zgodności z danymi względem ich niepewności - wartość ~1 oznacza, że model zwraca wartości
#o porównywalnej niepewności do niepewności danych pomiarowych, wartość znacznie mniejsza od 1 sugeruje overfitting; można to interpretować jako jeżeli member_rmse=k,
#to błąd danego członka zespołu to k*σ; wartość ta jest przydatna do diagnostyki, na niej częściowo opiera się DataMisfitController:

#DataMisfitController kończy działanie, gdy czas algorytmu, u nas: suma sztucznych kroków delta_t, dojdzie do T = 1 (domyślny parametr terminate_at = 1.0). Kroki
#czasowe są wyznaczane w oparciu o parametr Φ_j = ‖Γ^(−1/2)(G(θ_j) − y)‖^2; parametr ten jest powiązany z funkcją `member_rmse` zależnością member_rmse_j = √(Φ_j/n),
#gdzie n - długość wektora obserwabli; przy T=1 misfit powinien być rzędu poziomu szumu (jak dopasowanie zwraca wartości trochę mniejsze od 1 to też jest
#akceptowalne, bo część szumu jest dopasowywana)

#ta wersja funkcji powinna działać również dla Gammy która nie jest diagonalna

function member_rmse(ekp, G::AbstractMatrix)            
    R  = G .- get_obs(ekp; build=true)
    WR = EnsembleKalmanProcesses.lmul_obs_noise_cov_inv(ekp, R)
    return sqrt.(vec(sum(R .* WR, dims=1)) ./ size(R, 1))
end

#`loss_at_mean`: zwraca to samo co `compute_loss_at_mean(ekp)``, ale można policzyć to dla dowolnego G - przydatne dla zespołu końcowego, którego funkcja
#z biblioteki nie widzi, bo zespół końcowy nie jest przekazywany do `update_ensemble!`
function loss_at_mean(ekp, G::AbstractMatrix)
    r = vec(mean(G, dims=2)) .- get_obs(ekp; build=true)
    return dot(r, EnsembleKalmanProcesses.lmul_obs_noise_cov_inv(ekp, r)) / length(r)
end

#------------------------------------------------------------
#                   ZAPIS STANU W PĘTLI
#------------------------------------------------------------
obs_means = Vector{Vector{Float64}}()
obs_stds  = Vector{Vector{Float64}}()
df_ensemble_rows = []
df_summary_rows = []
sigma_obs = sqrt.(diag(Gamma))          #!UWAGA: poprawne tylko gdy Gamma diagonalna; używane tylko roboczo w końcowym printowaniu, docelowo zniknie
 
#Nazwy kolumn z parametrami z pliku konfiguracyjnego
param_syms = Tuple(Symbol(nm) for nm in est_params_names)
mean_syms  = Tuple(Symbol(nm, "_mean") for nm in est_params_names)
std_syms   = Tuple(Symbol(nm, "_std")  for nm in est_params_names)
 
#phi_i: N_par x J, wartości fizyczne (tj. te które używamy w symulacji a nie te z przestrzeni logarytmicznej) parametrów zespołu i
#G_i: N_obs x J, odpowiedź modelu dla tego zespołu
#rmse_lib, loss_lib: wartości z get_error_metrics(ekp); gdy ich nie ma (nothing), liczone funkcjami pomocniczymi

function record_state!(ekp, i, phi_i, G_i; rmse_lib=nothing, loss_lib=nothing)
    G_mean = vec(mean(G_i, dims=2))
    push!(obs_means, G_mean)
    push!(obs_stds, vec(std(G_i, dims=2)))
 
    member_rmses = member_rmse(ekp, G_i)                                  #RMSE każdego członka ensemble'a, see `member_rmse`
    rmse_mean = isnothing(rmse_lib) ? mean(member_rmses) : rmse_lib       #`avg_rmse` z biblioteki, gdy dostępne
    loss_meanG = isnothing(loss_lib) ? loss_at_mean(ekp, G_i) : loss_lib  #`loss z biblioteki, gdy dostępne
 
    #kontrola zgodności z funkcjami z biblioteki; docelowo zniknie z kodu
    if !isnothing(rmse_lib) && !isapprox(mean(member_rmses), rmse_lib; rtol=1e-6)
        @warn "Iteracja $i: avg_rmse z biblioteki ($rmse_lib) NIE JEST RÓWNE ze średnią z member_rmse ($(mean(member_rmses)))"
    end
    if !isnothing(loss_lib) && !isapprox(loss_at_mean(ekp, G_i), loss_lib; rtol=1e-6)
        @warn "Iteracja $i: loss z biblioteki ($loss_lib) NIE JEST RÓWNA z loss_at_mean ($(loss_at_mean(ekp, G_i)))"
    end
 
    u_mean = vec(mean(phi_i, dims=2))
    u_std  = vec(std(phi_i, dims=2))
 
    for j in 1:size(phi_i, 2)
        row = (Iteracja = i, Ensemble_ID = j, RMSE = member_rmses[j])
        params_tuple = NamedTuple{param_syms}(Tuple(phi_i[:, j]))
        push!(df_ensemble_rows, merge(row, params_tuple))
    end
 
    sum_row = (Iteracja = i, RMSE_mean = rmse_mean, RMSE_std = std(member_rmses), Loss = loss_meanG)
    mean_tuple = NamedTuple{mean_syms}(Tuple(u_mean))
    std_tuple  = NamedTuple{std_syms}(Tuple(u_std))
    push!(df_summary_rows, merge(sum_row, mean_tuple, std_tuple))
end

#------------------------------------------------------------
#                   PĘTLA ITERACYJNA EKI
#------------------------------------------------------------
#UWAGA: do modelu idą wartości FIZYCZNE, musimy korzystać więc z funckji `get_ϕ_final`; jest to istotne dla docelowego modelu - używając `bounded_below(0.0)`
#możemy wprowadzić warunek, że wartości parametrów < 0.0 są niefizyczne więc ich nie chcemy

#Numeracja: `Iteracja = k` w CSV to zespół po (k-1) aktualizacjach razem z odpowiedzią modelu G dla niego; ostatnia iteracja to zespół końcowy.
#UWAGA: pętla `for` na poziomie globalnym tworzy własny zasięg, dlatego przypisania do `G_final` i `finished` wewnątrz niej muszą mieć `global`;
#bez tego powstają zmienne lokalne, a globalne zostają `nothing`/`false` (=> zbędna ponowna ewaluacja, zduplikowany ostatni wiersz w CSV 
#i fałszywe ostrzeżenie o N_ITER)
G_final  = nothing
finished = false
for i in 1:N_ITER
    phi_i = get_ϕ_final(prior, ekp; return_array=true)
    G_i = G_ensemble(phi_i)
 
    terminated = update_ensemble!(ekp, G_i)

    if isnothing(terminated)
        m = get_error_metrics(ekp)      #pobiera gotowe funkcje metryk z biblioteki
        record_state!(ekp, i, phi_i, G_i; rmse_lib = m["avg_rmse"][end], loss_lib = m["loss"][end])
    else
        #DataMisfitController zakończył działanie pętli; zespół nie został zaktualizowany, więc phi_i i G_i są już zespołem końcowym
        #metryki dla nich trzeba policzyć przy pomocy funkcji pomocniczych bo funkcje z biblioteki tego nie wyznaczają
        global finished = true
        global G_final = G_i
        record_state!(ekp, i, phi_i, G_i)
        @info "DataMisfitController zakończył na iteracji $i, czas algorytmu = $(get_algorithm_time(ekp)[end])"      
        #get_algorithm_time zwraca wektor (skumulowany czas po każdej aktualizacji)
        break
    end
end

#jeżeli pętla skończyła się na N_ITER ==> kryterium DataMisfitController nie zostało osiągnięte, więc wykonała się ustalona maksymalna liczba iteracji;
#w tej sytuacji zespół końcowy musi zostać jeszcze zewaluowany
if isnothing(G_final)
    phi_last = get_ϕ_final(prior, ekp; return_array=true)
    G_final = G_ensemble(phi_last)
    record_state!(ekp, length(obs_means) + 1, phi_last, G_final)
end
finished || @warn "Pętla skończyła się na N_ITER, a czas algorytmu < 1: kryterium DataMisfitController nie zostało osiągnięte"


#------------------------------------------------------------
#                       WYNIK KOŃCOWY
#------------------------------------------------------------
#końcowa wartość phi to średnia arytmetyczna z zespołu końcowego w przestrzeni FIZYCZNEJ (`get_ϕ_final` zwraca już wartości fizyczne);
#NIE jest to to samo co `get_ϕ_mean_final(prior, ekp)`, które liczy średnią w przestrzeni logarytmicznej (u) i dopiero ją transformuje exp(mean(u)) (średnia geometryczna)

phi_end = get_ϕ_final(prior, ekp; return_array=true)
theta_hat = vec(mean(phi_end, dims=2))
for (nm, v) in zip(est_params_names, theta_hat)
    println(rpad(nm, 20), round(v, digits=4))
end

#TO KOŃCOWO ZNIKNIE ZE SKRYPTU ==> bo wartości nominalne znamy tylko dla toy modelu
println("błąd względem nominalnych [%]:", round.(100 .* (theta_hat .- X_NOMINAL) ./ X_NOMINAL, digits=1))
 
 
#------------------------------------------------------------
#                  ZAPIS DO PLIKÓW CSV
#------------------------------------------------------------
df_ensemble = DataFrame(df_ensemble_rows)
df_summary  = DataFrame(df_summary_rows)
 
CSV.write(joinpath(dir_name, "ensemble_data.csv"), df_ensemble)
CSV.write(joinpath(dir_name, "summary_metrics.csv"), df_summary)


#------------------------------------------------------------
#                        WYKRESY
#------------------------------------------------------------
p1 = plot(df_summary.Iteracja, df_summary.RMSE_mean, label="avg_rmse", xlabel="Iteracja", ylabel="Błąd", lw=2, title="Wartość RMSE w zależności od iteracji")
p2 = plot(df_summary.Iteracja, df_summary.Loss, label="loss", xlabel="Iteracja", ylabel="Funkcja straty", lw=2, title="Loss function w zależności od iteracji")
 
p3 = plot(xlabel="Iteracja", ylabel="Wartość parametru", title="Ewolucja parametrów", legend=:outerright)
for param_name in est_params_names
    param_mean = df_summary[:, "$(param_name)_mean"]
    param_std = df_summary[:, "$(param_name)_std"]
    plot!(p3, df_summary.Iteracja, param_mean, ribbon=param_std, label=param_name, lw=2)
end
hline!(p3, X_NOMINAL, ls=:dash, color=:gray, label=false)       #plotuje wartości nominalne parametrów ==> DOCELOWO ZNIKA ZE SKRYPTU
 
final_mean = obs_means[end]
final_std = obs_stds[end]
p4 = scatter(1:length(y_obs), y_obs, label="Dane pomiarowe", color=:red, marker=:x)
plot!(p4, 1:length(y_obs), final_mean, ribbon=final_std, label="Predykcja", color=:blue, lw=2, title="Dopasowanie do obserwacji")
plot!(p4, 1:length(y_nominal), y_nominal, label="Nominalne", color=:green, ls=:dash)
 
fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 700))
savefig(fig, joinpath(dir_name, "wykresy.png"))
 
println("Obliczenia zakończone. Dane zapisano w folderze: $dir_name")



#------------------------------------------------------------
#                  KONTROLNE WYDRUKI
#------------------------------------------------------------
rmse_absolute = [sqrt(mean((G_final[:, j] .- y_obs).^2)) for j in 1:size(G_final, 2)]   #w jednostkach obserwabli
rmse_normalized = member_rmse(ekp, G_final)                                             #w jednostkach σ (jak avg_rmse)
 
println("Średni RMSE bezwzględny: ", mean(rmse_absolute))
println("Średni RMSE znormalizowany: ", mean(rmse_normalized))
println("avg_rmse z biblioteki (ostatnia aktualizacja): ", get_error_metrics(ekp)["avg_rmse"][end])