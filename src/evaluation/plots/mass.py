# from src.utils.inference.event_metrics import get_response_for_event_energy
from src.utils.inference.event_metrics import calculate_event_energy_resolution, calculate_event_mass_resolution
from src.utils.inference.inference_metrics import get_sigma_gaussian
import numpy as np
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.lines import Line2D
import os 

def get_response_for_event_energy(matched_pandora, matched_, perfect_pid=False, mass_zero=False, ML_pid=False, pandora=False):
    if pandora:
        (
            mean_p,
            variance_om_p,
            distr_p,
            x_p,
            _,
            _,
            _,
            mass_over_true_pandora,
        ) = calculate_event_energy_resolution(matched_pandora, True, False)
    (
        mean,
        variance_om,
        distr,
        x,
        mean_baseline,
        variance_om_baseline,
        _,
        mass_over_true_model,
    ) = calculate_event_energy_resolution(matched_, False, False)
    if pandora:
        dic_pandora = calculate_event_mass_resolution(matched_pandora, True, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, fake=False)
    
    dic_model = calculate_event_mass_resolution(matched_, False, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, fake=False)
    # mean_mass_perfect_PID, var_mass_perfect_PID, distr_mass_perfect_PID, mass_true_perfect_PID, _, _, E_over_true_perfect_PID, E_over_true_reco_perfect_PID = calculate_event_mass_resolution(matched_, False, perfect_pid=True, mass_zero=False, ML_pid=False,  fake=True)
    # matched_.calibrated_E = matched_.pred_showers_E
    df_copy = matched_.copy(deep=True)
    df_copy.true_showers_E = matched_.reco_showers_E
    # matched_pandora.pandora_calibrated_pfo = matched_pandora.pred_showers_E
    # matched_.pred_pos_matched = matched_.true_pos
    dic_perfect_E = calculate_event_mass_resolution(df_copy, False, perfect_pid=False, mass_zero=False, ML_pid=True, fake=False)
    if pandora:
        dic_perfect_E_pandora = calculate_event_mass_resolution(matched_pandora, True, perfect_pid=False, mass_zero=False, ML_pid=True, fake=False)
    (
        mean_energy_over_true,
        var_energy_over_true,
        _,
        _,
    ) = get_sigma_gaussian(dic_model["E_over_true"], np.linspace(0, 2, 400), epsilon=0.005)
    mean_predtotrue = np.median(dic_model["E_over_true"])
    p16 = np.percentile(dic_model["E_over_true"], 16)
    p84 = np.percentile(dic_model["E_over_true"], 84)
    var_predtotrue = p84 - p16

    mean_mass= np.median(dic_model["mass_over_true_p"] )
    p16 = np.percentile(dic_model["mass_over_true_p"] , 16)
    p84 = np.percentile(dic_model["mass_over_true_p"] , 84)
    var_mass= p84 - p16
    if pandora:
        (
            mean_energy_over_true_pandora,
            var_energy_over_true_pandora,
            _,
            _,
        ) = get_sigma_gaussian(dic_pandora["E_over_true"], np.linspace(0, 2, 400), epsilon=0.005)
        mean_predtotruep = np.median(dic_pandora["E_over_true"])
        p16 = np.percentile(dic_pandora["E_over_true"], 16)
        p84 = np.percentile(dic_pandora["E_over_true"], 84)
        var_predtotruep = p84 - p16
        mean_massp= np.median( dic_pandora["mass_over_true_p"]  )
        p16 = np.percentile( dic_pandora["mass_over_true_p"]  , 16)
        p84 = np.percentile( dic_pandora["mass_over_true_p"]  , 84)
        var_massp= p84 - p16
    dic = {}
    if pandora:
        dic["mean_p"] = mean_p
        dic["variance_om_p"] = variance_om_p
        dic["energy_resolutions_p"] = x_p
        dic["distributions_pandora"] = distr_p
        
        dic["mass_over_true_model_perfect_E_pandora"] = dic_perfect_E_pandora["mass_over_true_p"]
        dic["mass_pandora"] = dic_pandora["mass_pred_p"] 
        dic["mass_pandora_mean"] = dic_pandora["mass_pred_p_mean"]
        dic["mass_pandora_var"] = dic_pandora["mass_pred_p_var"]
        dic["mass_model_mean16p"] = mean_massp
        dic["mass_model_var16p"] = var_massp
        dic["mass_true"] =  dic_pandora["mass_true"]
        dic["mass_true_mean"] =  dic_pandora["mass_true_mean"]
        dic["mass_true_var"] =  dic_pandora["mass_true_var"]
        dic["mass_over_true_model_perfect_E"] = dic_perfect_E["mass_over_true_p"]
        dic["mass_over_true_pandora"] = dic_pandora["mass_over_true_p"]
        dic["var_mass_pandora"] = dic_pandora["var_mass"]
        dic["mean_mass_pandora"] = dic_pandora["mean_mass"]
        dic["energy_over_true_pandora"] = dic_pandora["E_over_true"]
        dic["energy_over_true_reco_pandora"] = dic_pandora["E_over_true_reco"]
        dic["var_energy_over_true_pandora"] = var_energy_over_true_pandora
        dic["mean_energy_over_true_pandora"] = mean_energy_over_true_pandora
        dic["var_energy_over_true_pandora16"] = var_predtotruep
        dic["mean_energy_over_true_pandora16"] = mean_predtotruep

    dic["mass_over_true_model"] = dic_model["mass_over_true_p"]
    dic["variance_om"] = variance_om
    dic["mean"] = mean
    dic["energy_resolutions"] = x
    dic["mean_baseline"] = mean_baseline
    dic["variance_om_baseline"] = variance_om_baseline
    dic["distributions_model"] = distr
    dic["mass_model"] = dic_model["mass_pred_p"] 
    dic["mass_model_mean"] = dic_model["mass_pred_p_mean"]
    dic["mass_model_var"] = dic_model["mass_pred_p_var"]
    dic["mass_model_mean16"] = mean_mass
    dic["mass_model_var16"] = var_mass
    dic["mean_mass_model"] = dic_model["mean_mass"]
    dic["var_mass_model"] =  dic_model["var_mass"]
    dic["energy_over_true"] = dic_model["E_over_true"]
    dic["mean_energy_over_true"] = mean_energy_over_true
    dic["var_energy_over_true"] = var_energy_over_true
    dic["mean_energy_over_true16"] = mean_predtotrue
    dic["var_energy_over_true16"] = var_predtotrue
    dic["energy_over_true_reco"] = dic_model["E_over_true_reco"]
    
    return dic

def plot_mass(sd_hgb, sd_pandora, PATH_store_summary_plots):
    colors = {"ML": "red", "ML GTC": "green"}



    PATH_store = PATH_store_summary_plots
    label_ML="HitPF"
    label_ML_GTC="ML GTC",
    color_ML_GTC="green"
    filename="mass_resolution_comp_corrected_E_mass.pdf"
    perfect_pid = False
    mass_zero = False
    ML_pid = True
    matched_all = {label_ML: sd_hgb, label_ML_GTC: sd_hgb}
    matched_pandora = sd_pandora
    event_res_dic = {} 
    for key in matched_all:
            matched_ = matched_all[key]
            event_res_dic[key] = get_response_for_event_energy(
                    matched_pandora, matched_, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, pandora=True
                )


    dic=  event_res_dic[label_ML]



    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    plt.rcParams['text.usetex'] = False
    size_font = 17
    matplotlib.rcParams.update({'font.size': size_font})
    fig, ax = plt.subplots(1, 2,figsize=(16*2, 5*2))
    ax[1].set_xlabel(r"$M_{\mathrm{reco}}/M_{\mathrm{true}}$")
    bins = np.linspace(0, 2, 200)
    values = event_res_dic[label_ML]["mass_over_true_model"]
    weights = np.ones_like(values) / len(values)

    ax[1].hist(
        values,
        bins=bins,
        histtype="step",
        color="#E36414",
        weights=weights,
        linewidth=2
    )
    p16p = np.percentile(event_res_dic[label_ML]["energy_over_true_pandora"], 16)
    p84p = np.percentile(event_res_dic[label_ML]["energy_over_true_pandora"], 84)
    p16 = np.percentile(event_res_dic[label_ML]["energy_over_true"], 16)
    p84 = np.percentile(event_res_dic[label_ML]["energy_over_true"], 84)

    std_pandora = (p84p-p16p)/(2*np.median(event_res_dic[label_ML]["energy_over_true_pandora"]))
    std_model = (p84-p16)/(2*np.median(event_res_dic[label_ML]["energy_over_true"]))
    # ax[0].axvline(x=p16p, color='b', linestyle='-',)
    # ax[0].axvline(x=p84p, color='b', linestyle='-',)
    # ax[0].axvline(x=p16, color='orange', linestyle='-',)
    # ax[0].axvline(x=p84, color='orange', linestyle='-',)
    values = event_res_dic[label_ML]["mass_over_true_pandora"]
    weights = np.ones_like(values) / len(values)

    ax[1].hist(
        values,
        bins=bins,
        histtype="step",
        color="#0F4C5C",
        weights=weights,
        linewidth=2
    )

    ax[1].grid()


    def rms90(values):
        """RMS of the central 90% of the distribution."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        p5, p95 = np.percentile(arr, 5), np.percentile(arr, 95)
        core = arr[(arr >= p5) & (arr <= p95)]
        return np.sqrt(np.mean((core - np.mean(core))**2))

    var_m_model_1 = round(std_model*100, 1)
    var_m_pandora_1 = round(std_pandora*100, 1)

    var_m_model = round((dic["mass_model_var16"]/(2*dic["mass_model_mean16"])*100), 1)
    var_m_pandora = round((dic["mass_model_var16p"]/(2*dic["mass_model_mean16p"])*100), 1)

    rms90_mass_model   = round(rms90(event_res_dic[label_ML]["mass_over_true_model"]) * 100, 1)
    rms90_mass_pandora = round(rms90(event_res_dic[label_ML]["mass_over_true_pandora"]) * 100, 1)
    rms90_E_model      = round(rms90(event_res_dic[label_ML]["energy_over_true"]) * 100, 1)
    rms90_E_pandora    = round(rms90(event_res_dic[label_ML]["energy_over_true_pandora"]) * 100, 1)

    # sigma_e_over_true_pandora = round(event_res_dic[label_ML]["var_energy_over_true_pandora"]/event_res_dic[label_ML]["mean_energy_over_true_pandora"], 3)
    # sigma_e_over_true = round(event_res_dic[label_ML]["var_energy_over_true"]/event_res_dic[label_ML]["mean_energy_over_true"], 3)
    # mean_e_over_true_gtc, sigma_e_over_true_gtc = round(event_res_dic[label_ML_GTC]["mean_energy_over_true"], 3), round(
    #     event_res_dic[label_ML_GTC]["var_energy_over_true"], 3)
    values = event_res_dic[label_ML]["energy_over_true"]
    weights = np.ones_like(values) / len(values)

    ax[0].hist(
        values,
        bins=bins,
        histtype="step",
        color="#E36414",
        weights=weights,
        linewidth=2
    )
    
    values = event_res_dic[label_ML]["energy_over_true_pandora"]
    weights = np.ones_like(values) / len(values)

    ax[0].hist(
        values,
        bins=bins,
        histtype="step",
        color="#0F4C5C",
        weights=weights,
        linewidth=2
    )


    ax[0].grid(1)
    ax[0].set_xlabel(r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$")
    # ax[1].legend(loc='upper left')

    # from matplotlib.lines import Line2D
    custom_line1 = Line2D([0], [0], color="#E36414",label="HitPF "+"\n"+"$\sigma/\mu$={}$\%$".format(var_m_model_1)+"\n"+"RMS90={}$\%$".format(rms90_E_model))
    # custom_line_gt = Line2D([0], [0], color="green",label="ML GT "+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML GTC"]["var_mass_model"]), 2),
    #     ))

    custom_line_pandora = Line2D([0], [0], color="#0F4C5C",label="Pandora "+"\n"+"$\sigma/\mu$={}$\%$".format(var_m_pandora_1)+"\n"+"RMS90={}$\%$".format(rms90_E_pandora),)

    custom_line2 = Line2D([0], [0], color="#E36414",label="HitPF "+"\n"+"$\sigma/\mu$={}$\%$".format(var_m_model)+"\n"+"RMS90={}$\%$".format(rms90_mass_model), )

    custom_line_pandora2 = Line2D([0], [0], color="#0F4C5C",label="Pandora "+"\n"+"$\sigma/\mu$={}$\%$".format(var_m_pandora)+"\n"+"RMS90={}$\%$".format(rms90_mass_pandora), )


    title= r"$Z\rightarrow q\bar q (q=u,d,s), \quad \sqrt{s}=91$ GeV"
    leg = ax[0].legend(handles=[custom_line1, custom_line_pandora],loc='upper left',title_fontsize=size_font, title=title)
    leg._legend_box.align = "left"
    leg1 = ax[1].legend(handles=[custom_line2, custom_line_pandora2],loc='upper left',title_fontsize=size_font,title=title)
    leg1._legend_box.align = "left"
    ax[0].set_xlim([0.5, 1.5])
    ax[1].set_xlim([0.5, 1.5])
    ax[0].set_ylim([0, 0.19])
    ax[1].set_ylim([0, 0.19])
    ax[0].set_ylabel("Event Fraction")
    ax[1].set_ylabel("Event Fraction")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")

    plt.rcParams['font.size'] = size_font
    plt.rcParams['axes.labelsize'] = size_font
    plt.rcParams['xtick.labelsize'] = size_font
    plt.rcParams['ytick.labelsize'] = size_font
    plt.rcParams['legend.fontsize'] = size_font

    fig.tight_layout()
    fig.savefig(os.path.join(PATH_store, "mass_plot.pdf"), bbox_inches="tight")
