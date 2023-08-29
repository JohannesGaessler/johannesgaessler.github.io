#!/usr/bin/env python3

import os
from glob import glob
from multiprocessing import Pool

import yaml
import numpy as np
import matplotlib.pyplot as plt

QUANTIZATIONS = [
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"
]

MODEL_SIZES = dict(
    F16=13477973248,
    Q4_0=3825707872, Q4_1=4238650208, Q5_0=4651592544, Q5_1=5064534880, Q8_0=7129246560,
    Q2_K=2825841504, Q3_K_S=2948205408, Q3_K_M=3297905504, Q3_K_L=3597011808, Q4_K_S=3856640864,
    Q4_K_M=4080905056, Q5_K_S=4651592544, Q5_K_M=4783057760, Q6_K=5529095008
)
MODEL_SIZE_LIST_GIB = [MODEL_SIZES[q] / 1024 ** 3 for q in QUANTIZATIONS]


def get_yaml_dict(path):
    with open(path, "r") as f:
        props = yaml.safe_load(f)

    model_desc = props["model_desc"]
    model_desc = model_desc.replace(" - Small", "_S")
    model_desc = model_desc.replace(" - Medium", "_M")
    model_desc = model_desc.replace(" - Large", "_L")
    model_desc = model_desc.split()[-1]
    props["model_desc"] = model_desc

    props["probs"] = np.array(props["probs"])
    filter_array = props["probs"] > 0
    props["probs"] = props["probs"][filter_array]

    props["logits"] = np.array(props["logits"])[filter_array]
    return props


with Pool(16) as p:
    props = p.map(get_yaml_dict, glob("log_ppl/*.yml"))

perplexities = {p["model_desc"]: p["ppl_value"] for p in props}
logits       = {p["model_desc"]: p["logits"]    for p in props}
probs        = {p["model_desc"]: p["probs"]     for p in props}

assert len(props) == len(perplexities)

# plt.figure()
# plt.hist(probs, bins=np.linspace(0, 1, 201))
# plt.title(f"Probability distribution for perplexity on wikitext, 7b {quantization}")
# plt.xlim(0, 1)
# plt.xlabel("Token probability")
# plt.ylabel("Count")
# plt.savefig(f"plots/prob_hist_{quantization}.png", dpi=240)

# nll = -np.log(probs)

# plt.figure()
# plt.hist(probs, bins=np.linspace(0, 1, 101), density=True, weights=nll)
# plt.title(f"Perplexity contributions on wikitext, 7b {quantization}")
# plt.xlim(0, 1)
# plt.xlabel("Token probability")
# plt.ylabel("Rel. contributions to total perplexity")
# plt.savefig(f"plots/prob_hist_weighted_{quantization}.png", dpi=240)

# print(f"perplexity: {np.exp(np.mean(nll))}")
# print(f"perplexity_filtered: {np.exp(np.mean(nll[probs_F16 >= 0.05]))}")
# print(f"mean squared diff: {np.sqrt(np.mean(np.square(probs_F16 - probs)))}")

rmss_logits = []
rmss_probs  = []

for quantization in QUANTIZATIONS:
    logit_diffs    = logits[quantization]        - logits["F16"]
    prob_diffs     = probs[quantization]         - probs["F16"]

    logit_diffs_mean    = np.mean(logit_diffs)
    logit_diffs_std     = np.std(logit_diffs)
    prob_diffs_mean     = np.mean(prob_diffs)
    prob_diffs_std      = np.std(prob_diffs)

    print(f"=== {quantization} ===")
    print(f"Model size: {MODEL_SIZES[quantization]/1024**3:.2f} GiB")
    print(f"Perplexity: {perplexities[quantization]}")

    rms_logits = np.sqrt(np.mean(np.square(logit_diffs)))
    rms_probs  = np.sqrt(np.mean(np.square(prob_diffs)))

    rms_logits_err = np.std(np.square(logit_diffs)) / np.sqrt(logit_diffs.shape[0]) / (2 * rms_logits)
    rms_probs_err  = np.std(np.square(prob_diffs))  / np.sqrt(prob_diffs.shape[0])  / (2 * rms_probs)

    rmss_logits.append(rms_logits)
    rmss_probs.append(rms_probs)

    print(f"RMS logits: {rms_logits:.4e} +- {rms_logits_err:.4e}")
    print(f"Mean prob diffs: {prob_diffs_mean:.4e}")
    print(f"STD prob diffs: {prob_diffs_std:.4e}")
    print(f"RMS probs: {rms_probs:.4e} +- {rms_probs_err:.4e}")
    print()

    plt.figure()
    xlim = (logit_diffs_mean-3*logit_diffs_std, logit_diffs_mean+3*logit_diffs_std)
    plt.hist(logit_diffs, bins=100, range=xlim, weights=np.ones_like(logit_diffs)*100/logit_diffs.shape[0])
    plt.xlim(xlim)
    plt.title(f"Logit diff. distribution for 7b F16 and {quantization}")
    plt.xlabel("Logit difference")
    plt.ylabel("Percentage")
    os.makedirs("plots/logit_diff_hist", exist_ok=True)
    plt.savefig(f"plots/logit_diff_hist/logit_diff_hist_{quantization.lower()}.png", dpi=240)

    plt.figure()
    # xlim = (prob_diffs_mean-3*prob_diffs_std, prob_diffs_mean+3*prob_diffs_std)
    xlim = (-0.3, 0.3)
    plt.hist(prob_diffs, bins=100, range=xlim, weights=np.ones_like(prob_diffs)*100/prob_diffs.shape[0])
    plt.xlim(xlim)
    plt.ylim(0, 50)
    plt.title(f"Token prob. diff. distribution for 7b F16 and {quantization}")
    plt.xlabel("Token probability difference")
    plt.ylabel("Percentage")
    os.makedirs("plots/prob_diff_hist", exist_ok=True)
    plt.savefig(f"plots/prob_diff_hist/prob_diff_hist_{quantization.lower()}.png", dpi=240)

    plt.figure()
    plt.hist(logits["F16"], bins=np.linspace(-10, 30, 101), density=True, weights=np.square(logit_diffs))
    plt.xlim(-10, 30)
    plt.title(f"Rel. contribution to logit RMS between 7b F16 and {quantization}")
    plt.xlabel("Logit F16")
    plt.ylabel("Rel. contribution to logit RMS")
    os.makedirs("plots/logit_rms_contribution", exist_ok=True)
    plt.savefig(f"plots/logit_rms_contribution/logit_rms_contribution_{quantization.lower()}.png", dpi=240)

    plt.figure()
    plt.hist(probs["F16"], bins=np.linspace(0, 1, 101), density=True, weights=np.square(prob_diffs))
    plt.xlim(0, 1)
    plt.title(f"Rel. contribution to prob. RMS between 7b F16 and {quantization}")
    plt.xlabel("Token probability F16")
    plt.ylabel("Percent contribution to $\mathrm{RMS}_p$")
    os.makedirs("plots/prob_rms_contribution", exist_ok=True)
    plt.savefig(f"plots/prob_rms_contribution/prob_rms_contribution_{quantization.lower()}.png", dpi=240)

    ppl_diffs = []

    for percentage in range(0, 100):
        low  = (percentage + 0) / 100
        high = (percentage + 1) / 100
        filter_array = np.logical_and(probs["F16"] >= low, probs["F16"] < high)
        ppl_f16   = np.exp(np.mean(-np.log(probs["F16"][filter_array])))
        ppl_quant = np.exp(np.mean(-np.log(probs[quantization][filter_array])))
        ppl_diffs.append(ppl_quant - ppl_f16)

    filter_array = np.logical_and(probs["F16"] >= 1e-3, probs["F16"] < 1e-2)
    ppl_f16   = np.exp(np.mean(-np.log(probs["F16"][filter_array])))
    ppl_quant = np.exp(np.mean(-np.log(probs[quantization][filter_array])))
    lowest_bin_corrected = ppl_quant - ppl_f16
    print(f"Discarded during correction: {100*np.sum(probs['F16'] < 1e-3)/probs['F16'].shape[0]:.2f}%")
    print()

    plt.figure()
    plt.hist(np.linspace(0, 1, 100), bins=np.linspace(0, 1, 101), weights=ppl_diffs)
    plt.xlim(0, 1)
    plt.title(f"Perplexity differences between 7b F16 and {quantization}")
    plt.xlabel("Token probability F16")
    plt.ylabel("Perplexity difference")
    os.makedirs("plots/ppl_diff", exist_ok=True)
    plt.savefig(f"plots/ppl_diff/ppl_diff_{quantization.lower()}.png", dpi=240)

    ppl_diffs[0] = lowest_bin_corrected

    plt.figure()
    plt.hist(np.linspace(0, 1, 100), bins=np.linspace(0, 1, 101), weights=ppl_diffs)
    plt.xlim(0, 1)
    plt.title(f"Perplexity differences between 7b F16 and {quantization} (corrected lowest bin)")
    plt.xlabel("Token probability F16")
    plt.ylabel("Perplexity difference")
    os.makedirs("plots/ppl_diff", exist_ok=True)
    plt.savefig(f"plots/ppl_diff/ppl_diff_corrected_{quantization.lower()}.png", dpi=240)

    diff_means = []
    diff_stds  = []
    diff_rmss  = []

    for percentage in range(0, 100, 10):
        filter_array = np.logical_and(
            100*probs["F16"] >= percentage, 100*probs["F16"] < percentage + 10)
        diffs_filtered = logit_diffs[filter_array]
        diff_means.append(np.mean(diffs_filtered))
        diff_stds.append(np.std(diffs_filtered))
        diff_rmss.append(np.sqrt(np.mean(np.square(diffs_filtered))))

    plt.figure()
    plt.errorbar(np.arange(0.05, 1.0, 0.1), diff_means, yerr=diff_stds, marker=".", linestyle="None")
    plt.title(f"Logit diff. mean vs. std for 7b F16 and {quantization}")
    plt.xlabel("Token probability F16")
    plt.ylabel("Logit difference")
    os.makedirs("plots/logit_mean_std", exist_ok=True)
    plt.savefig(f"plots/logit_mean_std/logit_mean_std_{quantization.lower()}.png", dpi=240)

    # plt.figure()
    # plt.scatter(np.arange(0.05, 1.0, 0.1), diff_means)
    # plt.title(f"Mean logit diff. between 7b F16 and {quantization}")
    # plt.xlabel("Token probability F16")
    # plt.ylabel("Mean logit difference")
    # plt.savefig(f"plots/logit_mean_diff_{quantization}.png", dpi=240)

    plt.figure()
    plt.scatter(np.arange(0.05, 1.0, 0.1), diff_rmss, marker=".")
    plt.title(f"Logit RMS between 7b F16 and {quantization}")
    plt.xlabel("Token probability F16")
    plt.ylabel("Root mean square")
    os.makedirs("plots/logit_rms", exist_ok=True)
    plt.savefig(f"plots/logit_rms/logit_rms_{quantization.lower()}.png", dpi=240)

    rms_logit_history  = []
    rms_prob_history   = []
    perplexity_history = []

    for percentage in range(1, 101):
        high = percentage * logits[quantization].shape[0] // 100

        rms_logit_history.append(np.sqrt(np.mean(np.square(
            logits[quantization][:high] - logits["F16"][:high]
        ))) / rms_logits)
        rms_prob_history.append(np.sqrt(np.mean(np.square(
            probs[quantization][:high] - probs["F16"][:high]
        ))) / rms_probs)
        perplexity_history.append(np.exp(-np.mean(np.log(
            probs[quantization][:high]
        ))) / perplexities[quantization])

    plt.figure()
    plt.plot(range(1, 101), rms_logit_history, label="Logit RMS")
    plt.plot(range(1, 101), rms_prob_history, label="Token prob. RMS")
    plt.plot(range(1, 101), perplexity_history, label="Perplexity")
    plt.legend(loc="best")
    plt.xlabel("Percentage of dataset")
    plt.ylabel("Relative error vs. final result")
    os.makedirs("plots/metric_history", exist_ok=True)
    plt.savefig(f"plots/metric_history/metric_history_{quantization}.png", dpi=240)

os.makedirs("plots/combined", exist_ok=True)

plt.figure()
perplexity_list = [perplexities[q] for q in QUANTIZATIONS]
plt.scatter(MODEL_SIZE_LIST_GIB, perplexity_list, marker=".")
plt.xlabel("Model size [GiB]")
plt.ylabel("Perplexity")
for quant_name, x, y in zip(QUANTIZATIONS, MODEL_SIZE_LIST_GIB, perplexity_list):
    if quant_name == "Q5_0":
        x -= 0.08
    if quant_name in ["Q4_K_M", "Q5_K_S"]:
        y -= 0.01
    plt.text(x, y, quant_name, fontsize=6)
plt.savefig("plots/combined/combined_perplexity.png", dpi=240)

plt.figure()
plt.scatter(MODEL_SIZE_LIST_GIB, rmss_probs, marker=".")
plt.xlabel("Model size [GiB]")
plt.ylabel("Probability RMS")
for quant_name, x, y in zip(QUANTIZATIONS, MODEL_SIZE_LIST_GIB, rmss_probs):
    if quant_name == "Q5_K_S":
        y -= 0.001
    plt.text(x, y, quant_name, fontsize=6)
plt.savefig("plots/combined/combined_rms_probs.png", dpi=240)

plt.figure()
plt.scatter(MODEL_SIZE_LIST_GIB, rmss_logits, marker=".")
plt.xlabel("Model size [GiB]")
plt.ylabel("Logits RMS")
for quant_name, x, y in zip(QUANTIZATIONS, MODEL_SIZE_LIST_GIB, rmss_logits):
    if quant_name == "Q5_K_S":
        y -= 0.02
    plt.text(x, y, quant_name, fontsize=6)
plt.savefig("plots/combined/combined_rms_logits.png", dpi=240)

plt.figure()
plt.hist(probs["F16"], bins=np.linspace(0, 1, 101), density=True)
plt.title("Probability distribution for perplexity, 7b F16")
plt.xlim(0, 1)
plt.xlabel("Token probability")
plt.ylabel("Percentage")
plt.savefig("plots/prob_hist_f16.png", dpi=240)

plt.figure()
plt.hist(probs["F16"], bins=np.linspace(0, 1, 101), density=True, weights=-np.log(probs["F16"]))
plt.title("Rel. contributions to log. perplexity value, 7b F16")
plt.xlim(0, 1)
plt.xlabel("Token probability")
plt.ylabel("Percentage of total value")
plt.savefig("plots/ppl_contributions_f16.png", dpi=240)

# plt.show()
