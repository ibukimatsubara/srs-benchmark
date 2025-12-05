import pathlib
import json
import numpy as np
import scipy  # type: ignore
import math
import argparse


def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    rounded_value = round(value, decimals)
    if n_lead_zeros_CI > num_lead_zeros(rounded_CI):
        return str(f"{round(value, decimals - 1):.{decimals - 1}f}"), str(
            f"{round(CI, decimals - 1):.{decimals - 1}f}"
        )
    else:
        return str(f"{rounded_value:.{decimals}f}"), str(f"{rounded_CI:.{decimals}f}")


# tests to ensure that sigdigs is working as intended
value = 0.084011111
CI = 0.0010011111
assert sigdig(value, CI) == ("0.0840", "0.0010")

value2 = 0.083999999
CI2 = 0.0009999999
assert sigdig(value2, CI2) == ("0.0840", "0.0010")


def confidence_interval(values, sizes):
    identifiers = [i for i in range(len(values))]
    dict_x_w = {
        identifier: (value, weight)
        for identifier, (value, weight) in enumerate(zip(values, sizes))
    }

    def weighted_mean(z, axis):
        # creating an array of weights, by mapping z to dict_x_w
        data = np.vectorize(dict_x_w.get)(z)
        return np.average(data[0], weights=data[1], axis=axis)

    CI_99_bootstrap = scipy.stats.bootstrap(
        (identifiers,),
        statistic=weighted_mean,
        confidence_level=0.99,
        axis=0,
        method="BCa",
    )
    low = list(CI_99_bootstrap.confidence_interval)[0]
    high = list(CI_99_bootstrap.confidence_interval)[1]
    return (high - low) / 2


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    weights = np.float64(weights)  # force 64-bit precision to avoid errors sometimes
    average = np.average(values, weights=weights)
    # Bevington, P. R., Data Reduction and Error Analysis for the Physical Sciences, 336 pp., McGraw-Hill, 1969
    # https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    n_eff = np.square(np.sum(weights)) / np.sum(np.square(weights))
    variance = np.average((values - average) ** 2, weights=weights) * (
        n_eff / (n_eff - 1)
    )
    return (average, np.sqrt(variance))


if __name__ == "__main__":
    dev_mode_name = "FSRS-6-dev"
    dev_file = pathlib.Path(f"./result/{dev_mode_name}.jsonl")
    if dev_file.exists():
        with open(dev_file, "r") as f:
            common_set = set([json.loads(x)["user"] for x in f.readlines()])
    else:
        common_set = set()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--secs", action="store_true")
    parser.add_argument("--no-ci", action="store_true", help="Skip confidence interval calculation (use std instead)")
    args = parser.parse_args()

    # IL = interval lengths

    # FIL = fractional (aka non-integer) interval lengths

    # G = grades (Again/Hard/Good/Easy)

    # SR = same-day (or short-term) reviews

    # AT = answer time (duration of the review)

    base_models = [
        ("FSRS-6", 21, "IL, G, SR"),
        ("FSRS-6-cefr", 27, "IL, G, SR, CEFR"),
    ]
    if args.secs:
        models = [(dev_mode_name, None, None)] + [
            (f"{name}-secs", params, features.replace("IL", "FIL"))
            for name, params, features in base_models
        ]
    else:
        models = [(dev_mode_name, None, None)] + base_models
    if args.fast:
        for model, n_param, features in models:
            print(f"Model: {model}")
            m = []
            parameters = []
            sizes = []
            result_file = pathlib.Path(f"./result/{model}.jsonl")
            if not result_file.exists():
                continue
            with open(result_file, "r") as f:
                data = [json.loads(x) for x in f.readlines()]
            for result in data:
                if common_set and result["user"] not in common_set:
                    continue
                # if result["size"] > 1000:
                #     continue
                m.append(result["metrics"])
                sizes.append(result["size"])
                if "parameters" in result:
                    if isinstance(result["parameters"], list):
                        parameters.append(result["parameters"])
                    else:
                        parameters.extend(result["parameters"].values())
            if len(sizes) == 0:
                continue
            print(f"Total number of users: {len(sizes)}")
            print(f"Total number of reviews: {sum(sizes)}")
            for scale, size in (
                ("reviews", np.array(sizes)),
                ("log(reviews)", np.log(sizes)),
                ("users", np.ones_like(sizes)),
            ):
                print(f"Weighted average by {scale}:")
                for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                    metrics = np.array([item[metric] for item in m])
                    size = size[~np.isnan(metrics.astype(float))]
                    metrics = metrics[~np.isnan(metrics.astype(float))]
                    wmean, wstd = weighted_avg_and_std(metrics, size)
                    print(f"{model} {metric} (mean±std): {wmean:.4f}±{wstd:.4f}")
                print()

            # print(f"LogLoss 99%: {round(np.percentile(np.array([item['LogLoss'] for item in m]), 99), 4)}")
            # print(f"RMSE(bins) 99%: {round(np.percentile(np.array([item['RMSE(bins)'] for item in m]), 99), 4)}")
            if len(parameters) > 0:
                print(
                    f"parameters: {np.median(parameters, axis=0).round(6).tolist()}\n"
                )
                # print(f"parameters: {np.std(parameters, axis=0).round(2).tolist()}\n")

    else:
        for scale in ("users", "reviews"):
            print(f"Weighted by number of {scale}\n")
            print("| Model | #Params | LogLoss | RMSE(bins) | AUC | Input features |")
            print("| --- | --- | --- | --- | --- | --- |")
            for model, n_param, input_features in models:
                m = []
                parameters = []
                sizes = []
                result_file = pathlib.Path(f"./result/{model}.jsonl")
                if not result_file.exists():
                    continue
                with open(result_file, "r") as f:
                    data = [json.loads(x) for x in f.readlines()]
                for result in data:
                    if common_set and result["user"] not in common_set:
                        continue
                    m.append(result["metrics"])
                    sizes.append(result["size"])
                    if "parameters" in result:
                        parameters.append(result["parameters"])
                if len(sizes) == 0:
                    continue

                size = np.array(sizes) if scale == "reviews" else np.ones_like(sizes)
                result = f"| {model} | {n_param} |"
                for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                    metrics = np.array([item[metric] for item in m])
                    size = size[~np.isnan(metrics.astype(float))]
                    metrics = metrics[~np.isnan(metrics.astype(float))]
                    wmean, wstd = weighted_avg_and_std(metrics, size)
                    if args.no_ci:
                        # Use standard deviation instead of confidence interval
                        CI = wstd
                    else:
                        CI = confidence_interval(metrics, size)
                    rounded_mean, rounded_CI = sigdig(wmean, CI)
                    result += f" {rounded_mean}±{rounded_CI} |"
                print(result + f" {input_features} |")
