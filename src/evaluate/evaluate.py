import os
import evaluate

hf_metrics_path = os.path.join(os.path.dirname(__file__), "hf_metrics")
bleu = evaluate.load(os.path.join(hf_metrics_path, "bleu"))
rouge = evaluate.load(os.path.join(hf_metrics_path, "rouge"))


def calc_bleu(predictions, references):
    results = bleu.compute(
        predictions=predictions, references=references, smooth=True
    )
    return results or {}


def calc_rouge(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)
    return results or {}


def calc_all_metrics(predictions, references):
    bleu_score = calc_bleu(predictions, references)
    rouge_score = calc_rouge(predictions, references)

    results = {
        **{f"BLEU_{k}": v for k, v in bleu_score.items()},
        **{f"ROUGE_{k}": v for k, v in rouge_score.items()},
    }
    numeric_metrics = {}
    for k, v in results.items():
            try:
                if isinstance(v, list):
                    numeric_metrics[k] = [float(x) for x in v]
                else:
                    numeric_metrics[k] = float(v)
            except:
                numeric_metrics[k] = v
    return numeric_metrics


if __name__ == "__main__":
    ref = ["Print an error log message."]
    pred = ["print a message and exit with the given code"]

    results = calc_all_metrics(pred, ref)
    print(results)
