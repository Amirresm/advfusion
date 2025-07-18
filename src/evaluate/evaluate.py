import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def calc_bleu(predictions, references):
    results = bleu.compute(predictions=predictions, references=references, smooth=True)
    return results


def calc_rouge(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)
    return results


def calc_all_metrics(predictions, references):
    bleu_score = calc_bleu(predictions, references)
    rouge_score = calc_rouge(predictions, references)

    results = {
        **{f"BLEU_{k}": v for k, v in bleu_score.items()},
        **{f"ROUGE_{k}": v for k, v in rouge_score.items()},
    }
    return results


if __name__ == "__main__":
    ref = ["Print an error log message."]
    pred = ["print a message and exit with the given code"]

    results = calc_all_metrics(pred, ref)
    print(results)
