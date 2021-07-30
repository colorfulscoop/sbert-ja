import sys
import json


def format_sentence(xs):
    return xs.replace(" ", "")


def main():
    for line in sys.stdin:
        label, premise, hypothesis = line.strip("\n").split("\t")
        json_out = json.dumps(
            {"label": label,
             "premise": format_sentence(premise),
             "hypothesis": format_sentence(hypothesis)},
             ensure_ascii=False
        )
        print(json_out)


if __name__ == "__main__":
    main()
