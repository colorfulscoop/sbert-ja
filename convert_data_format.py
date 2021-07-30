import sys
import json


def strip(xs):
    return xs.replace(" ", "")


def main():
    for line in sys.stdin:
        label, premise, hypothesis = line.strip("\n").split("\t")
        json_out = json.dumps({"label": label, "premise": premise, "hypothesis": hypothesis}, ensure_ascii=False)
        print(json_out)


if __name__ == "__main__":
    main()
