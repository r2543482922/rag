# json2txt.py
import json, itertools, unicodedata

def flatten(items):
    for item in items:
        yield unicodedata.normalize("NFKC", item["label"])
        if "children" in item:
            yield from flatten(item["children"])

data = json.load(open("diseases.json", encoding="utf-8-sig"))
with open("icd10_cn.txt", "w", encoding="utf-8") as f:
    for line in flatten(data):
        f.write(line + "\n")


