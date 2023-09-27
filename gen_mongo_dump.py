import json
import logging
import os
import re
import sys
import unicodedata
from string import punctuation

# from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from pymongo import MongoClient
from tqdm import tqdm

# stop = stopwords.words("russian")
punctuation += "«»"


def get_clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
    return text


def sentence_parse(doc):
    sentences = []
    sentence_tags = []
    sentiments = []
    for sent in doc["Parse"]:
        text = get_clean_text(sent["Text"])
        tagged_tokens = {}
        sentiment = sent.get("sentiment", "Neutral")
        entities = sent.get("Entities", [])
        if entities:
            for entity in entities:
                name = get_clean_text(entity["EntityName"])
                search = re.search(re.escape(name), text, flags=re.I)
                span = search.span()

                pre_tokens = wordpunct_tokenize(text[: span[0]])
                entity_tokens = wordpunct_tokenize(text[span[0] : span[1]])
                post_tokens = wordpunct_tokenize(text[span[1] :])

                pre_tags = ["O"] * len(pre_tokens)
                post_tags = ["O"] * len(post_tokens)
                entity_tags = ["B-ORG"] + ["I-ORG"] * (len(entity_tokens) - 1)
                tokens = pre_tokens + entity_tokens + post_tokens
                tags = pre_tags + entity_tags + post_tags
                assert len(tokens) == len(tags)
                for i, (token, tag) in enumerate(zip(tokens, tags)):
                    if i not in tagged_tokens or tagged_tokens[i][1] == "O":
                        tagged_tokens[i] = [token, tag]

        if tagged_tokens != {}:
            sentences.append([tagged_tokens[i][0] for i in tagged_tokens])
            sentence_tags.append([tagged_tokens[i][1] for i in tagged_tokens])
            sentiments.append(sentiment)

    sentences, sentence_tags, sentiments = filter_sentences(
        sentences, sentence_tags, sentiments
    )
    assert len(sentences) == len(sentence_tags) == len(sentiments)
    error = len(sentences) == 0
    return sentences, sentence_tags, sentiments, error


def filter_sentences(sentences, sentence_tags, sentiments):
    tmp = [set(s) for s in sentence_tags]
    if len(tmp) == 0:
        return [], [], []
    if tmp.count({"O"}) / len(tmp) >= 0.5:
        new_tok = []
        new_tags = []
        new_sent = []
        count_t = count_o = 0
        for to, ta, se in zip(sentences, sentence_tags, sentiments):
            if count_o < count_t:
                new_tok.append(to)
                new_tags.append(ta)
                new_sent.append(se)
                if set(ta) == {"O"}:
                    count_o += 1
                else:
                    count_t += 1
            elif set(ta) != {"O"}:
                new_tok.append(to)
                new_tags.append(ta)
                new_sent.append(se)
                count_t += 1
        return new_tok, new_tags, new_sent
    return sentences, sentence_tags, sentiments


def do_the_thing(db, coll):
    errors = []
    result = []
    collection = MongoClient("mongodb://127.0.0.1:27017/")[db][coll]
    count = collection.count_documents(
        {"Parse": {"$exists": True, "$ne": []}, "isRepost": None}
    )
    for doc in tqdm(
        collection.find(
            {"Parse": {"$exists": True, "$ne": []}, "isRepost": None},
            {"Text": 1, "Parse": 1},
        ),
        total=count,
    ):
        tokens, tags, sent, err = sentence_parse(doc)
        if err:
            errors.append(doc["_id"])
        else:
            for text, text_tags, text_sent in zip(tokens, tags, sent):
                assert len(text) == len(text_tags)
                result.append(
                    {
                        # "_id": doc["_id"],
                        "text": text,
                        "labels": text_tags,
                        "sentiment_label": text_sent,
                    }
                )
    fname = f"dump/{collection.name}_Parse.jsonl"
    with open(fname, "w", encoding="utf8") as f:
        for line in result:
            # line.pop("_id")
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return fname


if __name__ == "__main__":
    os.makedirs("dump", exist_ok=True)

    fname = do_the_thing()
    logging.info(f"Written {fname}")
