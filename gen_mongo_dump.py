import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from string import punctuation

# from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from pymongo import MongoClient
from tqdm import tqdm

# stop = stopwords.words("russian")
punctuation += "«»"


def get_clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace(r"\n", " ")
        .replace("\n", " ")
        .replace("«", '"')
        .replace("»", '"')
        .replace("ё", "е")
        .replace("Ё", "Е")
    )
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
    text = re.sub(r"\s+", " ", re.sub(f"([{punctuation}])", r" \1 ", text)).strip()
    return text


def sentence_parse(doc):
    sentences = []
    sentence_tags = []
    sentiments = []
    for sent in doc["NewParse"]:
        text = get_clean_text(sent["Text"])
        tagged_tokens = {}
        sentiment = "Neutral"
        entities = sent.get("Entities", [])
        if entities:
            for entity in entities:
                name = get_clean_text(entity["EntityName"])
                _sentiment = entity.get("sentiment", "Neutral")
                _sentimentG = entity.get("gptEngSentiment", "Neutral")
                search = re.search(re.escape(name), text, flags=re.I)
                if search is None:
                    word = doc["Text"][
                        entity["EntityPositions"][0]["sent_offset"] : entity[
                            "EntityPositions"
                        ][0]["sent_offset"]
                        + entity["EntityPositions"][0]["sent_length"]
                    ]
                    clean_word = get_clean_text(word)
                    span = [
                        text.index(clean_word),
                        text.index(clean_word) + len(clean_word),
                    ]

                else:
                    span = search.span()

                pre_tokens = wordpunct_tokenize(text[: span[0]])
                entity_tokens = wordpunct_tokenize(text[span[0] : span[1]])
                post_tokens = wordpunct_tokenize(text[span[1] :])

                pre_tags = ["O"] * len(pre_tokens)
                post_tags = ["O"] * len(post_tokens)
                if _sentiment != "Wrong":
                    entity_tags = ["B-ORG"] + ["I-ORG"] * (len(entity_tokens) - 1)
                else:
                    entity_tags = ["I-ERR"] * (len(entity_tokens))
                _tokens = pre_tokens + entity_tokens + post_tokens
                _tags = pre_tags + entity_tags + post_tags
                assert len(_tokens) == len(_tags)
                for i, (token, tag) in enumerate(zip(_tokens, _tags)):
                    if i not in tagged_tokens or tagged_tokens[i][1] == "O":
                        tagged_tokens[i] = [token, tag]
                if _sentimentG != "Neutral":
                    sentiment = _sentimentG
                elif _sentiment != "Neutral":
                    sentiment = _sentiment

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


settings = {
    "collection": "Posts",
    "find_field": "NewParse.Entities.gptEngSentiment",
    "parse_func": sentence_parse,
}


def do_the_thing():
    errors = []
    result = []
    collection = MongoClient("mongodb://127.0.0.1:27017/")["MONGO_DB_NAME"][
        settings["collection"]
    ]
    count = collection.count_documents(
        {settings["find_field"]: {"$exists": True, "$ne": []}, "isRepost": None}
    )
    for doc in tqdm(
        collection.find(
            {settings["find_field"]: {"$exists": True, "$ne": []}, "isRepost": None},
            {"Text": 1, "NewParse": 1},
        ),
        total=count,
    ):
        tokens, tags, sent, err = settings["parse_func"](doc)
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
    fname = f"dump/{collection.name}_NewParse.jsonl"
    with open(fname, "w", encoding="utf8") as f:
        for line in result:
            # line.pop("_id")
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return fname


if __name__ == "__main__":
    os.makedirs("dump", exist_ok=True)

    fname = do_the_thing()
    logging.info(f"Written {fname}")
