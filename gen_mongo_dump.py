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


def get_clean_text(text: str):
    """
    Removes weird ivisible symbols and makes sure
    that one visible symbol corresponds to one character

    Args:
        text (str): input text

    Returns:
        str: clean text
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
    return text


def text_parse(doc: dict):
    """
    Given a text split into sentences,
    returns for each sentence a list of tokens,
    corresponding list of BIO-tags and its sentiment.

    Args:
        doc (dict): mongoDB document with a "Parse" field,
        containing list of sentences in text with NE and SA results

    Returns:
        tuple(list, list, list, bool): list of tokens, list of tags, list of sentiments,
        whether something went wrong during parse
    """
    sentences = []
    sentence_tags = []
    sentiments = []
    for sent in doc["Parse"]:
        tagged_tokens, sentiment = sentence_parse(sent)

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


def sentence_parse(sent: dict):
    """Generates BIO-markup for sentence of the text, based on extracted NE's.

    Args:
        sent (dict): document field with NE+SA results.
        Has fields:
        * "Text" for sentence text,
        * "Sentiment" for sentiment,
        * "Entities" for extracted NE's with "EntityName" as corresponding text fragment

    Returns:
        tuple(dict, str): BIO-tags for tokens of the sentence, sentence sentiment
    """
    text = get_clean_text(sent["Text"])
    tagged_tokens = {}
    # for the sake of experiment we assume
    # that sentiment of the sentence relates to the entity as well
    sentiment = sent.get("sentiment", "Neutral")
    entities = sent.get("Entities", [])
    if entities:
        for entity in entities:
            name = get_clean_text(entity["EntityName"])
            # we have only the name, but not coordinates in text
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
                # there can be several entities, so we need to update markup every time
                if i not in tagged_tokens or tagged_tokens[i][1] == "O":
                    tagged_tokens[i] = [token, tag]
    return tagged_tokens, sentiment


def filter_sentences(sentences: list, sentence_tags: list, sentiments: list):
    """
    If there is more than 1/2 of sentences without NE's in set,
    filters out some of them to create a more balanced set

    Args:
        sentences (list)
        sentence_tags (list)
        sentiments (list)

    Returns:
        Filtered lists of tokens/tags/sentiments
    """
    tmp = [set(stags) for stags in sentence_tags]
    if len(tmp) == 0:
        return [], [], []
    if tmp.count({"O"}) / len(tmp) > 0.5:
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
    """
    Parse all documents in the collection and generate a dump in the pre-defined folder
    """
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
        tokens, tags, sent, err = text_parse(doc)
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
    db, coll = sys.argv[1:3]
    fname = do_the_thing(db, coll)
    logging.info(f"Written {fname}")
