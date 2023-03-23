import os, json, glob
from utils.labelstudio import text2labelstudio_tokenized
from utils.nlp_tasks import run_spacy
import csv
import spacy


def biographynet_labelstudio_files():
    labelstudio_path = "outputs/biographynet/test/labelstudio"
    if not os.path.exists(labelstudio_path): os.makedirs(labelstudio_path, exist_ok=True)
    with open(f"{labelstudio_path}/token2spans.json", "w") as fspans:
        with open("outputs/biographynet/biographynet_test.jsonl") as f:
            for i, line in enumerate(f):
                bio = json.loads(line)
                metadata = {}
                metadata['source'] = bio['source']
                metadata['token_objects'] = bio['text_token_objects']
                task, token2spans = text2labelstudio_tokenized(bio['id_composed'], bio['text_clean'], metadata, fake_paragraph_size=5) 
                json.dump(task, open(f"{labelstudio_path}/{bio['id_composed']}.ls.json", "w"), indent=2)
                span_obj = {'text_id': bio['id_composed'], 'token2spans': token2spans}
                fspans.write(f"{json.dumps(span_obj)}\n")
                if i > 10: break


def wikipedia_labelstudio_files():
    articles_parent_path = "/Users/daza/Repos/my-vu-experiments/wiki_go_data/"
    labelstudio_path = "outputs/wikipedia_go/"
    if not os.path.exists(labelstudio_path): os.makedirs(labelstudio_path, exist_ok=True)
    if not os.path.exists(f"{labelstudio_path}/token2spans"): os.makedirs(f"{labelstudio_path}/token2spans", exist_ok=True)

    spacy_nlp = spacy.load("en_core_web_lg", disable="ner")

    def get_sentences_from_csv(path: str):
        doc_sents = []
        with open(path) as f:
            for s in csv.DictReader(f):
                if len(s) > 1:
                    doc_sents.append(s['sentslist'].strip("\n"))
        return doc_sents


    men_bios = list(glob.glob(f"{articles_parent_path}/men/*.csv"))
    women_bios = list(glob.glob(f"{articles_parent_path}/women/*.csv"))
    all_bios = men_bios + women_bios
    for filepath in all_bios:
        sentences = get_sentences_from_csv(filepath)
        full_text = " ".join(sentences)
        basename = os.path.basename(filepath).strip(".csv").split("df4")[0].strip("_")
        metadata = {}
        metadata['token_objects'] = run_spacy(full_text, spacy_nlp)['token_objs']
        task, token2spans = text2labelstudio_tokenized(basename.lower(), full_text, metadata, fake_paragraph_size=10) 
        json.dump(task, open(f"{labelstudio_path}/{basename}.ls.json", "w", encoding='utf-8'), indent=2)
        json.dump(token2spans, open(f"{labelstudio_path}/token2spans/{basename}.token2spans.json", "w", encoding='utf-8'), indent=2)


if __name__ == '__main__':
    # biographynet_labelstudio_files()
    wikipedia_labelstudio_files()