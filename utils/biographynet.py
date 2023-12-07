import os, glob
from typing import Any, Counter, Dict, List, TypeVar
from lxml import etree
import re, json
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import logging
logger = logging.getLogger(__name__)

from utils.extract_from_xml import retrieve_name, extract_person_information, extract_text
from utils.nlp_tasks import run_stanza_batch, AnnotatedToken
NLP_Processor = TypeVar("NLP_Processor")


##### ---------- MAIN CLASSES ---------- #####
@dataclass
class BiographyJSON:
    id_person: str
    version: str
    id_composed: str 
    source: str
    name: str
    partition: str
    meta_keys: List[str] # List containing the metadata fields that contain a value for each file
    had_html: bool
    original_path: str
    birth_pl: str
    birth_tm: str
    baptism_pl: str
    baptism_tm: str
    death_pl: str
    death_tm: str
    funeral_pl: str
    funeral_tm: str
    marriage_pl: str
    marriage_tm: str
    gender: str
    category: str 
    father: str
    mother: str
    partner: str
    religion: str
    educations: List[Dict[str, str]]
    faiths: List[Dict[str, str]]
    occupations: List[Dict[str, str]]
    residences: List[Dict[str, str]]
    text_clean: str
    text_original: str
    nlp_processor: str = None
    text_tokens: List[str] = None
    text_token_objects: List[AnnotatedToken] = None
    text_sentences: List[str] = None
    text_entities: List[Dict] = None
    text_timex: List[Dict] = None
    tokens_len: int  = None
    meta_len: int = None


class BiographyXML:
    def __init__(self, filename, xml_doc):
        self.tree = xml_doc.getroot()
        self.had_html = False
        self.filepath = filename
        self.partition = ""
        self.text_original = ""
        self.text_clean = ""
        try:
            self.source = self.tree.find("fileDesc/idno").text
        except:
            src_data = self.tree.findall("person/idno")
            for src in src_data:
                if src.get("type") == "source":
                    self.source = src.text
        
        self.id_bio, self.version = re.findall(r"\d{2,10}_\d{2}", filename)[0].split("_")
        self.id_composed = f"{self.id_bio}_{self.version}"

        text_found = False
        for ch in self.tree.getchildren():
            # Get Personal Data
            # print(ch.tag, ch.get('type'), ch.text)
            if ch.tag == "person":
                self.name = retrieve_name(ch)
                # Possible Keys: ['<name>', '<birth-time>', '<birth-place>', '<death-time>', ...]
                self.info = extract_person_information(ch)
            # Get Bio Raw Text
            elif ch.tag == "biography":
                text_found = True
                self.text_original = extract_text(ch)


    def show_children(self):
        for element in self.tree.iter():
            print(f"{element.tag} - {element.text}")


class DataCleaner:
    nl_special_chars = {"Á":	"A",
                "á":	"a",
                "À":	"A",
                "à":	"a",
                "Â":	"A",
                "â":	"a",
                "Ä":	"A",
                "ä":	"a",
                "É":	"E",
                "é":	"e",
                "È":	"E",
                "è":	"e",
                "Ê":	"E",
                "ê":	"e",
                "Ë":	"E",
                "ë":	"e",
                "Í":	"I",
                "í":	"i",
                "Ì":	"I",
                "ì":	"i",
                "Î":	"I",
                "î":	"i",
                "Ï":	"I",
                "ï":	"i",
                "Ĳ":	"IJ",
                "ĳ":	"ij",
                "Ó":	"O",
                "ó":	"o",
                "Ò":	"O",
                "ò":	"o",
                "Ô":	"O",
                "ô":	"o",
                "Ö":	"O",
                "ö":	"o",
                "Ú":	"U",
                "ú":	"u",
                "Ù":	"U",
                "ù":	"u",
                "Û":	"U",
                "û":	"u",
                "Ü":	"U",
                "ü":	"u",
                "Ý":	"Y",
                "ý":	"y",
                "Ÿ":	"Y",
                "ÿ":	"y",
                "«":	"\"",
                "»":	"\"",
                "ƒ":	"f",
                "ç": "c",
                "ﬁ": "fi"
                }
    punctuation_chars = {
        "—": " ",
        "–": " ",
        "“": "\"",
        "”": "\"",
        "„": "\"",
        "‘": "'",
        "’": "'",
        
    }
    
    def clean_html(self, html_text: str, html_filename: str = '') -> str:
        html_text = BeautifulSoup(html_text, 'html5lib').get_text()
        # Remove inline JavaScript/CSS:
        cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html_text.strip())
        # Then we remove html comments:
        cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", " ", cleaned)
        # Next we can remove the remaining tags:
        cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
        # Finally, we deal with whitespace
        cleaned = re.sub(r"&nbsp;", " ", cleaned)
        cleaned = re.sub(r"  ", " ", cleaned)

        if len(html_filename) > 0:
            with open(html_filename,'w', encoding="utf-8") as f:
                f.write(cleaned)
        return cleaned.strip()

    def clean_urls(self, text: str) -> str:
        clean_text = re.sub(r'www.\S+', "", text)
        clean_text = re.sub(r'http://\S+', "", clean_text)
        clean_text = re.sub(r'https://\S+', "", clean_text)
        clean_text = re.sub(r'@import\S+', "", clean_text)
        return clean_text

    def strip_accents(self, text: str) -> str:
        """
        Strip accents from input String.
        """
        # text = unicodedata.normalize('NFD', text)
        clean_text = ""
        for char in str(text):
            if char in self.nl_special_chars:
                clean_text += self.nl_special_chars[char]
            elif char in self.punctuation_chars:
                clean_text += self.punctuation_chars[char]
            elif not 0 <= ord(char) <= 217:
                print(f"IGNORED ASCII: {char}")
            else:
                clean_text += char
        clean_text = clean_text.encode('ascii', 'ignore').decode("utf-8")
        return clean_text
    
    def clean_initials(self, tokens: List[Dict]) -> str:
        if len(tokens) == 0: return []
        new_tokens = [t['text'] for t in tokens]
        for i, tok in enumerate(tokens[1:]):
            if tok['ner_iob'] == 'B' and tok['ner_type'] == "PERSON":
                prev_tok = tokens[i-1]
                if '.' in tok['text']:
                    new_tokens[i+1] = tok['text'].replace('.', '')
                if len(prev_tok) > 1 and '.' in prev_tok:
                    new_tokens[i] = new_tokens[i].replace('.', '')
        return new_tokens



##### ---------- MAIN FUNCTIONS ---------- #####

def get_development_set_ids(dev_dir: str, ignore_ids: List[str]) -> List[str]:
    dev_ids = []
    for folder in glob.glob(f"{dev_dir}/*"):
        for filename in glob.glob(f"{folder}/*"):
            basename = os.path.basename(filename)
            doc_id = basename.split('.')[0]
            if doc_id not in ignore_ids:
                dev_ids.append(doc_id)
    return dev_ids


def get_from_input_xml(root_dir: str, data_cleaner: DataCleaner, devset_ids: List[str] = [], testset_ids: List[str] = []) -> List[BiographyXML]:
    biography_objects = []
    train_sources, dev_sources, test_sources = [], [], [] # Statistics on the distribution of sources in the partitions

    for ix, bio_filepath in enumerate(glob.glob(root_dir)):
        tree = etree.parse(open(bio_filepath))
        bio = BiographyXML(bio_filepath, tree)
        # Bioport Source is useless, unless it has metadata....
        if bio.source == 'bioport' and len(bio.text_original) == 0 and len(bio.info.keys()) <= 1: continue

        if bio.id_composed in testset_ids:
            bio.partition = "test"
            test_sources.append(bio.source)
        elif bio.id_composed in devset_ids:
            bio.partition = "development"
            dev_sources.append(bio.source)
        else:
            bio.partition = "train"
            train_sources.append(bio.source)

        # Clean HTML from Documents
        clean_text = data_cleaner.clean_html(bio.text_original)
        clean_text = data_cleaner.strip_accents(clean_text)

        # Special Rule to detach headers from the first token in the text e.g. 'Hendrik Schoonakker1881-1964' (this is an html parser error)
        clean_text = re.sub(r"(\w+)([0-9]{4}-[0-9]{4})", r"\1 \2", clean_text)
        # Rule to Detached to words stuck together e.g. LikeThis (this is an .encode(ascii) error)
        clean_text = re.sub(r"([a-z]+)([A-Z]{1}[a-z]+)", r"\1 \2", clean_text)
        # Rule to split attached periods. For example: "predikambt.Loopbaan" (this is an html parser error)
        clean_text = re.sub(r"(\w+)(.)([A-Z]{1}[a-z]+)", r"\1\2 \3", clean_text)

        # Separate composed dates in the form '1881-1964' and '1881-64' so the tokenizers do not read them as a single token
        clean_text = re.sub(r"([0-9]{4})-([0-9]{4})", r"\1 - \2", clean_text)
        clean_text = re.sub(r"([0-9]{4})-([0-9]{2})", r"\1 - \2", clean_text)

        # Clean Raw Text: URL's
        clean_text = data_cleaner.clean_urls(clean_text)
        
        # Clean Extra Line Breaks, Double spaces, 
        clean_text = re.sub(r'[\r\n]+', " ", clean_text)
        clean_text = re.sub(r'\s+', " ", clean_text)
        
        # Clean Wikipedia Imports
        if bio.source == 'wikipedia':
            clean_text = re.sub(r'\{\{.+\}\}', " ", clean_text)


        bio.text_clean = clean_text.strip()
        bio.had_html = None
        biography_objects.append(bio)
        if ix >= 20: break

    logging.info("TRAIN:")
    logging.info(Counter(train_sources))
    logging.info("DEVELOPMENT:")
    logging.info(Counter(dev_sources))
    logging.info("TEST:")
    logging.info(Counter(test_sources))
    
    return biography_objects


def get_biographynet_objects(batch_bios_xml: List[BiographyXML], nlp: NLP_Processor) -> List[BiographyJSON]:
    batched_bios_json = []

    stanza_batch = run_stanza_batch([bio.text_clean for bio in batch_bios_xml], nlp)   

    for i, bio_xml_obj in enumerate(batch_bios_xml):
        metakeys = list(bio_xml_obj.info.keys())
        text_tokens = stanza_batch[i]["tokens"]
        data_row = BiographyJSON(
            id_person=bio_xml_obj.id_bio,
            version=bio_xml_obj.version, 
            id_composed=bio_xml_obj.id_composed,
            source=bio_xml_obj.source,
            name=bio_xml_obj.name,
            partition=bio_xml_obj.partition,
            meta_keys=metakeys,
            had_html=bio_xml_obj.had_html,
            original_path=bio_xml_obj.filepath,
            birth_pl=bio_xml_obj.info.get("<birth-place"),
            birth_tm=bio_xml_obj.info.get("<birth-time>"),
            baptism_pl=bio_xml_obj.info.get("<baptism-place>"),
            baptism_tm=bio_xml_obj.info.get("<baptism-time>"), 
            death_pl=bio_xml_obj.info.get("<death-place>"),
            death_tm=bio_xml_obj.info.get("<death-time>"),
            funeral_pl=bio_xml_obj.info.get("<funeral-place>"),
            funeral_tm=bio_xml_obj.info.get("<funeral-time>"),
            marriage_pl=bio_xml_obj.info.get("<marriage-place>"),
            marriage_tm=bio_xml_obj.info.get("<marriage-time>"),
            gender=bio_xml_obj.info.get("<gender>"),
            category=bio_xml_obj.info.get("<category>"), 
            father=bio_xml_obj.info.get("<father>"),
            mother=bio_xml_obj.info.get("<mother>"),
            partner=bio_xml_obj.info.get("<partner>"),
            religion=bio_xml_obj.info.get("<religion>"),
            educations=_category_to_list('education', bio_xml_obj.info),
            faiths=_category_to_list('faith', bio_xml_obj.info),
            occupations=_category_to_list('occupation', bio_xml_obj.info),
            residences=_category_to_list('residence', bio_xml_obj.info),
            text_original=bio_xml_obj.text_original,
            text_clean=bio_xml_obj.text_clean,
            text_tokens= text_tokens,
            text_token_objects = [asdict(tok) for tok in stanza_batch[i]["token_objs"]],
            text_sentences=stanza_batch[i]["sentences"],
            tokens_len=len(text_tokens),
            meta_len=len(metakeys)
        )
        batched_bios_json.append(data_row)
    
    return batched_bios_json


def create_labelstudio_file(biography: BiographyJSON, filepath: str):
    # logging.info(f" ----------- {biography.id_composed} -----------")
    def _detokenize_sentences(sentence_tokenized_texts, token_objs):
        all_sent_objs = []
        sent = []
        for tok in token_objs:
            if tok['is_sent_end']:
                sent.append(tok)
                all_sent_objs.append(sent)
                sent = []
            else:
                sent.append(tok)

        # print(len(sentence_tokenized_texts), len(all_sent_objs))
        assert len(sentence_tokenized_texts) == len(all_sent_objs)
        # print([len(x.split()) for x in sentence_tokenized_texts])
        # print([len(x) for x in all_sent_objs])
        detokenized_all = []
        for token_sent, objs_sent in zip(sentence_tokenized_texts, all_sent_objs):
            detokenized = ""
            assert len(token_sent.split()), len(objs_sent)
            for tok_obj in objs_sent:
                if tok_obj['space_after']:
                    detokenized += tok_obj['text'] + " "
                elif tok_obj['is_sent_end']: # CHECHE!!! This might break the no-abbr code. Double check...
                    detokenized += tok_obj['text'] + " "
                else:
                    detokenized += tok_obj['text']
            detokenized_all.append(detokenized)
        return detokenized_all

    # Create a nice 'pretty AND tokenized' text files to directly display in LabelStudio.
    text = ""
    detokenized_sentences = _detokenize_sentences(biography.text_sentences, biography.text_token_objects)
    for ix, sent in enumerate(detokenized_sentences):
        if ix > 0 and ix % 5 == 0:
            text += sent+"\n\n-~-\n\n"
        else:
            text += sent
    obj = {"text_id": biography.id_composed, "source": biography.source, "text": text}
    json.dump(obj, open(filepath, "w"), indent=2)


##### ---------- AUXILIARY FUNCTIONS ---------- #####
def get_partition_ids(filepath):
    with open(filepath) as f:
        return [x.strip() for x in f.readlines()]

def _category_to_list(cat_key: str, info_dict: Dict) -> List[Dict[str, str]]:
    '''
        - <education-k>, <education-k-begin>, <education-k-end>
        - <faith-k>, <faith-k-begin>, <faith-k-end>
        - <occupation-k>, <occupation-k-begin>, <occupation-k-end>
        - <residence-k>, <residence-k-begin>, <residence-k-end>

        RETURN: [{Cat: X, CatBegin: XX-XX-XXXX, CatEnd: XX-XX-XXXX} ...]
    '''
    catlist = []
    for key in sorted(info_dict.keys()):
        match_cat = re.match(rf'<{cat_key}-\d+>', key)
        if match_cat:
            cat_name = match_cat.string
            obj = {'name': str(info_dict[key]), 'begin': str(info_dict.get(f"{cat_name[:-1]}-begin>")), 'end': str(info_dict.get(f"{cat_name[:-1]}-end>"))}
            catlist.append(obj)
    if len(catlist) == 0: return None
    return catlist