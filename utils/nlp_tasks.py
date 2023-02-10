from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, TypeVar, Union
import logging
logger = logging.getLogger(__name__)
import stanza
StanzaToken = TypeVar("StanzaToken")

@dataclass
class AnnotatedToken():
    id: int
    text: str 
    lemma: str
    upos: str 
    xpos: str
    feats: str
    dep_head: int
    dep_rel: str
    ner_iob: str
    srl: str
    frame: str
    start_char: int 
    end_char: int
    space_after: bool
    sent_id: int
    is_sent_start: bool
    is_sent_end: bool


def run_stanza_batch(texts: List[List[str]], nlp: stanza.Pipeline) -> Dict:

    in_docs = [stanza.Document([], text=t) for t in texts]
    stanza_docs = nlp(in_docs)
    batched_docs = []
    for text, doc in zip(texts, stanza_docs):
        stanza_info, stanza_tokens, stanza_sents = [], [], []
        stanza_entities = []
        charstarts2token, charends2token = {}, {}
        tot_toks = 0
        doc_level_token_id = 0
        for sent_ix, s in enumerate(doc.sentences):
            sent_words = []
            tok_ents = [tok.ner for tok in s.tokens]

            sentence_tokens = []
            for tok in s.words:
                sentence_tokens.append(tok)
                charstarts2token[tok.start_char] = tot_toks 
                charends2token[tok.end_char] = tot_toks + 1
                tot_toks += 1

            stanza_entities += [{'text': ent.text, 'label': ent.type, 'start': ent.start_char, 'end': ent.end_char, 'start_token': charstarts2token[ent.start_char], 'end_token': charends2token[ent.end_char]} for ent in s.ents]
            
            shifted_sentence = sentence_tokens + ['</END>']
            for ix, (tok, next_tok) in enumerate(zip(sentence_tokens, shifted_sentence[1:])):
                sent_words.append(tok.text)
                try:
                    srl_info = (tok.srl, tok.frame)
                except:
                    srl_info = (None, None)
                obj = {'id': doc_level_token_id,
                        'text': tok.text, 
                        'lemma': tok.lemma, 
                        'upos': tok.upos, 
                        'xpos': tok.xpos,
                        'feats': tok.feats,
                        'dep_head': tok.head,
                        'dep_rel': tok.deprel,
                        'ner_iob': tok_ents[ix],
                        'srl': srl_info[0],
                        'frame': srl_info[1],
                        'start_char': tok.start_char, 
                        'end_char': tok.end_char,
                        'space_after': False if next_tok != '</END>' and tok.end_char == next_tok.start_char else True,
                        'sent_id': sent_ix,
                        'is_sent_start': True if ix == 0 else False,
                        'is_sent_end': False
                        }
                nlp_token = AnnotatedToken(**obj)
                stanza_info.append(nlp_token)
                stanza_tokens.append(tok.text)
                doc_level_token_id += 1
            # The last char of a sentence needs some manual inspecion to properly set the space_after and is_sent_end!
            if len(stanza_info) > 0:
                stanza_info[-1].is_sent_end = True
                if tok.end_char < len(text):
                    lookahead_char = text[tok.end_char]
                    if lookahead_char != " ":
                        stanza_info[-1].space_after = False
                else:
                    stanza_info[-1].space_after = False
            stanza_sents.append(" ".join(sent_words))
        batched_docs.append({'stanza_doc': doc, 'sentences':stanza_sents, 'tokens': stanza_tokens, 'token_objs': stanza_info, 'entities': stanza_entities})
    return batched_docs