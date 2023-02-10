"""
 Label Studio Basic Input Format:
    List of Tasks = [{Task1}, ..., {TaskN}]
    Task_i = {
        "data": {
                "text": "This is the raw text sentence example ..."
                },
        "predictions": [
                        {
                        "model_version": "en_core_web_sm",
                        "result": [
                                    {
                                        "from_name": "label",
                                        "to_name": "text",
                                        "type": "labels",
                                        "value": {
                                        "start": 62,
                                        "end": 71,
                                        "text": "this hour",
                                        "labels": [
                                            "TIME"
                                        ]
                                        }
                                    },
                                    { And all the rest of <LABELED SPANS>}
                                    ...
                                ]
                        },
                        { And all the rest of <MODEL VERSIONS>}
                                    ...
                        ]
      }
"""

from typing import Dict, Callable, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict



@dataclass
class SpanAnnotation():
    annotation_layer: str
    start_char: int
    end_char: int
    text: str
    label: str
    start_token: int = None
    end_token: int = None
    tokens: List[str] = None
    text_id: str = None
    annotator_id: int = None
    annotation_id: int = None


@dataclass
class AnnotatedDocument():
    text_id: str
    text: str
    tokens: List[str]
    annotated_spans: List[SpanAnnotation]
    tokens2spans: Dict[int, Tuple[int, int]] = None # Map TokenIndex -> (CharStart, CharEnd)
    annotator_id: str = None
    annotation_time: float = None

    def get_iob(self):
        iob_labels = ['O']*len(self.tokens)
        for span in self.annotated_spans:
            labeled_span = [span.label]*len(span.tokens)
            for i, _ in enumerate(labeled_span):
                if i == 0:
                    labeled_span[i] = f"B-{labeled_span[i]}"
                else:
                    labeled_span[i] = f"I-{labeled_span[i]}"
            for counter, token_index in enumerate(range(span.start_token, span.end_token)):
                iob_labels[token_index] = labeled_span[counter]
        return iob_labels

    def to_conll(self):
        iob_labels = self.get_iob()
        conll_lines = []
        for tok, lbl in zip(self.tokens, iob_labels):
            conll_lines.append(f"{tok}\t{lbl}")
        return "\n".join(conll_lines)
    
    def to_json(self, strict_tokens: bool = False):
        all_annotations = {}
        for i, span in enumerate(self.annotated_spans):
            entity_object = {
                "ID": f"{span.annotator_id}_{i}",
                "surfaceForm": span.text,
                "category": span.label,
                "locationStart": span.start_char,
                "locationEnd": span.end_char,
                "tokenStart": span.start_token,
                "tokenEnd": span.end_token,
                "method": f"human_{span.annotator_id}"
            }
            if not strict_tokens or (strict_tokens and span.start_token and span.end_token):
                if span.annotation_layer in all_annotations:
                    all_annotations[span.annotation_layer].append(entity_object)
                else:
                    all_annotations[span.annotation_layer] = [entity_object]
        return all_annotations




def tokenize_sentences(token_objs):
    sentecized_tokens = defaultdict(list)
    token2spans = {}

    # If no sentences ids are known then we can't tokenize per sentence
    if 'sent_id' not in token_objs[0]:
        token_texts = []
        for i, tok in enumerate(token_objs):
            token_texts.append(tok['text'])
            token2spans[i] = (tok['start_char'], tok['end_char'])
        return token_texts, token2spans

    # The normal process ...
    for i, token in enumerate(token_objs):
        sentecized_tokens[token['sent_id']].append(token)
        token2spans[i] = (token['start_char'], token['end_char'])
    all_tokenized = []
    for sent_id, token_objs in sentecized_tokens.items():
        tokenized_sentece = []
        for tok in token_objs:
            tokenized_sentece.append(tok['text'])
        all_tokenized.append(" ".join(tokenized_sentece))
    return all_tokenized, token2spans


def text2labelstudio_tokenized(text_id: str, text: str, metadata: Dict[str, Any], fake_paragraph_size: int=0) -> Dict[str, Any]:
    # Create a nice 'pretty AND tokenized' text files to directly display in LabelStudio.
    tokenized_sentences, token2spans = tokenize_sentences(metadata['token_objects'])
    if fake_paragraph_size > 0:
        new_text = []
        for ix, sent in enumerate(tokenized_sentences):
            if ix > 0 and ix % fake_paragraph_size == 0:
                new_text.append("\n\n-~-\n\n")
                new_text.append(sent)
            else:
                new_text.append(sent)
    obj = {"text_id": text_id, "original": text, "text": " ".join(new_text), "sentences": tokenized_sentences}
    for k,v in metadata.items():
        if k not in ['token_objects']:
            obj[k] = v
    return obj, token2spans


def text2labelstudio_annotated(text_id: str, text: str, span_annotators: Dict[str, Callable]) -> Dict[str, Any]:
    # Create Empty LabelStudio Task
    task = {
        "data": {
                "text_id": text_id,
                "text": text,
                },
        "predictions": [] 
    }
    # Apply an annotation function or model for required outputs
    for annotator_name, annotator_function in span_annotators.items():
        result = []
        annos = annotator_function(text)
        # Transfer the Spans to LabelStudio JSON element
        for elem in annos:
            formatted = {"from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {"start": elem['start'],
                                    "end": elem['end'],
                                    "text": elem['text'],
                                    "labels": [elem['label']],
                                    }
                        }
            result.append(formatted)

        # Append to LabelStudio Task
        task["predictions"].append(
                {
                    "model_version": annotator_name,
                    "result": result
                }
        )

    return task