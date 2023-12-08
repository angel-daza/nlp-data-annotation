from dataclasses import dataclass, asdict
import json, os
from typing import Any, List, Dict, Tuple
from sklearn.metrics import classification_report
from tabulate import tabulate
from collections import Counter, defaultdict
from nltk.metrics.agreement import AnnotationTask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from spacy.tokens import Doc

from utils.labelstudio import AnnotatedDocument, SpanAnnotation, RelationAnnotation

LABEL_2_ID = {
    "O": 0,
    "PER": 1,
    "ORG": 2,
    "LOC": 3,
    "TIME": 4,
    "ARTWORK": 5,
    "MISC": 6
}

ANNOTATOR_IDS = {
    "1": "Angel D",
    "4": "Celonie R",
    "7": "Martijn vd W",
    "8": "Ouidad B"
}


def analyze_annotated_corpus(annotation_type: str, partition: str, labelstudio_json_path: str, token2spans_path: str = None):
    if annotation_type == "entities":
        if token2spans_path:
            labelstudio_annotations, doc2annotators, annotator2docs = load_tokenized_annotation_objects(annotation_type, labelstudio_json_path, token2spans_path)
        else:
            raise NotImplementedError("Currently it is only possible to process pretokenized files. Please Provide a dictionary of Token2Spans")
    elif annotation_type == "relations":
        labelstudio_annotations, doc2annotators, annotator2docs = load_annotation_full_objects("relations", [labelstudio_json_path], token2spans_path)
    else:
        raise NotImplementedError(f"There is not code to handle annotation_type = '{annotation_type}'")
    
    f1_interagreement_dict = defaultdict(dict)
    alpha_interagreement_dict = defaultdict(dict)
    all_errors = []

    # # Annotator Basic Stats
    for ref_anno_id in annotator2docs.keys():
        print(f"============================ ANALYZING ANNOTATOR: {ref_anno_id} ============================")
        ref_annotated_docs = [doc for tid, docs in labelstudio_annotations.items() for doc in docs if doc.annotator_id == ref_anno_id]
        analyze_annotator(annotation_type, ref_anno_id, ref_annotated_docs)
        # At each iteration, ref_anno_id will be considered "gold" and the other is considered "a predictor"
        hyp_anno_ids = [hyp_anno_id for hyp_anno_id in annotator2docs.keys() if ref_anno_id != hyp_anno_id]
        print(f"(HYPS: {hyp_anno_ids})")
        errors_annotator_pair, f1_interagreement_dict, alpha_interagreement_dict = compare_annotators(ref_anno_id, hyp_anno_ids, labelstudio_annotations, 
                                                                            f1_interagreement_dict, alpha_interagreement_dict, show_label_detail=False)
        all_errors += errors_annotator_pair

    if not os.path.exists(f"outputs/biographynet/{partition}/statistics/"): os.mkdir(f"outputs/biographynet/{partition}/statistics/")
    # Save Tokenized Corpus
    tok_corpus_path = f"outputs/biographynet/{partition}/corpus_tokenized.json"
    tokenized_corpus = {}
    for tid, anno_layers in labelstudio_annotations.items():
        tokenized_corpus[tid] = {'tokens': anno_layers[0].tokens, 'token2spans': anno_layers[0].tokens2spans}
    json.dump(tokenized_corpus, open(tok_corpus_path, "w"), indent=2)

    # Save DataFrame with All Annotated Data
    if annotation_type == "entities":
        all_annotations = create_annotations_table(labelstudio_annotations, len(annotator2docs.keys()))
        annotations_df = pd.DataFrame(all_annotations)
        annotations_df.to_csv(f"outputs/biographynet/{partition}/statistics/annotations_all.tsv", sep="\t")
    elif annotation_type == "relations":
        all_annotations = create_relation_annotations_table(labelstudio_annotations, doc2annotators)
        annotations_df = pd.DataFrame(all_annotations)
        annotations_df.to_csv(f"outputs/biographynet/{partition}/statistics/annotations_relations_all.tsv", sep="\t")
    

    # Save DataFrame for Error Analysis
    error_df = pd.DataFrame(all_errors)
    error_df.to_csv(f"outputs/biographynet/{partition}/statistics/annotator_errors.tsv", sep="\t")
    
    # Average agreement and show a SINGLE MATRIX for annotators across the whole corpus
    valid_annotator_ids = list(annotator2docs.keys())
    plot_matrix_of_averages(f1_interagreement_dict, valid_annotator_ids, f'outputs/biographynet/{partition}/statistics/MEAN_f1_agreement.png')
    plot_matrix_of_averages(alpha_interagreement_dict, valid_annotator_ids, f'outputs/biographynet/{partition}/statistics/MEAN_alpha_agreement.png')
        


def load_tokenized_annotation_objects(annotation_layer: str, filepath: str, basepath_token2spans: str) -> Tuple[Dict[str, List[AnnotatedDocument]], Dict[str, List], Dict[str, List]]:
    ls_tasks = json.load(open(filepath))
    token2spans = get_token2span_mapper(basepath_token2spans)
    annotations = defaultdict(list)
    doc2annotators = defaultdict(list)
    annotator2docs = defaultdict(list)
    for task_obj in ls_tasks:
        adjusted_span_annotations, adjusted_tokens = adjust_span_annotations_tokenized(annotation_layer, task_obj, token2spans[task_obj['text_id']], task_obj['text'].split())
        annotations[task_obj['text_id']].append(AnnotatedDocument(
                                                    text_id=task_obj['text_id'],
                                                    text=task_obj['original'],
                                                    tokens=adjusted_tokens,
                                                    tokens2spans=token2spans[task_obj['text_id']],
                                                    annotator_id=task_obj['annotator'],
                                                    annotated_spans=adjusted_span_annotations,
                                                    annotation_time=task_obj['lead_time']
                                                ))
        doc2annotators[task_obj['text_id']].append(task_obj['annotator'])
        annotator2docs[task_obj['annotator']].append(task_obj['text_id'])
    return annotations, doc2annotators, annotator2docs 


def load_annotation_full_objects(annotation_layer: str, filepaths: List[str], basepath_token2spans: str) -> Tuple[Dict[str, List[AnnotatedDocument]], Dict[str, List], Dict[str, List]]:
    """Loads the exported JSON-FULL File from LabelStudio

    Args:
        annotation_layer(str): Linguistic layer name to refer to the annotation spans (semantic roles, phrases, entities, etc...)
        filepath (str): The JSON min. File that is directly produced by LabelStudio
        basepath_json_files (str): the dicrectory where the original biographynet files can be found

    Returns:
        Tuple[Dict, Dict, Dict]:
            annotations: 
            doc2annotators: {'text_id': [List[int] of Annotator IDs ]} All annotators that contributed spans in a single text
            annotator2docs: {'annotator_id': [List[str] of Text IDs ]} All texts annotated by a single annotator
    """
    ls_tasks = []
    for filepath in filepaths:
        ls_tasks += json.load(open(filepath))
    
    token2spans = get_token2span_mapper(basepath_token2spans)
    annotated_documents = defaultdict(list)

    doc2annotators = defaultdict(list)
    annotator2docs = defaultdict(list)
    for task_obj in ls_tasks:
        text_id = task_obj['data']['text_id']
        full_text = task_obj['data']['original']
        tok2span = token2spans[text_id]
        full_tokenized = []

        document_tokens = task_obj['data']['text'].split()
        visual_separator = "\n\n-~-\n\n"
        strip_visual_separator = "-~-" # This is because the strip() gets rid of the \n but they count for char spans!
        fake_charstarts2token, fake_charends2token = {}, {}
        doc_char_offset, token_counter = 0, 0
        real_tokens = []
        for tok in document_tokens:
            if tok == strip_visual_separator:
                start = doc_char_offset
                end = doc_char_offset + len(visual_separator)
                doc_char_offset += len(visual_separator) + 1
            else:
                start = doc_char_offset
                end = doc_char_offset + len(tok)
                fake_charstarts2token[start] = token_counter
                fake_charends2token[end] = token_counter
                token_counter += 1
                real_tokens.append(tok)
                doc_char_offset += len(tok) + 1

        # This Iterates per Annotation LAYER (i.e. per annotator on the same document text)
        for annotator_annos in task_obj['annotations']:
            annotated_entities, annotated_relations = [], []
            spanID_2_entity = defaultdict(dict)
            annotation_id = annotator_annos['id']
            annotator_id = annotator_annos['completed_by']
            annotation_time = annotator_annos['lead_time']
            
            for anno in annotator_annos["result"]:
                if anno.get('type') == 'relation':
                    # print("REL", anno)
                    for lbl in anno['labels']:
                        relation = RelationAnnotation('relations', text_id, anno['from_id'], anno['to_id'], lbl, lbl.split(":")[-2])
                        annotated_relations.append(relation)
                elif anno.get('type') == 'labels':
                    # start_token = charstarts2token.get(anno['value']['start'], -15893)
                    # end_token = charends2token.get(anno['value']['end'], -15894)
                    start_token = fake_charstarts2token[anno['value']['start']]
                    end_token = fake_charends2token[anno['value']['end']]
                    tokens = real_tokens[start_token:end_token+1]
                    entity = SpanAnnotation('entities',
                                            tok2span[start_token][0], 
                                            tok2span[end_token][1],
                                            anno['value']['text'], 
                                            anno['value']['labels'][0],
                                            start_token=start_token,
                                            end_token=end_token+1,
                                            tokens=tokens,
                                            text_id=text_id)
                    annotated_entities.append(entity)
                    spanID_2_entity[anno['id']] = entity
                    print(start_token, end_token, f"{start_token}{end_token}")
                    print("REL", anno)
                    print(f"'{anno['value']['text']}' <---> '{full_text[tok2span[start_token][0]:tok2span[end_token][1]]}' <--> {tokens}")
                    print("----")
            
            doc2annotators[text_id].append(annotator_id)
            annotator2docs[annotator_id].append(text_id)
            # Populate the Relation Objects with more understandable Data and Link them to EntityObjects
            for a in annotated_relations:
                subj = spanID_2_entity[a.from_id]
                obj = spanID_2_entity[a.to_id]
                a.subject_entity = subj
                a.object_entity = obj
                # if a.from_id == a.to_id:
                #     # print(a.default_subject, a.clean_label, obj.text)
                # else:
                #     # print(subj.text, a.clean_label, obj.text)
            # Create AnnotatedDocument
            document = AnnotatedDocument(text_id, full_text, full_tokenized, annotated_relations, annotated_relations, tok2span, annotator_id, annotation_time)
            annotated_documents[text_id].append(document)

    return annotated_documents, doc2annotators, annotator2docs 



def adjust_span_annotations_tokenized(annotation_layer_name: str, task_obj: Dict[str, Any], token2spans: Dict, document_tokens: List[str]) -> Tuple[List[SpanAnnotation], str]:
    visual_separator = "\n\n-~-\n\n"
    strip_visual_separator = "-~-" # This is because the strip() gets rid of the \n but they count for char spans!
    fake_charstarts2token, fake_charends2token = {}, {}
    doc_char_offset, token_counter = 0, 0
    real_tokens = []
    for tok in document_tokens:
        if tok == strip_visual_separator:
            start = doc_char_offset
            end = doc_char_offset + len(visual_separator)
            doc_char_offset += len(visual_separator) + 1
        else:
            start = doc_char_offset
            end = doc_char_offset + len(tok)
            fake_charstarts2token[start] = token_counter
            fake_charends2token[end] = token_counter
            token_counter += 1
            real_tokens.append(tok)
            doc_char_offset += len(tok) + 1
    # Fix annotations if needed
    span_annotations: List[SpanAnnotation] = []
    for x in task_obj["label"]:
        start_token = fake_charstarts2token[x['start']]
        end_token = fake_charends2token[x['end']] + 1
        tokens = real_tokens[start_token:end_token]
        s = SpanAnnotation(annotation_layer_name, 
                            token2spans[start_token][0], 
                            token2spans[end_token-1][1], 
                            " ".join(tokens), 
                            x['labels'][0],
                            start_token, 
                            end_token, 
                            tokens, 
                            text_id=task_obj['text_id'], 
                            annotator_id=task_obj['annotator'], 
                            annotation_id=task_obj['annotation_id'])
        span_annotations.append(s)
        print(f"{s.label} ---> '{task_obj['original'][s.start_char:s.end_char]}' <---> '{s.text}' {s.start_char} {s.end_char} [{s.tokens}]")
        print('--------------')
    return span_annotations, real_tokens

 
def _strip_bordering_spaces(span_text: str, start: int, end: int):
    ''' This function aims to avoid the very common mistake of including leading and/or trailing spaces in the untokenized annotated span
        in case the span was correct is just returns the same span, start and end, so there is no harm...    
    '''
    clean_start, clean_end = start, end
    if len(span_text) == 0: return start, end
    if span_text[0] == ' ':
        clean_start += 1
    if span_text[-1] == ' ':
        clean_end -= 1
    elif span_text[-1] == ',':
        clean_end -= 1
    return clean_start, clean_end


def get_token2span_mapper(filepath: str) -> Dict[str, Dict[int, Tuple[int, int]]]:
    token_map = json.load(open(filepath))
    for text_id in token_map.keys():
        token_map[text_id] = {int(k):v for k,v in token_map[text_id].items()}
    return token_map

    


def analyze_annotator(annotation_layer: str,annotator_id: int, annotations: List[AnnotatedDocument]):
    annotated_per_text ={}
    annotated_labels = []
    total_annotation_time = 0
    print(f"\n\n------------ STATS for Annotator #{annotator_id} ({ANNOTATOR_IDS.get(annotator_id)}) ------------")
    for anno_doc in annotations:
        print(f"\tAnnotated Text {anno_doc.text_id} (annotation time = {anno_doc.annotation_time/60: .2f} minutes)")
        total_annotation_time += anno_doc.annotation_time
        if annotation_layer == "entities":
            this_labels = [anno_sp.label for anno_sp in anno_doc.annotated_spans]
        elif annotation_layer == "relations":
            this_labels = [anno.clean_label for anno in anno_doc.annotated_relations]
        annotated_per_text[anno_doc.text_id] = this_labels
        annotated_labels += this_labels

    
    print(f"Total Annotated Texts = {len(annotated_per_text)}")
    print(f"Total Labels = {len(annotated_labels)}")
    print(f"Labels per Text (density) = {len(annotated_labels)/len(annotated_per_text): .2f}")
    print(f"Total Annotation Time = {total_annotation_time/60:.2f} minutes")
    print(f"Efficiency Score = {len(annotated_labels)/(total_annotation_time/60):.4f} lbl/m")
    print(f"Label Distribution:\n{tabulate(Counter(annotated_labels).most_common(), headers=['LABEL', 'COUNT'])}")




def compare_annotators(ref_annotator_id: int, hyp_annotator_ids: List[int], labelstudio_annotations: Dict[str, List[AnnotatedDocument]], 
                        f1_interannotator_dict: Dict[str, Dict], alpha_interannotator_dict: Dict[str, Dict], show_label_detail: bool) -> Tuple[List[Dict], Dict, Dict]:
    annotator_pair_errors = []

    for hyp_annotator_id in hyp_annotator_ids:
        print(f"\t ---------- {ref_annotator_id} vs ANNOTATOR: {hyp_annotator_id} ----------")
        for text_id, annotated_docs in labelstudio_annotations.items():
            print(f"\t On Text: {text_id}")
            # Get Relevant Annotated Spans for THIS Text ID (only REF vs HYP)
            gold_annotation = []
            predicted_annotation = []
            for doc in annotated_docs:
                if ref_annotator_id == doc.annotator_id:
                    gold_annotation = doc.annotated_spans
                elif hyp_annotator_id == doc.annotator_id:
                    predicted_annotation = doc.annotated_spans
            # Compute F1, Alpha and Kappa agreements for this Text
            text_scores = compare_annotators_text(gold_annotation, predicted_annotation, show_label_detail, strictness_level=3, verbose=True)
            f1_interannotator_dict[text_id][(ref_annotator_id, hyp_annotator_id)] = text_scores['f1-score']
            alpha_interannotator_dict[text_id][(ref_annotator_id, hyp_annotator_id)] = text_scores['alpha']
            # Error Analysis
            for ref, hyp in text_scores['label_errors']:
                # print("LABEL ERR", ref.text, ref.label, hyp.label)
                annotator_pair_errors.append({'text_id': ref.text_id, 
                                    'ref_annotator_id': ref.annotator_id, 
                                    'ref_annotation_id': ref.annotation_id,
                                        'hyp_annotator_id': hyp.annotator_id,
                                        'hyp_annotation_id': hyp.annotation_id,
                                        'ref_span': f"{ref.start_char}_{ref.end_char}",
                                        'hyp_span': f"{hyp.start_char}_{hyp.end_char}",
                                        'ref_ner_text': ref.text,
                                        'hyp_ner_text': hyp.text,
                                        'ref_label': ref.label,
                                        'hyp_label':hyp.label,
                                        'error_type': "LABEL_ERR"
                                        })
            for ref, hyp in text_scores['span_errors']:
                # print("SPAN ERR", f"'{ref.text}'", f"'{hyp.text}'", ref.label, hyp.label)
                annotator_pair_errors.append({'text_id': ref.text_id, 
                                    'ref_annotator_id': ref.annotator_id, 
                                    'ref_annotation_id': ref.annotation_id,
                                        'hyp_annotator_id': hyp.annotator_id,
                                        'hyp_annotation_id': hyp.annotation_id,
                                        'ref_span': f"{ref.start_char}_{ref.end_char}",
                                        'hyp_span': f"{hyp.start_char}_{hyp.end_char}",
                                        'ref_ner_text': ref.text,
                                        'hyp_ner_text': hyp.text,
                                        'ref_label': ref.label,
                                        'hyp_label':hyp.label,
                                        'error_type': "SPAN_ERR"
                                        })
    
    return annotator_pair_errors, f1_interannotator_dict, alpha_interannotator_dict


def compare_annotators_text(reference: List[SpanAnnotation], hypothesis: List[SpanAnnotation], label_detail: bool, strictness_level: int, verbose: bool) -> Dict[str, Any]:
    f1_agreement = compute_f1_agreement(reference, hypothesis, strictness_level=strictness_level, verbose=verbose)
    if label_detail:
        for label in list(LABEL_2_ID.keys()) + ["weighted avg"]:
            if label in f1_agreement['classification_report']:
                support = f1_agreement['classification_report'][label]['support']
                prec = f1_agreement['classification_report'][label]['precision']*100
                rec = f1_agreement['classification_report'][label]['recall']*100
                f1 = f1_agreement['classification_report'][label]['f1-score']*100
                if verbose:
                    print(f"\t\t{label[:4].upper()} Agreement REF <--> HYP:\t{support}\t\t{prec:.1f} P\t\t{rec:.1f} R\t\t{f1:.1f} F1")
    elif verbose:
        print(f"Agreement REF <--> HYP:\n\t{f1_agreement['classification_report']['weighted avg']}")
    # Compute Pairwise NLTK Agrements
    nltk_report = compute_nltk_pairwise_agreement(reference, hypothesis, level=strictness_level, verbose=verbose)
    return {
        'support': f1_agreement['classification_report']['weighted avg']['support'],
        'precision': f1_agreement['classification_report']['weighted avg']['precision'],
        'recall': f1_agreement['classification_report']['weighted avg']['recall'],
        'f1-score': f1_agreement['classification_report']['weighted avg']['f1-score'],
        'alpha': nltk_report['alpha'],
        'kappa': nltk_report['kappa'],
        'label_errors': f1_agreement['label_errors'],
        'span_errors': f1_agreement['span_errors']
    }
    


def compute_f1_agreement(reference: List[SpanAnnotation], hypothesis: List[SpanAnnotation], strictness_level: int, verbose: bool = True) -> Dict[str, Any]:
    """Computes Agreement between two different sets of annotations. For simplicity, always the first one is the Truth (Reference) 
        and the second one is the set of spans that we will evaluate (Hypothesis)

    Args:
        reference (List[SpanAnnotation]): Ground Truth of Annotations
        hypothesis (List[SpanAnnotation]): 'Predicted' annotations that we wish to evaluate
        level (int): 
                    Level 1 = Label Match: Only ordered labels are evaluated. Label matches are sufficient to be considered correct.
                    Level 2 = Partial Match: Start of the Spans have to match to be considered correct.
                    Level 3 = Exact Match: only spans which match both start and end limits are considered as correct.

    Returns:
        Dict[str, int]: Dictionary with the metrics after evaluating the two sets and the SpanAnnotations that contain Errors
            "classification_report": Dictionary containing the F1 Agreement Results
            "label_errors": List[SpanAnnotations] that have the same span but different labels
            "span_errors": List[SpanAnnotations] that have the same labels but differ in the span range
    """
    reference = sorted(reference, key=lambda x: x.start_char)
    hypothesis = sorted(hypothesis, key=lambda x: x.start_char)
    label_error, span_error = [], []
    if strictness_level == 1:
        ref_labels = [x.label for x in reference]
        hyp_labels = [x.label for x in hypothesis]
        if len(ref_labels) > len(hyp_labels):
            diff_len = len(ref_labels) - len(hyp_labels)
            hyp_labels = hyp_labels + ["O"]*diff_len
        elif len(ref_labels) < len(hyp_labels):
            hyp_labels = hyp_labels[:len(ref_labels)]
        for ix, ref in enumerate(reference):
            if ix < len(hypothesis) and hyp_labels[ix] != ref_labels[ix]:
                label_error.append((ref, hypothesis[ix]))
            elif ix >= len(hypothesis):
                label_error.append((ref, None))
    elif strictness_level == 2:
        ref_labels, hyp_labels = [], []
        for ref in reference:
            match_found = False
            for hyp in hypothesis:
                if ref.start_char == hyp.start_char:
                    if ref.end_char == hyp.end_char:
                        span_error.append((ref, hyp))
                    if ref.label == hyp.label:
                        match_found = True
                        ref_labels.append(ref.label)
                        hyp_labels.append(hyp.label)
                        break
                    else:
                        label_error.append((ref, hyp))
            if not match_found:
                ref_labels.append(ref.label)
                hyp_labels.append("O")
    elif strictness_level == 3:
        ref_labels, hyp_labels = [], []
        for ref in reference:
            match_found = False
            for hyp in hypothesis:
                if ref.start_char == hyp.start_char and ref.end_char != hyp.end_char:
                    span_error.append((ref, hyp))
                if ref.start_char == hyp.start_char and ref.end_char == hyp.end_char:
                    if ref.label == hyp.label:
                        match_found = True
                        ref_labels.append(ref.label)
                        hyp_labels.append(hyp.label)
                        break
                    else:
                        label_error.append((ref, hyp))
            if not match_found:
                ref_labels.append(ref.label)
                hyp_labels.append("O")
    else:
        raise ValueError("Please provide a valid level for evaluation! Valid Levels = [1,2,3]")
    
    report_dict = classification_report(ref_labels, hyp_labels, output_dict=True, zero_division=0) # dict_keys(['LOC', 'PER', ... ,'accuracy', 'macro avg', 'weighted avg'])
    if verbose:
        report_str = classification_report(ref_labels, hyp_labels, zero_division=0)
        print(report_str)

    return {"classification_report": report_dict, "label_errors": label_error, "span_errors": span_error}


def compute_nltk_pairwise_agreement(reference: List[SpanAnnotation], hypothesis: List[SpanAnnotation], level: int, verbose: bool) -> Dict[str, int]:
    """_summary_

    Args:
        reference (List[SpanAnnotation]): _description_
        hypothesis (List[SpanAnnotation]): _description_
        level (int): _description_
        method (str): _description_

    Returns:
        Dict[str, int]: _description_
    """
    # Convert to NLTK Input Tuples
    agreement_data = [] # List of (coder_id, item, label) Tuples
    reference = sorted(reference, key=lambda x: (x.start_char, x.annotator_id))
    hypothesis = sorted(hypothesis, key=lambda x: (x.start_char, x.annotator_id))
    highest_anno_id = hypothesis[-1].annotator_id
    if level == 1 or level == 2: # Partial Span Agreement (only starts match)
        for ref in reference:
            match_found = False
            for hyp in hypothesis:
                if ref.start_char == hyp.start_char:
                    match_found = True
                    agreement_data.append(((str(ref.annotator_id)), ref.text, ref.label))
                    agreement_data.append(((str(hyp.annotator_id)), hyp.text, hyp.label))
                    break
                if hyp.start_char > ref.end_char and hyp.annotator_id == highest_anno_id: break
            if not match_found:
                agreement_data.append(((str(ref.annotator_id)), ref.text, ref.label))
                agreement_data.append(((str(hyp.annotator_id)), ref.text, "O"))
    elif level == 3: # Full Span Agreement
        for ref in reference:
            match_found = False
            for hyp in hypothesis:
                if ref.start_char == hyp.start_char and ref.end_char == hyp.end_char:
                    match_found = True
                    agreement_data.append(((str(ref.annotator_id)), ref.text, ref.label))
                    agreement_data.append(((str(hyp.annotator_id)), hyp.text, hyp.label))
                    break
                if hyp.start_char > ref.end_char and hyp.annotator_id == highest_anno_id: break
            if not match_found:
                agreement_data.append(((str(ref.annotator_id)), ref.text, ref.label))
                agreement_data.append(((str(hyp.annotator_id)), ref.text, "O"))
    else:
        raise ValueError("Please provide a valid level for evaluation! Valid Levels = [1,2,3]")
    
    # Apply Desired Metric to the collected Data
    nltk_dict = {}
    task = AnnotationTask(agreement_data)
    try:
        a = task.alpha()
        nltk_dict['alpha'] = a
        if verbose: print(f"Alpha = {a}")
    except:
        nltk_dict['alpha'] = 0.0
        if verbose: print(f"Could not compute Alpha")
    try:
        k = task.kappa()
        nltk_dict['kappa'] = k
        if verbose: print(f"Kappa = {k}")
    except:
        nltk_dict['kappa'] = 0.0
        if verbose: print(f"Could not compute Kappa")
    
    return nltk_dict


def create_annotations_table(labelstudio_annotations: Dict[str, List[AnnotatedDocument]], tot_annotators: int) -> List[Dict]:
    annotation_table = []
    for text_id, annotation_layers in labelstudio_annotations.items():
        unified_spans = defaultdict(list)
        for layer in annotation_layers:
            for span in layer.annotated_spans:
                unified_spans[f"{span.start_char}_{span.end_char}"].append(span)
            
        for span_id, spans in unified_spans.items():
            span = spans[0]
            all_labels = " ".join(list(set([sp.label for sp in spans])))
            aggr_level = len(spans)/ tot_annotators
            annotation_table.append({
                'text_id': text_id,
                'annotator_id': span.annotator_id, 
                'annotated_span': span_id,
                'span_start': span.start_char,
                'span_end': span.end_char,
                'token_start': str(span.start_token),
                'token_end': str(span.end_token),
                'annotated_text': span.text,
                'annotated_label': all_labels,
                'agreement_level': aggr_level,
                'STATUS': 'CORRECT' if aggr_level >= 1.0 else '-',
                'NEW_SPAN_TOKS': '-',
                'context_window': " ".join(get_context_window(layer.tokens, span, window_size=8))       
            })

    return sorted(annotation_table, key=lambda x: (x['text_id'], x['span_start'], -x['agreement_level']))


def create_relation_annotations_table(labelstudio_annotations: Dict[str, List[AnnotatedDocument]], doc2annotators: Dict[str, List[str]]) -> List[Dict]:
    annotation_table = []

    corpus_level_relation_stats = []

    for text_id, annotation_layers in labelstudio_annotations.items():
        unified_relations = defaultdict(list)
        person_id = text_id.split("_")[0]
        for layer in annotation_layers:
            for rel in layer.annotated_relations:
                unified_relations[f"{rel.subject_entity.text}##{rel.clean_label}##{rel.object_entity.text}"].append(rel)
                corpus_level_relation_stats.append(rel.relation_type)
            
        for rel_id, rels in unified_relations.items():
            rel = rels[0]
            all_labels = " ".join(list(set([sp.clean_label for sp in rels])))

            if rel.subject_entity.text == rel.object_entity.text:
                rel_subject = person_id
                subj_rel_status = True
            else:
                rel_subject = rel.subject_entity.text
                subj_rel_status = False

            annotation_table.append({
                'text_id': text_id,
                'annotator_id': rel.subject_entity.annotator_id, 
                'annotated_relation': rel_id,
                'subj_relation_is_bio_id': subj_rel_status,
                'subj_span_start': rel.subject_entity.start_char,
                'subj_span_end': rel.subject_entity.end_char,
                'obj_span_start': rel.object_entity.start_char,
                'obj_span_end': rel.object_entity.end_char,
                'subj_token_start': str(rel.subject_entity.start_token),
                'subj_token_end': str(rel.subject_entity.end_token),
                'obj_token_start': str(rel.object_entity.start_token),
                'obj_token_end': str(rel.object_entity.end_token),
                'annotated_subj_text': rel_subject, # rel.subject_entity.text,
                'annotated_obj_text': rel.object_entity.text,
                'annotated_label': all_labels,
                'agreement_level': len(rels)/ len(doc2annotators[rel.subject_entity.text_id]),
                'STATUS': '_',
                'NEW_SPAN_TOKS': '_',
                # 'context_window': " ".join(get_context_window(layer.tokens, span, window_size=8))       
            })

    return sorted(annotation_table, key=lambda x: (x['text_id'], x['subj_span_start'], -x['agreement_level']))


def get_context_window(document_tokens: List[str], target_span: SpanAnnotation, window_size: int = 5):
    window_init, window_end = 0, len(document_tokens)
    if target_span.start_token and target_span.start_token - window_size > 0: 
        window_init = target_span.start_token - window_size
    if target_span.end_token and target_span.end_token + window_size < len(document_tokens):
        window_end = target_span.end_token + window_size
    return document_tokens[window_init:window_end]


def build_agreement_matrix(agreement_dict: Dict[str, float], valid_annotator_ids: List[str]) -> np.array:
    """_summary_

    Args:
        agreement_dict (Dict[str, float]): Dictionary with annotator-wise scores. Such as: {('9', '8'): 0.6298004338820666, ('9', '6'): 0.6069219895750508, ...}
        valid_annotator_ids (List[str]): List of annotator Ids to evaluate Such as: ['9', '8']

    Returns:
        List[List[float]]: Full Matrix with valid annotation scores, and 0 on the 'empty' spots
    """
    # Create Empty Matrix
    full_matrix = np.identity(len(valid_annotator_ids))
    valid_annotator_ids = sorted([vid for vid in valid_annotator_ids], key= lambda x: int(x))
    id_to_index = {vid: ix for ix, vid in enumerate(valid_annotator_ids)}
    # Populate Matrix
    print(agreement_dict)
    for (x, y), score in agreement_dict.items():
        if x in valid_annotator_ids and y in valid_annotator_ids:
            full_matrix[id_to_index[x], id_to_index[y]] = score
    return full_matrix
    

def plot_matrix_per_text(interagreement_dict: Dict[str, Dict[str, Any]], valid_annotator_ids: List[int], agreement_metric: str, basepath: str = "."):
    for text_id, text_agreement in interagreement_dict.items():
        print(f"--------------- {text_id} --------------")
        matrix = build_agreement_matrix(text_agreement, valid_annotator_ids=valid_annotator_ids)
        sns.heatmap(matrix, xticklabels=valid_annotator_ids, yticklabels=valid_annotator_ids, annot=True, cmap="Blues")
        plt.savefig(f'{basepath}/{text_id}_{agreement_metric}_agreement.png')
        plt.clf()


def plot_matrix_of_averages(interagreement_dict: Dict[str, Dict[str, Any]], valid_annotator_ids: List[int], filename: str = "agreement_matrix.png"):
    all_agreements = defaultdict(list)
    for text_id, text_agreement in interagreement_dict.items():
        for annotator_pair, score in text_agreement.items():
            all_agreements[annotator_pair].append(score)
    averaged_agreement = {}
    for pair, scores in all_agreements.items():
        averaged_agreement[pair] = np.mean(scores)

    matrix = build_agreement_matrix(averaged_agreement, valid_annotator_ids=valid_annotator_ids)
    print(matrix)
    sns.heatmap(matrix, xticklabels=valid_annotator_ids, yticklabels=valid_annotator_ids, annot=True, cmap="Blues")
    plt.savefig(filename)
    plt.clf()


#### Functions for STAGE 2 ####

def generate_gold_annotations(annotations_layer: str, anonotations_path: str, tokenized_corpus_path: str, output_path: str, format: str):
    tokenized_dict = json.load(open(tokenized_corpus_path))
    annotated_docs = read_annotations_table(annotations_layer, anonotations_path, tokenized_dict)
    if format == "conll":
        annotated2conll(annotated_docs, output_path)
    elif format == "json":
        annotated2json(annotated_docs, output_path)


def read_annotations_table(annotation_layer: str, anonotations_path: str, tokenized_dict: Dict[str, List[str]]) -> Dict[str, List[AnnotatedDocument]]:
    annotated_spans = defaultdict(list)
    annotations = defaultdict(list)
    with open(anonotations_path) as f:
        header = f.readline().strip("\n").split("\t")
        header[0] = 'index'
        for line in f:
            row = line.strip("\n").split("\t")
            row_dict = {header[i]: row[i] for i in range(len(row))}
            text_id = row_dict['text_id']
            if row_dict['STATUS'] == 'CORRECT':
                start, end = row_dict['token_start'], row_dict['token_end']
                label = row_dict['annotated_label']
            elif row_dict['STATUS'] == 'SPAN_CHANGE':
                start, end = row_dict['NEW_SPAN_TOKS'].split('_')
                label = row_dict['annotated_label']
            else: 
                continue
            if start != 'None' and end != 'None':
                int_start, int_end = int(start), int(end)
                span_tokens = tokenized_dict[text_id]["tokens"][int_start:int_end]
                char_start, char_end = recover_span_chars(tokenized_dict[text_id], int_start, int_end)
                if char_start and char_end and char_end - char_start > 0:
                    span_text = " ".join(span_tokens).strip()
                    annotated_spans[text_id].append(SpanAnnotation(annotation_layer, char_start, char_end, span_text, label, int_start, int_end, span_tokens, annotator_id="gold"))

    for text_id, spans in annotated_spans.items():
        annotations[text_id].append(AnnotatedDocument(text_id, "", tokens=tokenized_dict[text_id]['tokens'], annotated_spans=spans, tokens2spans=tokenized_dict[text_id]['token2spans']))

    return annotations

def recover_span_chars(tokens_dict: Dict[str, List], token_start: int, token_end: int) -> Tuple[str, int, int]:
    char_start, char_end = None, None
    relevant_spans = [sp for tok_id, sp in tokens_dict["token2spans"].items() if  token_start <= int(tok_id) < token_end]
    if len(relevant_spans) > 0:
        char_start = relevant_spans[0][0]
        char_end = relevant_spans[-1][-1]
    return char_start, char_end


def annotated2conll(documents: Dict[str, List[AnnotatedDocument]], output_path: str):
    with open(output_path, "w") as fout:
        for text_id, docs in documents.items():
            for doc in docs:
                conll_str = doc.to_conll()
                fout.write(f"## Text_ID = {doc.text_id}\n{conll_str}\n\n")


def annotated2json(documents: Dict[str, List[AnnotatedDocument]], output_path: str):
    json_docs = {}
    for text_id, docs in documents.items():
            for doc in docs:
                json_docs[text_id] = doc.to_json()
    json.dump(json_docs, open(output_path, "w"), indent=2)



def get_basic_doc_schema(text_id: str, text: str, basic_nlp_processor: str):
    json_doc = {
                "text_id": text_id,
                "nlp_preprocessor": basic_nlp_processor,
                "data": {
                    "text": text,
                    "tokenization": [],
                    "morpho_syntax": []
                }
            }
    return json_doc


def generate_interannotator_json(annotation_layer: str, labelstudio_json_path: str, output_json_path: str, token2spans_path: str, tokenized_corpus_path: str, annotations_all_tsv_path: str):
    labelstudio_annotations, _, _ = load_tokenized_annotation_objects(annotation_layer, labelstudio_json_path, token2spans_path)
    
    tokenized_dict = json.load(open(tokenized_corpus_path))
    annotated_issues_solved_docs = read_annotations_table(annotation_layer, annotations_all_tsv_path, tokenized_dict)
    
    all_annotated_docs = []
    for text_id, docs in labelstudio_annotations.items():
        json_doc = get_basic_doc_schema(text_id, docs[0].text, "stanza_nl")
        json_doc["data"]["tokenization"] = docs[0].tokens
        gold_docs = annotated_issues_solved_docs[text_id]
        all_docs = docs + gold_docs
        for doc in all_docs:
            for layer_name, spans in doc.to_json(strict_tokens=True).items():
                if layer_name in json_doc["data"]:
                    json_doc["data"][layer_name] += spans
                else:
                    json_doc["data"][layer_name] = spans
        
        all_annotated_docs.append(json_doc)
    json.dump(all_annotated_docs, open(output_json_path, "w"), indent=2)


if __name__ == "__main__":
    
    partition = "train"

    ##########################################################################################
    ######################## CASES FOR SPAN-BASED (ENTITY) ANNOTATIONS ########################
    ##########################################################################################

    # ### ----- CASE 1: Analyze and compare multiple annotators over a group of documents.
    # # Output: An Annotations Table that can be adited to solve annotators conflict
    # analyze_annotated_corpus(annotation_type="entities", 
    #                          partition=partition, 
    #                          labelstudio_json_path="testing_ls.json", # This is the file exported from the LabelStudio interface as "Export JSON MIN"
    #                          token2spans_path=f"outputs/biographynet/{partition}/token2spans_{partition}.json")
    
    # ### ----- CASE 2: An Annotated table with the conflicts solved is already produced in CASE 1. 
    # # Here we output a file with the GOLD ANNOTATIONS (e.g. in CoNLL-U Format) 
    # generate_gold_annotations(annotations_layer="entities",
    #                             anonotations_path=f"outputs/biographynet/{partition}/statistics/annotations_all.tsv", 
    #                             tokenized_corpus_path=f"outputs/biographynet/{partition}/corpus_tokenized.json",  
    #                             output_path=f"outputs/biographynet/{partition}/testing_ls_gold.json",
    #                             format="json"
    #                         )

    # generate_gold_annotations(annotations_layer="entities",
    #                             anonotations_path=f"outputs/biographynet/{partition}/statistics/annotations_all.tsv", 
    #                             tokenized_corpus_path=f"outputs/biographynet/{partition}/corpus_tokenized.json",  
    #                             output_path=f"outputs/biographynet/{partition}/testing_ls_gold.conll",
    #                             format="conll"
    #                        )

    # ### ----- CASE 3: Generate a JSON containing all of the seen annotations (can be loaded by future scripts to compute/visualize further interannotator operations)
    # generate_interannotator_json(annotation_layer="entities", 
    #                              labelstudio_json_path="testing_ls.json", 
    #                              output_json_path=f"outputs/biographynet/{partition}/testing_ls_all_humans.json", 
    #                              token2spans_path=f"outputs/biographynet/{partition}/token2spans_{partition}.json", 
    #                              tokenized_corpus_path=f"outputs/biographynet/{partition}/corpus_tokenized.json",
    #                              annotations_all_tsv_path=f"outputs/biographynet/{partition}/statistics/annotations_all.tsv")
    

    ##########################################################################################
    ################# CASES FOR RELATION-BASED (ENTITY+RELATION) ANNOTATIONS #################
    ##########################################################################################
    analyze_annotated_corpus(annotation_type="relations", 
                             partition=partition, 
                             labelstudio_json_path="testing_ls_full.json", # This is the file exported from the LabelStudio interface as "Export Full JSON"
                             token2spans_path=f"outputs/biographynet/{partition}/token2spans_{partition}.json")