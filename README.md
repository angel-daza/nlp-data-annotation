# nlp-data-annotation
This repo contains code to prepare data to annotate for NLP tasks (using LabelStudio). It also contains scripts to pre-annotate with models or rules, recover the annotations, and compute basic inter-annotator agreements 

The idea of the repository is to be general for annotations of future projects. Here, the BiographyNet project is showcased as an example for using the code here but more scripts can be added to preprocess and use labelStudio for annotations in other corpora.

## BiographyNet Annotations

### Pre-process Corpus

The main file of this repository is `process_biographynet.py`.

1. Create a virtual environment:
```
python -m venv .my_venv
```

2. Activate the Environment and Install the Python packages:
```
source .my_venv/bin/activate
pip install -r requirements.txt
```

3. Run the main file `python process_biographynet.py`. This will generate a subfolder indide `outputs/` that will contain the NLP pre-processed biographies inside three partitions: train, development or test. These partitions are determined by the text files `test_ids.txt` and `dev_ids.txt`, everything else will be put inside the train partition. Line 12 of the script has the variable `BIONET_XML_DIR` which should point to the root folder containing all the XML files from BiographyNet. Currently, the code is pointing to the `toy_bionet_xml` directory included in this repo to run the code as a test in 20 random BiographyNet XML Files.

### Create LabelStudio Files

The next script to run in `create_labelstudio_files.py` where the files required to be loaded with the LabelStudio interface are created. Inside this script there is an example to create the labelstudio files for the toy data included here. The output is inside the `outputs/<partition>/labelstudio` folder an includes one `*.ls.json` file per biography. The script also generates a `token2spans_<partition>.json` file containing a mapping from all tokenized biographies to their corresponding character spans. The format of this file is:
```
{
    "99999_99": {"0": [0, 5], "1": [6, 12], "3": [13, 21], ...},
    "888888_88": {"0": [0, 13], "1": [13, 22], ...}
}
```

In that example there are 2 biographies, the first one with ID `99999_99` and the second one with ID `888888_88`. The first biography has 3 tokens and the character span of the first token (Token0) starts at char 0 and ends at char 5, the next token (Token1) starts at char 6 and ends at char 12, etcetera...

Once these files are created they can be imported to labelstudio and the annotations can be performed. The following section explains how to read the exported file from LabelSudio which already will contain the annotations.

### Read LabelStudion Annotations

#### Entities (Span-Based Annotations)

For reading the span-based annotations use the script `labelstudio_annotation_reader.py`. This script can read an exported **JSON-MIN** file from label studio containing one or several annotator's spans. It has functions for several stages of annotations, including: 
1. `analyze_annotated_corpus()`: Code for inter-annotator agreement based on Krippendorf's Alpha and Relative F1 Scores across annotators.
2. `analyze_annotated_corpus()`: The same function also generates TSV Spreadsheet for analyzing annotator's divergence in annotations and correct them using gDocs Sheets or Excel. The annotation outputs in TSV can be found in the subfolder of the current dataset partition: `outputs/.../statistics/annotations_all.tsv` 
3. `generate_gold_annotations()`: Read the TSV of consensuated human_gold annotations that was corrected via the spreadsheet and generate the official gold annotations baes on the INTERSECTION or manually revised annotations of the N annotators.
4. `generate_interannotator_json()`: Generate a JSON containing all annotations from different annotators for further calcultions on the annotation task.

#### Entities + Relations (Named Relations between to annotated spans)

For reading the span-based annotations use the script `labelstudio_annotation_reader_relations.py`. This script read an exported **JSON-MIN** file from label studio containing one or several annotator's spans.