from datetime import datetime
from typing import Dict, List
import logging, sys, json, os, re
from dataclasses import asdict
import stanza

from utils.biographynet import DataCleaner, get_biographynet_objects, get_from_input_xml, get_partition_ids


BIONET_XML_DIR = "/Users/daza/DATA/BiographyNet/bioport_export_2017-03-10/"
BIONET_OUTPUT_DIR = "outputs/biographynet"
NLP_BATCH_SIZE = 4


def main():
    
    # Logging Config
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{BIONET_OUTPUT_DIR}/bionet_to_json.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr], datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Start Logging")

    # Create Output Folders
    if not os.path.exists(BIONET_OUTPUT_DIR): os.makedirs(BIONET_OUTPUT_DIR)
    if not os.path.exists(BIONET_OUTPUT_DIR+"/train"): os.makedirs(BIONET_OUTPUT_DIR+"/train")
    if not os.path.exists(BIONET_OUTPUT_DIR+"/test"): os.makedirs(BIONET_OUTPUT_DIR+"/test")
    if not os.path.exists(BIONET_OUTPUT_DIR+"/development"): os.makedirs(BIONET_OUTPUT_DIR+"/development")
    for part in ["train", "development", "test"]:
        folder_path = f"{BIONET_OUTPUT_DIR}/{part}"
        if not os.path.exists(f"{folder_path}/text"): os.makedirs(f"{folder_path}/text")
        if not os.path.exists(f"{folder_path}/json"): os.makedirs(f"{folder_path}/json")
    
    # Get the IDs for Test Set Partition
    testset_ids = get_partition_ids("/Users/daza/DATA/BiographyNet/test_ids.txt")
    # Get the IDs for Development Partition
    devset_ids = get_partition_ids("/Users/daza/DATA/BiographyNet/dev_ids.txt")
    print(testset_ids)

    root_dir = f"{BIONET_XML_DIR}/*" 
    file_counter, empty_bios, with_html = 0, 0, 0
    all_meta_keys = set()
    data_cleaner = DataCleaner()

    # Get Info from XML and build Global Raw Vocabularies
    biography_xml_objects = get_from_input_xml(root_dir, data_cleaner, devset_ids, testset_ids)

    # Add Basic NLP Processing to Biographies - And Create Dataset Partitions -
    stanza_nlp = stanza.Pipeline(lang="nl", processors="tokenize,lemma,pos,depparse")
    total_batches = len(biography_xml_objects) // NLP_BATCH_SIZE + 1
    with open(f'{BIONET_OUTPUT_DIR}/biographynet_metadata.jsonl', 'w') as fout:
        for ix in range(0, len(biography_xml_objects), NLP_BATCH_SIZE):
            bio_xml_objects = biography_xml_objects[ix:ix+NLP_BATCH_SIZE]
            curr_batch = ix//NLP_BATCH_SIZE + 1
            print(f"-------- PROCESSING BATCH {curr_batch}/{total_batches} ---------")
            data_rows = get_biographynet_objects(bio_xml_objects, stanza_nlp)
            for data_row, bio_xml_obj in zip(data_rows, bio_xml_objects):
                # Write Biography to File
                fout.write(json.dumps(asdict(data_row))+"\n")
                if bio_xml_obj.partition == "test":
                    folder_path = BIONET_OUTPUT_DIR + "/test"
                elif bio_xml_obj.partition == "development":
                    folder_path = BIONET_OUTPUT_DIR + "/development"
                else:
                    folder_path = BIONET_OUTPUT_DIR + "/train"
                # Create Text File
                if len(data_row.text_tokens) == 0: 
                    empty_bios += 1
                    bio_to_txt_file(f"{folder_path}/text", data_row.id_person, data_row.version, data_row.source, data_row.text_clean)
                # Create Invididual JSON for a Biography (equivalent to a row in the biographynet_metadata.jsonl)
                with open(f"{folder_path}/json/{data_row.id_person}_{data_row.version}.json", "w") as individual_fout:
                    json.dump(asdict(data_row), individual_fout, indent=2)
                # Control Statements
                file_counter+=1
    
    print(f"Total Files Analyzed = {file_counter}")
    print(f"\nEmpty Biographies = {empty_bios} of {file_counter}")


def bio_to_txt_file(filepath, id, version, source, text):
    out_path = f"{filepath}/{id}_{version}.{source}.txt"
    with open(out_path, "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()