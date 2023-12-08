import os, json, glob
from utils.labelstudio import text2labelstudio_tokenized


def biographynet_labelstudio_files(partition: str):
    labelstudio_path = f"outputs/biographynet/{partition}/labelstudio"
    if not os.path.exists(labelstudio_path): os.makedirs(labelstudio_path, exist_ok=True)
    all_token2spans = {}
    for i, filepath in enumerate(glob.glob(f"outputs/biographynet/{partition}/json/*")):
        bio = json.load(open(filepath))
        metadata = {} # This is the "metadata" that will appear as columns in the LabelStudio Interface
        metadata['source'] = bio['source']
        metadata['token_objects'] = bio['text_token_objects']
        task, token2spans = text2labelstudio_tokenized(bio['id_composed'], bio['text_clean'], metadata, fake_paragraph_size=5) 
        json.dump(task, open(f"{labelstudio_path}/{bio['id_composed']}.ls.json", "w"), indent=2)
        # span_obj = {'text_id': bio['id_composed'], 'token2spans': token2spans}
        all_token2spans[bio['id_composed']] = token2spans
        # fspans.write(f"{json.dumps(span_obj)}\n")
        # if i > 10: break
    with open(f"outputs/biographynet/{partition}/token2spans_{partition}.json", "w") as fspans:
        json.dump(all_token2spans, fspans, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # biographynet_labelstudio_files(partition='test')
    biographynet_labelstudio_files(partition='train')