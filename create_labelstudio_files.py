import os, json
from utils.labelstudio import text2labelstudio_tokenized


def biographynet_labelstudio_files():
    labelstudio_path = "outputs_cheche/biographynet/test/labelstudio"
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





if __name__ == '__main__':
    biographynet_labelstudio_files()