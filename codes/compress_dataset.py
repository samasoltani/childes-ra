import gzip
import shutil

with open('dom_neural_dataset.json', 'rb') as f_in:
    with gzip.open('dom_neural_dataset.json.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("فایل فشرده ساخته شد: dom_neural_dataset.json.gz")
