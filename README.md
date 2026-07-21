# jun26_bds_int_covid

## Dataset setup

The raw dataset is not included in this repository. Files under `data/raw/` are listed in `.gitignore` so large image data stays out of Git.

After cloning, download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle and copy the extracted contents into `data/raw/`. You should end up with class folders such as `COVID`, `Normal`, `Lung_Opacity`, and `Viral Pneumonia` (along with the metadata files) directly inside `data/raw/`.

If you use the [Kaggle API](https://www.kaggle.com/docs/api):

```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip -d data/raw
```

Processed outputs are written to `data/processed/`, which is also gitignored.
