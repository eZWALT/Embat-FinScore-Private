# data/

X Ray (Embat) dump. Fields: [`data_dictionary.md`](data_dictionary.md).

## Source

| | |
|--|--|
| URL | https://f5xe6kyx7jpysotw.public.blob.vercel-storage.com/output_hackspain_data.zip |
| Size | ~187 MB zip · ~617 MB CSVs |
| Kind | Synthetic SME treasury, 2024-09-01 → 2026-09-01 |
| Scale | 250 groups, 1,286 companies |

```bash
mkdir -p data/raw
curl -L -o data/raw/output_hackspain_data.zip \
  https://f5xe6kyx7jpysotw.public.blob.vercel-storage.com/output_hackspain_data.zip
unzip -n data/raw/output_hackspain_data.zip -d data/raw
```

The CSVs sit directly in `data/` on the team laptops (gitignored). Download the zip above if they are missing.

## Layout

```text
data/
├── README.md
├── data_dictionary.md
├── raw/          # zip + output/*.csv (gitignored)
└── processed/
```

Zip and CSVs stay out of git. The dictionary is committed.
