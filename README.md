# miscelaneous-notebooks

Quick guideline to download Kaggle datasets using the API (with Pipenv).

## Prerequisites
- Python environment managed with **Pipenv**
- A valid Kaggle API token configured

## Steps
1. Make sure you have the required dependencies:
```bash
pipenv install kaggle 
```
2. Verify that Kaggle CLI works correctly: 
```bash
kaggle --version
kaggle datasets list | head
```
3. Download the dataset you want and delete unnecessary files: 
```bash
kaggle datasets download -d <DATASET_REF>
unzip <DATASET_NAME>.zip
rm <DATASET_NAME>.zip
```
Where:

- `<DATASET_REF>` is the Kaggle dataset identifier (e.g. `owner/dataset-name`)
- `<DATASET_NAME>` is the dataset slug (the part after `/` in the ref)


