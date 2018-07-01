How to run code:

- Create `./data/` folder.
- Download `training.zip` from https://drive.google.com/uc?export=download&id=1XU0YQkH5jEmg7OBXsH6uX1shCd7a2gRD and extract it in `./data/`
- Download `test_images.tar.gz` from https://drive.google.com/uc?export=download&id=195--p90lFpiqcdtNGpUq2RsGsKM-dZ5j and extract it in `./data/`

Now the folder structure is:

data/
  |- training/
  |    |- images/
  |    |- groundtruth/
  |- test_images/

- Run `python tf_areal_images.py`
- Run `python mask_to_submissions.py` to get submissions.csv file
