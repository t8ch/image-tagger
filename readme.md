# image tagger

Using the GCP vision API for labeling (local) images and writing as tags to metadata

## instructions
- If not done, set up google cloud platform account and vision API and download crendtials (https://cloud.google.com/vision/docs/quickstart-client-libraries#before-you-begin)
- Set up python environment; via `pip`: run `pip install -r pip-env.txt` (in your loc env.)
- from `image-tagger.py` run `tag_images()`:
```
tag_images(folder_path, credentials_path='image-tagger-credentials.json', write_log_file=True, dry_run=False)
    This function retrieves the label annotations of all images in the folder from the GCP vision API.
        Labels can be written to metadata of the image files.
     
    Arguments:
        folder_path {path} -- Folder that contains relevant images.
                              The folder is processed recursively through all levels.
     
    Keyword Arguments:
        credentials_path {path} --  (local) path of the json file with the credentials for the vision api instance
                                    (see https://cloud.google.com/vision/docs/quickstart-client-libraries#before-you-begin for instructions).
        write_log_file {bool} --    Whether to create a log file of the run. (default: {True})
        dry_run {bool} --   Whether to "dry run", i.e. should the labels be written to the metadata. Labels are written if False. (default: {False})
     
    Returns:
        image_tagger object -- Can be used for running further labelings with the same instance.
```