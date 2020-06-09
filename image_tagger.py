# https://cloud.google.com/vision/docs/quickstart-client-libraries#before-you-begin

import io 
import os
import logging
from datetime import datetime
# Imports the Google Cloud client library 
from google.cloud import vision 
from google.cloud.vision import types 
from google.cloud.vision import enums

from PIL import Image
import pyexiv2

# valid image file types according to GCP
img_types=['JPEG','JPG','PNG','GIF','BMP','WEBP','RAW','ICO','PDF','TIFF','TIF']
img_types.extend([x.lower() for x in img_types])

# these seem to be the relevant attributes (across programs, digikam); hierarchies missing, tho
attributes = ['Xmp.dc.subject', 'Xmp.digiKam.TagsList', 'Iptc.Application2.Keywords']

credentials_json_path = "image-tagger-credentials.json"                

def read_downsized_img(path):
    # resize and export image
    # 307200px sufficient according to GCP docu
    # used instead of image_file.read()
    target_size = 307200 * 2
    img = Image.open(path)
    width, height = img.size
    old_size = width*height
    if old_size>target_size:
        ratio = target_size/old_size
        img.thumbnail((int(width*ratio), int(height*ratio)))
    buffer = io.BytesIO()
    img.save(buffer, "JPEG")
    content = buffer.getvalue()
    return content

def get_all_img_paths(folder):
    paths = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(tuple(img_types))]:
            paths.append(os.path.join(dirpath, filename))
    return paths

def only_new_keywords(new_proposal_tags, old_tags):
    # reduce new tags to those which don't have a word overlapping with existing tags
    old_words = []
    new_tags = []
    for tag in old_tags:
        old_words.extend(tag.replace('and', '_').split('_'))
    # new words in existing tags? 
    for prop in new_proposal_tags:
        if any(x in prop for x in old_words):
            print('rejected b/c of duplication: ', prop)
        else:
            new_tags.append(prop)
    return new_tags

class image_tagger():
    """
    This class contains the methods to initialize and run the image tagger
    """
    def __init__(self, credentials_path=credentials_json_path):
        self.credentials_path = credentials_path
        self.write_log_file = True
        self.dry_run = False

        # set credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json_path
        # Instantiates a client 
        self.client = vision.ImageAnnotatorClient() 
        
    def label_single_image(self, path):
        # The name of the image file to annotate
        file_name = os.path.abspath(path)

        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            # content = image_file.read()
            content = read_downsized_img(file_name)

        image = types.Image(content=content)

        # Performs label detection on the image file
        response = self.client.label_detection(image=image)
        labels = response.label_annotations

        tags = [x.description for x in labels[:5]]
        tags = [x.replace('&','and') for x in tags]
        tags = [x.replace(' ','_') for x in tags]
        tags = [x.replace('__','_') for x in tags]
        logging.info(f"proposed tags for {file_name}: {tags}")

        metadata = pyexiv2.ImageMetadata(file_name)
        metadata.read()

        for attr in attributes:
            try:
                if attr not in metadata.keys():
                    metadata[attr] = tags
                else:
                    # remove new tags witrh overlapping words
                    # tags = only_new_keywords(tags, metadata[attr].value)
                    metadata[attr].value.extend(tags)
                    # remove identical duplicates
                    metadata[attr].value=list(set(metadata[attr].value))
            except:
                logging.error('something failed when trying to assign ', attr, 'in file ', path)
                pass
        if self.dry_run == False:
             metadata.write()

    def label_batch(self, path_list):
        for path in path_list:
            self.label_single_image(path)

    def label_images_in_folder(self, folder_path, write_log_file=True, dry_run=False):
        self.write_log_file = write_log_file
        self.dry_run = dry_run

        # set up logging/printing
        dt = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        log_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'image_tagger_info_{dt}.log')
        logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
        if not self.write_log_file:
            logging.getLogger().removeHandler(logging.getLogger().handlers[0])
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        logging.info(f'dry_run={self.dry_run}')

        # run image tagging
        img_paths = get_all_img_paths(folder_path)
        self.label_batch(img_paths)

        # close loggers
        logging.getLogger().handlers.clear()

def tag_images(folder_path, credentials_path=credentials_json_path, write_log_file=True, dry_run=False):
    """This function retrieves the label annotations of all images in the folder from the GCP vision API.
        Labels can be written to metadata of the image files.

    Arguments:
        folder_path {path} -- Folder that contains relevant images. The folder is processed recursively through all levels.

    Keyword Arguments:
        credentials_path {path} --  (local) path of the json file with the credentials for the vision api instance 
        (see https://cloud.google.com/vision/docs/quickstart-client-libraries#before-you-begin).
        write_log_file {bool} -- Whether to create a log file of the run. (default: {True})
        dry_run {bool} -- Whether to "dry run", i.e. should the labels be written to the metadata. Labels are written if False. (default: {False})

    Returns:
        image_tagger object -- Can be used for running further labelings with the same instance.
    """
    tagger = image_tagger(credentials_path)
    tagger.label_images_in_folder(folder_path, write_log_file, dry_run)
    return tagger

# TODO: documentation
# TODO: pip env
# TODO: docker?
# TODO: don't add too similar labels (find good distance measure)
# TODO: add keywords at second level