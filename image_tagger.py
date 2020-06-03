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

# set logging config
write_log_file = True
dt = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'image_tagger_info_{dt}.log')
logging.basicConfig(filename=filename, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
if not write_log_file:
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logging.info('an info')
# close loggers
logging.getLogger().handlers.clear()


# set credentials
credentials_json_path = "image-tagger-credentials.json"                
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json_path
# Instantiates a client 
client = vision.ImageAnnotatorClient() 

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

def label_single_image(path):
    # The name of the image file to annotate
    file_name = os.path.abspath(path)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        # content = image_file.read()
        content = read_downsized_img(file_name)

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    tags = [x.description for x in labels[:5]]
    tags = [x.replace('&','and') for x in tags]
    tags = [x.replace(' ','_') for x in tags]
    tags = [x.replace('__','_') for x in tags]
    print(f"proposed tags for {file_name}: {tags}")

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
            print('something failed when trying to assign ', attr)
            pass
    metadata.write()

def label_batch(path_list):
    for path in path_list:
        label_single_image(path)

def get_all_img_paths(folder):
    paths = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(tuple(img_types))]:
            paths.append(os.path.join(dirpath, filename))
    return paths

def label_images_in_folder(folder_path):
    img_paths = get_all_img_paths(folder_path)
    label_batch(img_paths)

# this happens when a tag '1st_level/2nd_level' is set in digikam
#
# $ exiftool -a G1 -s test.jpg
#  _______________________________________
# TagsList                        : 1st_level/2nd_level, 1st_level
# LastKeywordXMP                  : 1st_level/2nd_level, 1st_level
# HierarchicalSubject             : 1st_level|2nd_level, 1st_level
# CatalogSets                     : 1st_level|2nd_level, 1st_level
# Subject                         : 2nd_level, 1st_level
# Keywords                        : 2nd_level, 1st_level

######
### BATCH PROCESSING (apparently 10MB limit on batch if not uploaded to cloud)
######

# https://github.com/andrikosrikos/Google-Cloud-Support/blob/master/Google%20Vision/multiple_features_request_single_API_call.py
# https://github.com/googleapis/google-cloud-python/issues/5661#issuecomment-406694528


# # set credentials  
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/herfurtht/projects-gitreps/image-tagger/image-tagger-th-756c89d87c82.json"                
# # Instantiates a client 
# client = vision.ImageAnnotatorClient() 

# features = [
#     types.Feature(type=enums.Feature.Type.LABEL_DETECTION)
# ]

# requests = []
# for filename in ['resources/mhplus.jpg', 'resources/test.jpg']:
#     with open(filename, 'rb') as image_file:
#         image = types.Image(
#             content = image_file.read())
#     request = types.AnnotateImageRequest(
#         image=image, features=features)
#     requests.append(request)

# response = client.batch_annotate_images(requests)

# for annotation_response in response.responses:
#     print(annotation_response)

# TODO: documentation
# TODO: dry run/ class
# TODO: logger
# TODO: CLI
# TODO: streamlit
# TODO: pip env
# TODO: docker?
# TODO: don't add too similar labels (find good distance measure)
# TODO: add keywords at second level