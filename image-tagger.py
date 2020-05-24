# https://cloud.google.com/vision/docs/quickstart-client-libraries#before-you-begin

import io 
import os 
# Imports the Google Cloud client library 
from google.cloud import vision 
from google.cloud.vision import types 
from google.cloud.vision import enums

from PIL import Image
import pyexiv2

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

# set credentials  
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/herfurtht/projects-gitreps/image-tagger/image-tagger-th-756c89d87c82.json"                
# Instantiates a client 
client = vision.ImageAnnotatorClient() 

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
    tags = [x.replace('&','') for x in tags]
    tags = [x.replace(' ','_') for x in tags]
    tags = [x.replace('__','_') for x in tags]
    print(tags)

    metadata = pyexiv2.ImageMetadata(file_name)
    metadata.read()

    # these seem to be the relevant attributes (across programs); hierarchies missing, tho
    attributes = ['Xmp.dc.subject', 'Xmp.digiKam.TagsList', 'Iptc.Application2.Keywords']
    for attr in attributes:
        try:
            if attr not in metadata.keys():
                metadata[attr] = tags
            else:
                metadata[attr].value.extend(tags)
                # remove duplicates
                metadata[attr].value=list(set(metadata[attr].value))
        except:
            print('no!')
            pass
    metadata.write()

def label_batch(path_list):
    for path in path_list:
        label_single_image(path)

label_batch(['resources/test.jpg', 'resources/mhplus.jpg'])

#
# this happens when a tag '1st_level/2nd_level' is set in digikam
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


# set credentials  
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/herfurtht/projects-gitreps/image-tagger/image-tagger-th-756c89d87c82.json"                
# Instantiates a client 
client = vision.ImageAnnotatorClient() 

features = [
    types.Feature(type=enums.Feature.Type.LABEL_DETECTION)
]

requests = []
for filename in ['resources/mhplus.jpg', 'resources/test.jpg']:
    with open(filename, 'rb') as image_file:
        image = types.Image(
            content = image_file.read())
    request = types.AnnotateImageRequest(
        image=image, features=features)
    requests.append(request)

response = client.batch_annotate_images(requests)

for annotation_response in response.responses:
    print(annotation_response)
#   do_something_with(annotation_response)


# TODO: don't add too similar labels
# TODO: add keywords at second level