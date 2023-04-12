import asyncio
import io
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition
from PIL import Image, ImageDraw, ImageFont
import urllib.request


# This key will serve all examples in this document.
API_KEY = '2bf3fa20e6d94740af639c24f5c8c3ff'

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = 'https://facial-recognition-poc.cognitiveservices.azure.com/'

# Base url for the Verify and Facelist/Large Facelist operations
# IMAGE_BASE_URL = 'https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/'

# Used in the Person Group Operations and Delete Person Group examples.
# You can call list_person_groups to print a list of preexisting PersonGroups.
# SOURCE_PERSON_GROUP_ID should be all lowercase and alphanumeric. For example, 'mygroupname' (dashes are OK).
PERSON_GROUP_ID = "novoimages" # assign a random ID (or name it anything)

# Used for the Delete Person Group example.
TARGET_PERSON_GROUP_ID = str(uuid.uuid4()) # assign a random ID (or name it anything)

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

'''
Create the PersonGroup
'''
# Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
print('Person group:', PERSON_GROUP_ID)
face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID, recognition_model='recognition_04')
# Define pavan
men = face_client.person_group_person.create(PERSON_GROUP_ID, name="varun")

'''
Detect faces and register them to each person
'''
# Find all jpeg images of friends in working directory (TBD pull from web instead)
path = os.path.join(os.path.dirname(__file__), ".\images\singleperson.jpeg")
img_file = open("{}".format(path), 'r+b')
# Add to pavan person

# Check if the image is of sufficent quality for recognition.
sufficientQuality = True
detected_faces = face_client.face.detect_with_url(url='https://whatsapp123url.blob.core.windows.net/faceapipoc/singleperson.JPG' , detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
if not detected_faces:
    raise Exception('No face detected')
for face in detected_faces:
    if face.face_attributes.quality_for_recognition != QualityForRecognition.high:
        sufficientQuality = False
        break
    face_client.person_group_person.add_face_from_url(PERSON_GROUP_ID, men.person_id, 'https://whatsapp123url.blob.core.windows.net/faceapipoc/singleperson.JPG')
    print("face {} added to person {}".format(face.face_id, men.person_id))


'''
Train PersonGroup
'''
# Train the person group
print("pg resource is {}".format(PERSON_GROUP_ID))
rawresponse = face_client.person_group.train(PERSON_GROUP_ID, raw= True)
print(rawresponse)

while (True):
    training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
    print("Training status: {}.".format(training_status.status))
    print()
    if (training_status.status is TrainingStatusType.succeeded):
        break
    elif (training_status.status is TrainingStatusType.failed):
        face_client.person_group.delete(person_group_id=PERSON_GROUP_ID)
        sys.exit('Training the person group has failed.')
    time.sleep(5)

'''
Identify a face against a defined PersonGroup
'''
# Group image for testing against
test_image = "https://whatsapp123url.blob.core.windows.net/faceapipoc/4G8A7138.JPG"

print('Pausing for 10 seconds to avoid triggering rate limit on free account...')
# time.sleep (10)

# Detect faces
face_ids = []
# We use detection model 3 to get better performance, recognition model 4 to support quality for recognition attribute.
faces = face_client.face.detect_with_url(test_image, detection_model='detection_01', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition','age', 'emotion'])
for face in faces:
    # Only take the face if it is of sufficient quality.
    if face.face_attributes.quality_for_recognition == QualityForRecognition.high or face.face_attributes.quality_for_recognition == QualityForRecognition.medium:
        face_ids.append(face.face_id)

# Identify faces
results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
print(results)
urllib.request.urlretrieve(test_image,"text.png")
img = Image.open("text.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(r'C:\Windows\Fonts\Arial.ttf', 35)
print('Identifying faces in image')
if not results:
    print('No person identified in the person group')
for identifiedFace in results:
    if len(identifiedFace.candidates) > 0:
        for face in faces:
            if face.face_id == identifiedFace.face_id:
                response_name = face_client.person_group_person.get(PERSON_GROUP_ID,identifiedFace.candidates[0].person_id)
                print('Person is identified for face ID {} in image, with a confidence of {}.'.format(identifiedFace.face_id, identifiedFace.candidates[0].confidence)) # Get topmost confidence score
                print(identifiedFace)
                face_id_project = identifiedFace.face_id
                age = face.face_attributes.age
                emotion = face.face_attributes.emotion
                neutral = '{0:.0f}%'.format(emotion.neutral * 100)
                happiness = '{0:.0f}%'.format(emotion.happiness * 100)
                anger = '{0:.0f}%'.format(emotion.anger * 100)
                sandness = '{0:.0f}%'.format(emotion.sadness * 100)
                personId = identifiedFace.candidates[0].person_id
                personName = response_name.name
                rect = face.face_rectangle
                left = rect.left
                top = rect.top
                right = rect.width + left
                bottom = rect.height + top
                draw.rectangle(((left, top), (right, bottom)), outline='green', width=5)
                draw.text((right + 4, top+70), 'personName: ' + personName, fill="#000", font=font)
                draw.text((right + 4, top+105), 'Happy: ' + happiness, fill="#000", font=font)
                draw.text((right + 4, top+140), 'Sad: ' + sandness, fill="#000", font=font)
                draw.text((right + 4, top+175), 'Angry: ' + anger, fill="#000", font=font)
                draw.text((right + 4, top+210), 'Neutral: ' + neutral, fill="#000", font=font)
                img.show()
                img.save("result.png") 
        # Verify faces
        # verify_result = face_client.face.verify_face_to_person(identifiedFace.face_id, identifiedFace.candidates[0].person_id, PERSON_GROUP_ID)
        # print('verification result: {}. confidence: {}'.format(verify_result.is_identical, verify_result.confidence))
    else:
        print('No person identified for face ID {} in image.'.format(identifiedFace.face_id))
 

# print()
# print('End of quickstart.')