import io
import os
import glob
from PIL import Image
import glob

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(__file__),
    'test.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

'''print('Labels:')
for label in labels:
    print(label.description)'''

def detect_faces(path):
    """Detects faces in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    folder = os.fsencode(path)

    filenames = []

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            filenames.append(filename)

    for name in filenames:
        with io.open(path + '/' + name, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.face_detection(image=image)
        faces = response.face_annotations

        # Names of likelihood from google.cloud.vision.enums
        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                           'LIKELY', 'VERY_LIKELY')
        landmarks_name = ('Left_Eye','Right_Eye', 'Mouth','Nose','Left_Ear','Right_Ear','Forehead',
                           'Forehead', 'Chin','Left_Eye_Brow', 'Right_Eye_Brow')

        #print('Faces:')

        img=Image.open(path + '/'+ name)


        for face in faces:
            '''
            print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
            print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
            print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
            print('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))

            vertices = (['({},{})'.format(vertex.x, vertex.y)
                        for vertex in face.bounding_poly.vertices])

            print('face bounds: {}'.format(','.join(vertices)))
            '''

            print(face.landmarks)
            #width = face.landmarks[17].position.x - face.landmarks[19].position.x
            #height = face.landmarks[18].position.y - face.landmarks[16].position.y
            #x = face.landmarks[16].position.x - width/2
            #y = face.landmarks[16].position.y
            #print(x,y,width,height)
            #left_eye = img.crop((0,0,10,10))
            #print(path + '/Left_Eye/' + name)

            #width ,height = img.size
            #print(width, height)
            #left_eye.show()
            #left_eye.save(path + '/Left_Eye/' + name)




levels = ('level-0','level-1','level-2','level-3')

for dir in levels:
    detect_faces("data_processed/"+dir)
