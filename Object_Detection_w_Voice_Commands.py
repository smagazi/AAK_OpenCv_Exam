import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound

def speech(text):
    print(text)
    lang = "en"
    output = gTTS(text=text, lang=lang, slow=False)

    file_location = "./sounds/output.mp3"
    output.save(file_location)
    playsound(file_location)


# define the video stream
# possible args: 0 for built in laptop camera, 1 for external web cameras
# correspond to your laptop's front camera
video_stream = cv2.VideoCapture(0)
labels = list()

while True:
    
    ret, frame = video_stream.read() #unpack each frame
    bbox, label, conf = cv.detect_common_objects(frame) #conf: identifies what the object is, outputs a float value
    output_image = draw_bbox(frame, bbox, label, conf) #frame: image in the form of a numPy array

    cv2.imshow("Object Detection", output_image) #display the image with the box around the detected object

    for item in label: #adds objects that have not been seen yet into the labels list
        if item in labels:
            continue
        labels.append(item)

    if cv2.waitKey(1) & 0xFF == ord("q"): #exit this while True if 'q' is pressed
        break

#function that determines whether a word needs to be preceeded by 'a' or 'an'
vowels = ['a', 'e', 'i', 'o', 'u']
def a_or_an(word):
    first_letter = word[0]
    if first_letter in vowels:
        return "an"
    return "a"

#using the list of all objects seen, this code creates the *grammatical* sentence that will be spoken
#using gTTS (aka Google's Text to Speech library)
i = 0
new_sentence = []
for label in labels:
    determiner_word = a_or_an(label)
    if i == 0:
        new_sentence.append("I found", determiner_word, f"{label}, and, ")
    else:
        new_sentence.append(determiner_word, f"{label}")
    i += 1

speech(" ".join(new_sentence))