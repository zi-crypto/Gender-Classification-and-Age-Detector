import cv2

# General Variables
GENDER_MODEL = 'weights/deploy_gender.prototxt' # The Model Architecture (JSON Like) Containingn the Sturcture of all the Neural Netwok Layer's Definition
GENDER_PROTO = 'weights/gender_net.caffemodel'  # The PRE-Trained Gender Classification Model, Containing the model weights

AGE_MODEL    = 'weights/deploy_age.prototxt'    # The Model Architecture (JSON Like) Containingn the Sturcture of all the Neural Netwok Layer's Definition
AGE_PROTO    = "weights/age_net.caffemodel"     # The PRE-Trained Gender Classification Model, Containing the model weights

GENDER_LIST  = ["Male" ,"Female"]               # Gender List 
AGE_INTERVALS= ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)'] # Intervals of Age

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) # The Shape of the Input Image for image processing and to eliminate the effect of illunination

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Load the Gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Load the Age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

# Start the Webcam
video_capture = cv2.VideoCapture(0)

# Functions
def getFaces(frame, scaleFactor=1.1, minNeighbors=3):
    return face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

def getGender(blob):
    gender_net.setInput(blob) # Setting our blob as the model input
    gender_preds = gender_net.forward() # Model Predictions
    gender = GENDER_LIST[gender_preds[0].argmax()] # Getting the Highest Prediction with argmax() and set the gender as the corresponding element 0:Male 1:Female
    return gender

def getAge(blob):
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_INTERVALS[age_preds[0].argmax()]
    return age

while True:
    # Read a frame from the webcam
    _, frame = video_capture.read()
    
    # Convert the from to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Creating a blob to be the input for The Gender and Age Models
    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False) # Creating the blob from our frame image

    # Detect faces in the frame
    faces = getFaces(frame, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Predict the Gender
        gender = getGender(blob)

        # Predict the Age
        age = getAge(blob)

        # Draw a rectangle around the face on model and Labeling the Face as Male or Female with the age for each
        if gender == "Male":
            cv2.rectangle(frame, (x,y), (x+w, y+w), (255, 0, 0), 2) 
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, age, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+w), (203, 192, 255), 2)
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (203, 192, 255), 2)
            cv2.putText(frame, age, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (203, 192, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam
video_capture.release()

# Close all windows
cv2.destroyAllWindows()

