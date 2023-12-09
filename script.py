import cv2
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import mediapipe as mp
import os
import paho.mqtt.client as mqtt
import json
from twilio.rest import Client as whats

###########from twilio.rest import Client as whats

# Twilio credentials
account_sid = 'AC538f78d4fd0b49f50118807b49176cf5'
auth_token = 'a81691f062ed3375aecf3c57ce77258e'

# Create a Twilio client
whats = whats(account_sid, auth_token)

# Send a WhatsApp message

#############################
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
############################
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
############################
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
###########################

def on_message_rate(client, userdata, msg):
    global rate
    rate = msg.payload

    # Save the image to a file
    # with open('C:/Users/FreeComp/Desktop/test_images/sent/image.JPEG', 'wb') as image_file:
    #     image_file.write(image)
    print("Rate= ",rate)
    
    # Increment the request counter    
    # Check if the maximum number of requests has been reached
    
def on_message_photo(client, userdata, msg):
    global request_counter
    # Assuming the image is sent as a byte array
    image = msg.payload

    # Save the image to a file
    with open('C:/Users/FreeComp/Desktop/stem_projects/autism_mariam/joo.JPEG', 'wb') as image_file:
        image_file.write(image)
    print("image is saved ")
    
    # Increment the request counter
    request_counter += 1
    
    # Check if the maximum number of requests has been reached
    
################

if __name__ == '__main__':
  while 1:  
        request_counter = 0
        max_requests = 1

        client = mqtt.Client()
        client2 = mqtt.Client()
        
        # Assign the callback function
        client.on_message = on_message_rate
        client2.on_message=on_message_photo

        # Connect to the broker
        client.connect("192.168.77.179")  # Change this to your MQTT broker IP/hostname
        client2.connect("192.168.77.179")

        client2.subscribe("photo")
        client.subscribe("heartbeat")
        # Subscribe to the topic

        # Change this to your topic

        client2.loop_start()
        client.loop_start()

        try:
                # Keep the script running
            while request_counter < max_requests:
                    pass

        except KeyboardInterrupt:
                # Handle KeyboardInterrupt
                print("KeyboardInterrupt detected.")

        finally:
                # Cleanup actions
                client.loop_stop()  
                client.disconnect()
                client2.loop_stop()  
                client2.disconnect()
                print("Disconnected from MQTT broker")
                


        #########################

        ###########################
        actions = np.array(['fine', 'dengerous'])
        mp_holistic = mp.solutions.holistic # Holistic model
        mp_drawing = mp.solutions.drawing_utils # Drawing utilities

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1,1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))

        ############################
        model.load_weights('C:/Users/FreeComp/Desktop/stem_projects/autism_mariam/poseW.h5')
        sequence = []
        sentence = []
        threshold = 0.5
        i=0
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while (i<1):

                # Read feed
                
                img = cv2.imread('C:/Users/FreeComp/Desktop/stem_projects/autism_mariam/joo.JPEG')
                # Make detections
                image, results = mediapipe_detection(img, holistic)

                
                # Draw landmarks
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-1:]
                
                if len(sequence) == 1:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    massege={actions[np.argmax(res)]}
                    message = whats.messages.create(
                    from_='whatsapp:+14155238886',
                    body=f'person is  '+str(massege)+'heart rate of '+str(rate),
                    to='whatsapp:+201096037369'  # Replace with the recipient's phone number
                    )
                    print("Message sent. SID:", message.sid)
                    
                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                        break
                

                           
                i=i+1
                