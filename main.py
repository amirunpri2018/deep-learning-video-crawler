#using VGG16 model which is avalaible in keras 
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
import os
import traceback
import threading
import tensorflow as tf

#string for storing the predictions
label='' 
frame=None
model = VGG16()
#we need to have a single graph for a program
graph = tf.get_default_graph()

class PredictorThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        global graph,model
		#checking for start and end of the video
        while(frame.any()!=None):
            
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            #adding an additional dimension as required by the keras preprocess_input method   
            image = image.reshape((1,) + image.shape)
			# preprocessing the input image to make it as required by VGG16 model
            image = preprocess_input(image)
            
			with graph.as_default():
                global label
                yhat = model.predict(image) #predicts top 5 labels
                pred = decode_predictions(yhat)[0] 
                label=''
                for _,name,_ in pred:
                    label=label+" "+name          

if __name__== "__main__":
    cap = cv2.VideoCapture('test_now.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output_test_now_new.avi',fourcc, 20.0, (640,480))
    ret, original = cap.read()
    frame = cv2.resize(original, (224, 224))
	#using a separate thread to predict the labels so as to not block the UI thread & make it unresponsive/slow
    predictor_thread = PredictorThread()
    predictor_thread.start()
    
    try:
        while(cap.isOpened()):
            ret, original = cap.read()
            frame = cv2.resize(original, (224, 224))
            cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Classification", original)
            
            if ret==True:
				#checking for a particular label & then append it to a separate video stream
                if(label.find("revolver")!=-1):
                    temp=cv2.resize(original,(640,480))
                    out.write(temp)
          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
				
    except:
            tb = traceback.format_exc()
            print(tb)
            os.system('pause')
	finally:
			out.release()
            cap.release()
            frame=None
            cv2.destroyAllWindows()
		