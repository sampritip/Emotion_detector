from fer import FER
import cv2
import logging
import time
vid = cv2.VideoCapture(1) #set 0 for inbuilt webcam, 1 for extrenally connected cam

while True:
    _, img = vid.read()   #real time
    #img = cv2.imread('images/surprised.jpg')   #for images
    if img is None:
            count = 0
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
    img = cv2.resize(img, (700,500))
    detector = FER(mtcnn=True)
    output = detector.detect_emotions(img)

    for dict in output:
        #print(dict)     #Uncomment for complete output
        box = dict['box']
        emotions = dict['emotions']
        max = 0
        emo = 'random'
        for key,value in emotions.items():
            if value > max:
                max = value
                emo = key
        print(emo,max)     #Prints top emotion
        cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)
        text = emo + " : " + str(max)
        cv2.putText(img,text, (box[0],box[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color = (255, 0, 0),thickness = 2)
   

    if len(output) == 0:
        print("- - - - - No faces detected - - - - -")
    cv2.imshow("image",img)
    #cv2.imwrite('surprised.jpg',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




