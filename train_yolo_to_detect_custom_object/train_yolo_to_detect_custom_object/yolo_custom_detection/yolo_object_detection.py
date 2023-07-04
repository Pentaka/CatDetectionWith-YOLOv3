import cv2
import numpy as np
import glob
import random


# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")#yolo modeli yüklemesi

# Name custom object
classes = ["kedi"]

# Images path
images_path = glob.glob(r"C:\Users\shun_\train_yolo_to_detect_custom_object\yolo_custom_detection\test\*.jpg")#test verileri


#yolo ağındaki çıktı katmanları için rastgele renk oluşturma
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#fotoğrafları rastgele karıştırma
random.shuffle(images_path)
# tüm resimleri için dolaşarak
for img_path in images_path:
    # görüntüyü yükleme ve boyutunu ayarlama
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    height, width, channels = img.shape

    # Nesne tespiti
    blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)#416x416, 512x512, 608x608, 800x800, 1024x1024
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Bilgileri ekranda gösterme
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # nesne algılandı
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Dikdörtgen koordinatları
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)#tekrarlayan tespitleri filtreleme
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN #nesneleri bularak dikdörtgen içie alma ve sınıf etiketi atma
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 2)

            # Calculate and display similarity percentage
            similarity_percentage = int(confidences[i] * 100)
            cv2.putText(img, f"{similarity_percentage}%", (x, y + 60), font, 1, color, 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()
