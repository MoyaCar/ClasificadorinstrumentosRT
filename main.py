from fastai.vision import *
import cv2 as cv
import time
import threading

#variables para modelo
root = '.'
path = Path(root)
classes = ['bajo', 'charango', 'clasica', 'contrabajo', 'electrica', 'ukelele']
data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)
learn = cnn_learner(data,models.resnet34)
#carga del modelo
learn.load('inst_rt_b')

#variables globales para opencv
cap = cv.VideoCapture(0)
begin = time.time()
fp = 0
terminado = False
prediccion = ''
font = cv.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 255, 255) 
thickness = 2

def reproducir_video():
    while(True):
        global terminado
        ret, frame = cap.read()
        frame = cv.putText(frame, prediccion, org, font,fontScale, color, thickness, cv.LINE_AA)
        cv.imshow('video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            terminado = True
            break
    cap.release()
    cv.destroyAllWindows()

def analisis_segundoplano():
    time.sleep(10)
    while(True):
        global fp
        fp = fp +1
        ret, frame = cap.read()
        

        if (fp % 90 == 0):
            predecir_img(frame)

        if (terminado == True):
            break
    

def predecir_img(frame):
    global prediccion
    cv.imwrite('frame.jpg',frame)
    print('captura guardada')
    img = open_image(path/'frame.jpg')
    print('imagen asignada')
    pred_class,pred_idx,outputs = learn.predict(img)
    print(pred_class)
    prediccion = str(pred_class)
    

t1 = threading.Thread(target=reproducir_video)
t2 = threading.Thread(target=analisis_segundoplano)

t1.start()
t2.start()

t1.join()
t2.join()


print(fp)


#pequenio algoritmo para calcular los fps. Bastante malo, pero le doy un 28fps
#lo dejo para testear en maquina con mayor potencia  o estabilidad que mi tortoshiba
#TODO modularizar!
""" end = time.time()
print(begin)
print(end)
totaltime= end - begin
print(totaltime)
print(fp)
fps= fp / totaltime
print(fps) """


# las lineas importantes que fui extrayendo del notebook. TODO, boorar y modularizar
""" root = '.'
path = Path(root)
print(path.ls()) 
classes = ['bajo', 'charango', 'clasica', 'contrabajo', 'electrica', 'ukelele']
img = open_image(path/'bajo.jpeg')
img2 = open_image(path/'traba.jpeg')
data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)
learn = cnn_learner(data,models.resnet34)
learn.load('inst_rt_b')
pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)
pred_class,pred_idx,outputs = learn.predict(img2)
print(pred_class) """