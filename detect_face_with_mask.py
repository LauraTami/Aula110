# importe a biblioteca opencv
import cv2
import numpy as np

import tensorflow as tf

model= tf.keras.models.load_model('keras_model.h5')
  
# defina um objeto de captura de vídeo
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture o vídeo quadro a quadro
    ret, frame = vid.read()

    img= cv2.resize(frame,(224,224))

    test_image= np.array(img,dtype=np.float32)
    test_image= np.expand_dims(test_image, axis=0)

    normalised_image= test_image/225.0

    prediction= model.predict(normalised_image)

    print("Previsão: ", prediction)
  
    # Exiba o quadro resultante
    cv2.imshow('quadro', frame)
      
    # Saia da tela com a barra de espaço
    key = cv2.waitKey(1)
    
    if key == 32:
        print("Fechando")
        break
  
# Após o loop, libere o objeto capturado
vid.release()

# Destrua todas as janelas
cv2.destroyAllWindows()