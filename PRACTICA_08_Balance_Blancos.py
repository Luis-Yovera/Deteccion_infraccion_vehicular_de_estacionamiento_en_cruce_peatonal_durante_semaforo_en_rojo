import cv2
import numpy as np

def ajustar_balance_yuv(imagen):
    # Se convierte la imagen al espacio de color YUV
    yuv_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)
    
    # Se separan los canales Y, U y V
    canal_y = yuv_img[:,:,0].copy()
    canal_u = yuv_img[:,:,1].copy()
    canal_v = yuv_img[:,:,2].copy()
    
    # Luego se calculan la media de los canales U y V
    media_u = np.mean(canal_u)
    media_v = np.mean(canal_v)
    
    # Se ajustan los canales U y V para equilibrar la imagen
    canal_u = cv2.add(canal_u, np.array([128 - media_u], dtype=np.uint8))
    canal_v = cv2.add(canal_v, np.array([128 - media_v], dtype=np.uint8))
    
    # Combinación de los canales ajustados
    yuv_equilibrado = cv2.merge([canal_y, canal_u, canal_v])

    # Convertir la imagen ajustada de nuevo al espacio de color RGB
    imagen_balanceada = cv2.cvtColor(yuv_equilibrado, cv2.COLOR_YUV2BGR)
    
    return imagen_balanceada

# Se incializa el video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar el ajuste de balance
    FRAME_AJUSTADO = ajustar_balance_yuv(frame)

    # Mostrar la imagen original y la ajustada
    cv2.imshow("Original", frame)
    cv2.imshow("Balance de Color", FRAME_AJUSTADO)

    # 'q' para salir 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()