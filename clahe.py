import os
import cv2

pasta_entrada = './saudavel/artigoV3Verde512/'

pasta_saida = './saudavel/artigoV3Verde512Clahe/'
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

def equalizar_histograma(imagem):
    lab_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    equalizacao = cv2.equalizeHist(l)
    updated_lab_img = cv2.merge((equalizacao, a, b))
    equalizado_img = cv2.cvtColor(updated_lab_img, cv2.COLOR_LAB2BGR)
    return equalizado_img

for nome_arquivo in os.listdir(pasta_entrada):
    caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
    imagem = cv2.imread(caminho_entrada)
    
    if imagem is not None:
        imagem_equalizada = equalizar_histograma(imagem)
        
        caminho_saida = os.path.join(pasta_saida, nome_arquivo)
        cv2.imwrite(caminho_saida, imagem_equalizada)

