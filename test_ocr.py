from crnn.crnn import crnnOcr as crnnOcr
from PIL import Image
partImg = Image.open('/home/gavin/Desktop/test_ocr2.png')##单行文本图像
partImg = partImg.convert('L')
simPred = crnnOcr(partImg)##识别的文本
print(simPred)