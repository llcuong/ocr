import scan
import split
import detect
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import cv2

#Step 1: Scan image
im_dir = ""
im_file_path = "static/media/upload.jpg"

scanner = scan.DocScanner(True)
valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]
if im_file_path:
    img = scanner.scan(im_file_path)
#Step 2: Split image
images = detect.Segment()

#Step 3: Ocr image
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
detector = Predictor(config)
chuoi =''
for i in range(0,len(images)):
    line = Image.fromarray(images[i])
    s = detector.predict(line)
    chuoi += s +'\n'
print('='*10)
print('\n')   
print(chuoi)
print('\n')   
print('='*10)
def display():
    return chuoi
