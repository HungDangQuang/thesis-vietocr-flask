import flask
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

app = flask.Flask(__name__)
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
detector = Predictor(config)


    
@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    image_num = 1
    result_string = ""
    for file_id in files_ids:
        print("\nSaving Image ", str(image_num), "/", len(files_ids))
        imagefile = flask.request.files[file_id]
        # filename = werkzeug.utils.secure_filename(imagefile.filename)
        # print("Image Filename : " + imagefile.filename)
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # imagefile.save(timestr+'_'+filename)
        # image_num = image_num + 1
        result_string += detector.predict(Image.open(imagefile)) + " "
        # print("Recognition result:" + detector.predict(Image.open(imagefile)))
    # print("\n")

    print(result_string)
    print("\n")
    return result_string.rstrip(result_string[-1])

@app.route("/hello",methods=["GET"])
def sayHello():
    return "hello from flask"    

app.run(host="0.0.0.0",port=5002)
