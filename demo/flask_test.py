from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify
import base64
from time import gmtime, strftime

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('test.html')

  
@app.route("/colorization/", methods=['POST'])
def sum():
    rgba = request.form.get("rgba")
    width = int(float(request.form.get("width")))
    height = int(float(request.form.get("height")))
    z = float(request.form.get("z"))

    imgdata = base64.b64decode(rgba.split(',')[1])
    prefix = rgba.split(',')[0]
    timestamp = strftime("%Y%m%d%H%M%S", gmtime())

    f_name = './data/hint/hint_'+ timestamp +'.png'
    with open(f_name, 'wb') as f:
        f.write(imgdata)
        f.close()

    #f_name = './data/output/output_'+ timestamp +'.png'

    #for test
    f_name = 'test.png'
    with open(f_name, 'rb') as f:
        output = base64.b64encode(f.read()).decode("utf-8")
        f.close()

    data = {'output': prefix+","+output}
    data = jsonify(data)
    return data

@app.route("/simplification/", methods=['POST'])
def simplification():
    line = request.form.get("line")
    width = int(float(request.form.get("width")))
    height = int(float(request.form.get("height")))

    imgdata = base64.b64decode(line.split(',')[1])
    prefix = line.split(',')[0]
    timestamp = strftime("%Y%m%d%H%M%S", gmtime())

    f_name = './data/line/line_'+ timestamp +'.png'
    with open(f_name, 'wb') as f:
        f.write(imgdata)
        f.close()

    #f_name = './data/simplified/simplified_'+ timestamp +'.png'

    #for test
    f_name = 'test.png'
    with open(f_name, 'rb') as f:
        output = base64.b64encode(f.read()).decode("utf-8")
        f.close()

    data = {'output': prefix+","+output}
    data = jsonify(data)
    return data

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234)
