# pip install flask-swagger-ui
# pip install flask_swagger
import jieba
from flask import Flask, jsonify, abort, request
from flask_swagger import swagger
# import jiebahelper
import jieba.posseg as pseg
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/swagger'

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    SWAGGER_URL,
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Jiebao Application"
    }
)

# Register blueprint at URL
# (URL must match the one given to factory function above)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


#  https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#parameter-object


@app.route("/swagger")
def spec():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Segment API"
    return jsonify(swag)


@app.route('/')
def index():
    return 'Jiebao Segment API by Python.'


from flask import make_response


@app.errorhandler(404)
def not_found(error):
    # 当我们请求  # 2 id的资源时，可以获取，但是当我们请求#3的资源时返回了404错误。并且返回了一段奇怪的HTML错误，而不是我们期望的JSON，这是因为Flask产生了默认的404响应。客户端需要收到的都是JSON的响应，因此我们需要改进404错误处理：
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(400)
def para_error(error):
    # 数据错误
    return make_response(jsonify({'error': 'Parameter Error'}), 400)


@app.route('/segment', methods=['POST'])
def segment():
    '''
        切词。不带词性，去停词
        ---
        tags:
          - segment
        parameters:
          - in: body
            name: body
            description: 内容
            required: true
            schema:
                type: string
     '''
    a = request.data.strip()
    if a == '':
        abort(400)
    ret = jieba.cut(a)
    return ret


@app.route('/segmentpos', methods=['POST'])
def segmentpos():
    '''
        切词。带词性，去停词
        ---
        tags:
          - segment
        parameters:
          - in: body
            name: body
            description: 内容
            required: true
            schema:
                type: string
     '''
    a = request.data.strip()
    if a == '':
        abort(400)
    ret = pseg.cut(a)
    return ret


@app.route('/segmentall', methods=['POST'])
def segmentall():
    '''
        切词。带词性，不去停词
        ---
        tags:
            - segment
        parameters:
          - in: body
            name: body
            description: 内容
            required: true
            schema:
                type: string
    '''
    a = request.data.strip()
    if not a:
        abort(400)
    ret = jieba.cut(a,cut_all=True)
    return ret


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

