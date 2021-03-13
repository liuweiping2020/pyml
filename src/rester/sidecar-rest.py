import json
from flask import Flask, Response

app = Flask(__name__)


@app.route("/health")
def health():
    result = {'status': 'UP'}
    return Response(json.dumps(result), mimetype='application/json')


@app.route("/getUser")
def getUser():
    result = {'username': 'python', 'password': 'python'}
    return Response(json.dumps(result), mimetype='application/json')


@app.route("/test")
def getTest():
    result = {'username1': 'python1', 'password': 'python'}
    return Response(json.dumps(result), mimetype='application/json')


@app.route("/message/<int:id>")
def getMsg(id):
    result = {'msg': 'python1', 'id': str(id)}
    # return [result]
    return Response(json.dumps(result), mimetype='application/json')

if __name__ == '__main__':
    app.run(port=3001, host='0.0.0.0')
