from flask import Flask, abort, jsonify
from flask import request

from configer.dmnconfig import DMNConfig
from trainer.dynmemnettrainer import DynMemNetTrainer

app = Flask(__name__)


@app.route('/api/v1.0/dmn', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    algos_config = DMNConfig()
    dmn = DynMemNetTrainer(algos_config)
    dmn.run()
    task=True

    return jsonify({'task': task}), 201

if __name__ == '__main__':
    app.run(debug=True)


#测试
#curl -i -H "Content-Type: application/json" -X POST -d '{"title":"Read a book"}'
    # http://localhost:5000/api/v1.0/dmn