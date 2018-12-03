from flask import Flask, request
app = Flask(__name__)


@app.route('/', defaults={'path': ''}, methods=['POST', 'GET'])
@app.route('/<path:path>')
def hello(path):
    print(request.data)
    return 'hoge'