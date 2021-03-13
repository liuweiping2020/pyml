# -*- coding: utf-8 -*-
import asyncio

from flask import Flask, Response, json
# import tornado.web
from wasp_eureka import EurekaClient

# from flasgger import Swagger
# from flask_restplus import Api, fields, Resource

app = Flask(__name__)
# from tornado.options import define, options
#
# define("port", default=7171, help="run on the given port", type=int)


# class IndexHandler(tornado.web.RequestHandler):
#     def get(self):
#         self.write('[GET] python tornado...')


app_name = 'linjk-python-eureka-client'

ip = 'localhost'
my_eureka_url = 'http://localhost:1111'
port = 3001

loop = asyncio.get_event_loop()  # 创建事件循环

eureka = EurekaClient(app_name=app_name, port=port, ip_addr=ip,
                      hostname="localhost", eureka_url=my_eureka_url, loop=loop)


@app.route('/getUser', methods=['GET'])
def getUser():
    """
      This is the language awesomeness API
      Call this api passing a language name and get back its features
      ---
      tags:
        - Awesomeness Language API
      parameters:
        - name: language
          in: path
          type: string
          required: true
          description: The language name
        - name: size
          in: query
          type: integer
          description: size of awesomeness
      """
    result = {'username': 'python', 'password': 'python'}
    return Response(json.dumps(result), mimetype='application/json')


async def main():
    result = await eureka.register()
    print("[Register Rureka] result: %s" % result)
    app.run(port=port, host='0.0.0.0')

    while True:
        await asyncio.sleep(60)
        await eureka.renew()


if __name__ == "__main__":
    loop.run_until_complete(main())
ß