# encoding:utf-8
import sys

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def get_mongo_client(uri="mongodb://0.0.0.0:27017/", db="ca"):
    """
    Connect to MongoDB
    """
    try:
        c = MongoClient(uri)

    except ConnectionFailure as e:
        sys.stderr.write("Could not connect to MongoDB: %s" % e)
        sys.exit(1)

    dbh = c[db]
    print('connected succeed.')
    print(('URI: ' + uri + '  Database: ' + db))
    return dbh
