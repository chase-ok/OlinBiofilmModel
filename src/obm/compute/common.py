
import pickle
import zlib
import urllib2

def encode(obj):
    data = zlib.compress(pickle.dumps(obj, -1), 9)
    return urllib2.base64.encodestring(data)

def decode(data):
    decoded = urllib2.base64.decodestring(data)
    return pickle.loads(zlib.decompress(decoded))

PORT = 8000
SERVER_URL = "http://localhost:{0}/".format(PORT)
SPEC_URL = SERVER_URL + "spec"
RESULT_URL = SERVER_URL + "result"
