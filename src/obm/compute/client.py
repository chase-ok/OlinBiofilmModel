
from obm import models
from obm.compute import common
import urllib2
import time

def compute(spec_data):
    spec = common.decode(spec_data)
    result = models.compute_probabilistic(spec)
    return common.encode(result)

def get_spec_data():
    return urllib2.urlopen(common.SPEC_URL).read().strip()

def send_result_data(result_data):
    request = urllib2.Request(common.RESULT_URL)
    request.add_data(result_data)
    urllib2.urlopen(request)

def start():
    while True:
        try:
            result_data = compute(get_spec_data())
            send_result_data(result_data)
        except:
            print "Error!"
            time.sleep(3)
