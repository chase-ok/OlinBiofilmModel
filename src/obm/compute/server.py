
from obm import specs, models, utils
from obm.compute import common
from BaseHTTPServer import *
import threading
import random
import os
import shutil

def start():
    _setup_spec_queue()
    httpd = HTTPServer(('', common.PORT), _RequestHandler)
    httpd.serve_forever()

_all_specs = []
_to_run = set()
_num_remaining = dict()
_spec_lock = threading.Lock()

def _setup_spec_queue(num_repeats=5):
    with _spec_lock:
        for spec in specs.Spec.all():
            _all_specs.append(spec.uuid)

            num_found = len(models.Result.table.raw.getWhereList('spec_uuid=="{0}"'.format(spec.uuid)))
            _num_remaining[spec.uuid] = num_repeats - num_found
            if _num_remaining[spec.uuid] > 0: _to_run.add(spec.uuid)

def _log_completed(spec_uuid):
    with _spec_lock:
        _num_remaining[spec_uuid] -= 1
        if _num_remaining[spec_uuid] <= 0:
            try:
                _to_run.remove(spec_uuid)
            except KeyError: pass

def _get_next_spec():
    with _spec_lock:
        if _to_run:
            return specs.Spec.get(random.sample(_to_run, 1)[0])
        else:
            return None


class _RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.startswith("/spec"):
            self._do_spec()
        elif self.path.startswith("/info"):
            self._do_info()
        elif self.path.startswith('/data.h5'):
            self._do_data()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/result"):
            self._do_result()
        else:
            self.send_response(404)
            self.end_headers()

    def _do_info(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        remaining = sum(_num_remaining.itervalues())
        self.wfile.write('# Remaining = {0}'.format(remaining))

    def _do_data(self):
        try:
            f = open(utils.DEFAULT_H5_FILE, 'rb')
        except IOError:
            self.send_error(404, "File not found")
            return

        self.send_response(200)
        self.send_header("Content-type", 'application/octet-stream')
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()

        shutil.copyfileobj(f, self.wfile)

    def _do_spec(self):
        spec = _get_next_spec()
        if spec:
            data = common.encode(spec)
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(400)
            self.end_headers()

    def _do_result(self):
        try:
            data = self.rfile.read(int(self.headers['Content-Length']))
            result = common.decode(data)
            result.save()
            _log_completed(result.spec_uuid)

            self.send_response(200)
            self.send_header("Content-type", 'text/plain')
            self.end_headers()
        except:
            self.send_response(404)
            self.end_headers()
