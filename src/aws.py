
from subprocess import Popen, PIPE
import time

def launch_spot(dns):
	proc = Popen('ssh -i basic-key.pem ubuntu@{0}'.format(dns), shell=False, stdin=PIPE)
	time.sleep(3)
	proc.communicate('yes')
	time.sleep(6)
	proc.communicate('touch started.info; cd OlinBiofilmModel/src; python client.py &')
	time.sleep(7)
	proc.terminate()

launch_spot('ec2-50-112-77-238.us-west-2.compute.amazonaws.com')