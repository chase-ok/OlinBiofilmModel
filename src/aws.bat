
@echo off

for /F %%x in (dns.txt) do start cmd /c "(echo touch started.info && echo cd OlinBiofilmModel/src && echo nohup python client.py) | ssh -o UserKnownHostsFile=NUL -o StrictHostKeyChecking=no -i basic-key.pem ubuntu@%%x"