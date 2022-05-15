import subprocess
import os
# python3_command = "scriptpy2.py Balabizo"  # launch your python2 script using bash
#
# process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()  # receive output from the python2 script
subprocess.call(".\script.bat ad", shell=True )
