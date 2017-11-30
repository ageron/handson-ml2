#!/usr/bin/env python

"""

********************************
DANGER - W.I.P. - TESTING ONLY!!
********************************

Clean jupyter notebook for git operations
Based on "Keeping IPython notebooks under Git version control"
(see:
  https://gist.github.com/pbugnion/ea2797393033b54674af
  http://pascalbugnion.net/blog/ipython-notebooks-and-git.html
  http://stackoverflow.com/a/20844506/827862
)
"""

import sys
import json

sys.stderr.write("\n\nCAUTION ! W.I.P ! Only dropping some test metadata, don't commit!\n\n")

def log(x):
  sys.stderr.write("\n\n[{}]\n\n\n".format(x))
def logj(x):
  sys.stderr.write("\n\n")
  json.dump(x, sys.stderr, sort_keys=True, indent=1, separators=(",",": "))
  sys.stderr.write("\n\n")

log(sys.argv)
#sys.exit(17)

nb = sys.stdin.read()
json_in = json.loads(nb)

logj(json_in["metadata"])
del json_in["metadata"]["nav_menu"]
del json_in["metadata"]["toc"]
json_in["metadata"]["language_info"]["version"]="17.0"
logj(json_in["metadata"])

json.dump(json_in, sys.stdout, sort_keys=True, indent=1, separators=(",",": "))
