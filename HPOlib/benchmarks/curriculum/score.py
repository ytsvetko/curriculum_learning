import HPOlib.benchmark_util as benchmark_util
import HPOlib.benchmark_functions as benchmark_functions

import time
import sys
import subprocess
import json
import hashlib

def DictHash(d):
  m = hashlib.md5()
  m.update(str(tuple(sorted(d.items()))).encode("utf-8"))
  return m.hexdigest()

def main():
  eval_mode = sys.argv[1]
  train_mode = sys.argv[2]
  del sys.argv[2]
  del sys.argv[1]

  starttime = time.time()
  args, params = benchmark_util.parse_cli()
  assert args["folds"] == "1", args
  params_hash = DictHash(params)
  sys.stderr.write("Run hash: {}\n".format(params_hash))
  with open(params_hash + ".out", "w") as stdout_file:
    with open(params_hash + ".err", "w") as stderr_file:
      exit_code = subprocess.call(
          ["../score3.py", json.dumps(params), params_hash, eval_mode, train_mode],
          stdout=stdout_file, stderr=stderr_file)
      if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code)
  output = open(params_hash + ".out").readlines()[-1]
  result = float(output.strip().split()[-1])
  duration = time.time() - starttime
  print "Result for ParamILS: SAT, {}, 1, {}, -1, {}".format(
      abs(duration), result, str(__file__))

if __name__ == "__main__":
  main()
    
