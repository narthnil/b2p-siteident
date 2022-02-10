import glob
import shutil
import time

from os import path

for model in glob.glob("./results/ssl-*/*/v*"):
    if path.isfile(path.join(model, "logs.csv")):
        continue
    else:
        if ((not path.isfile(path.join(model, "model_best.pth.tar"))
             or not path.isfile(path.join(model, "checkpoint.pth.tar"))) and
                len(glob.glob(path.join(model, "*"))) < 2):
            print(model)
            opts_fp = glob.glob(path.join(model, "*"))[0]
            last_modified = (time.time() - path.getmtime(opts_fp)) / 60 / 60
            print("Last modified: {:.2f}hrs".format(last_modified))
            if last_modified > 12:
                shutil.rmtree(model)
