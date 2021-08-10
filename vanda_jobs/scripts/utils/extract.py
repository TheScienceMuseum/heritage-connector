import bz2
import gzip
import fnmatch
import json
import os



def bz2Reader(folder_path):
    print(f"checking {folder_path}, {os.path.isdir(folder_path)}")
    for dirpath, dirnames, files in os.walk(folder_path):
        count = 0
        for f in fnmatch.filter(files, "*.jsonl.bz2"):
            fileName = dirpath + "/" + f
            print(fileName)
            with bz2.open(fileName, "rb") as bz_file:
                try:
                    for line in bz_file:
                        count += 1
                        yield json.loads(line)
                finally:
                    print(
                        f'Stopped at {count} iterations and line {line}')


def gzipReader(folder_path):
    print(f"checking {folder_path}, {os.path.isdir(folder_path)}")
    for dirpath, dirnames, files in os.walk(folder_path):
        count = 0
        for f in fnmatch.filter(files, "*.jsonl.gz"):
            fileName = dirpath + "/" + f
            print(fileName)
            with gzip.open(fileName, "rb") as gz_file:
                try:
                    for line in gz_file:
                        count += 1
                        yield json.loads(line)
                finally:
                    print(
                        f'Stopped at {count} iterations and line {line}')
