#!/usr/bin/env python

import shutil
import sys

def main():
  try:
    name = sys.argv[1]
  except:
    name = 'bbai'
  shutil.make_archive(name, 'zip', 'src')
  print("saved {}.zip".format(name))

if __name__ == '__main__':
  main()
