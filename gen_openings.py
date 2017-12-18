#!/usr/bin/env python

from __future__ import print_function, division
from game import Game, Placement, pieces, letter_pieces, piece_letters
from collections import deque
from copy import copy
from openings import Book
from planner import SearchPlanner

import numpy as np
import planner
import pickle
import sys

def main():
  d = 3
  planner = SearchPlanner(Game(), d, {'ceiling2': -0.0041221927846757751,
                                      'combo': 0.1571798812494129,
                                      'heights': 0.014573390080121143,
                                      'holes+': (-0.6894840769652173,
                                                 -0.021916193769641212),
                                      'landing': 0.13924007492725471,
                                      'lines': -0.45717727665593205,
                                      'points': 0.12979604317378571,
                                      'tspin': 0.48444853402200727,
                                      'wells': -0.13033725587749273})
  book = Book()
  n = 3
  fout = 'tbook.bin'
  try:
    print('Preparing book with d =', d, 'and n =', n)
    book.prepare(planner, n)
    print('Created', sum(len(x) for x in book.values()), 'entries', file=sys.stderr)
  finally:
    print('Writing to', fout)
    pickle.dump(book, open(fout, 'wb'), protocol = 2, fix_imports = True)

if __name__ == '__main__':
  main()
