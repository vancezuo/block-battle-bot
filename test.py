import time

from game import Game, pieces
from planner import SearchPlanner
from collections import Counter
from pprint import pprint
from openings import Book

import evaluate
import time
import random
import numpy as np
import pickle


def main():
  times = []
  turns = []
  scores = []
  ranks = Counter()
  percentile = Counter()
  num_moves = []

  for j in range(10):
    game = Game(solid_rate = 15, garbage_rate = 12)

    planner = SearchPlanner(game, 2,
                           {'ceiling2': -0.0041221927846757751,
                            'combo': 0.1571798812494129,
                            'heights': 0.014573390080121143,
                            'holes+': (-0.6894840769652173,
                                       -0.021916193769641212),
                            'landing': 0.13924007492725471,
                            'lines': -0.45717727665593205,
                            'points': 0.12979604317378571,
                            'tspin': 0.48444853402200727,
                            'wells': -0.13033725587749273},
                            pickle.load(open('book.bin', 'rb')))

    usage = 0
    tspins = 0

    for i in range(75):
      start = time.time()

      print('----')

      planner.time = (10 - usage)*1000

      move = planner.move() #, rank, num_move
      if move is None and game.own.skips == 0:
        print(game.current_placement)
        print(str(game.own.field.cells).replace(' True', '#').replace('False', '.'))
        break

      print(game.current_placement, '=>', move)

      _, tspin = game.own.move(move)
      tspins += tspin
      end = time.time()

      print(str(game.own.field.cells).replace(' True', '#').replace('False', '.'))
      print('turn:', i)
      print('points:', game.own.points)
      print('combo:', game.own.combo)
      print('skips:', game.own.skips)
      print('time:', end - start)
      print('tspins:', tspins)

      times.append(end - start)
      usage = max(0, usage + (end - start) - 0.5)

      print('overall time:', 10 - usage)

      if not game.advance():
        print('BLOCK OVERFLOW')
        break

      if usage > 10:
        print('TIMEOUT')
        break

      # if move is None:
      #   print('SKIP')
      #   time.sleep(5)

      # time.sleep(1)

    scores.append(game.own.points)
    turns.append(i)

  print('----')
  print('avg score:', sum(scores) / len(scores))
  print('avg turns:', sum(turns) / len(turns))
  print('avg time usage:', sum(times) / len(times))
  print('worst times:', sorted(times, reverse = True)[:5])

if __name__ == '__main__':
  # random.seed(12345)
  
  main()

  # import cProfile
  # cProfile.run('main()')