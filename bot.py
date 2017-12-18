#!/usr/bin/env python

from __future__ import print_function, division
from sys import stdin, stdout, stderr

from planner import Planner, SearchPlanner
from game import Game, piece_letters

import pickle
import os

import logging
def logger():
  return logging.getLogger(__name__)


class Parser(object):
  def __init__(self, game, planner):
    self.game = game
    self.planner = planner
    self.player_names = []

  def execute(self, line):
    parts = line.split()
    command, args = parts[0], parts[1:]
    execute_switch = {
      'action': self.act,
      'settings': self.set,
      'update': self.update
    }
    return execute_switch[command](*args)

  def act(self, type, value):
    if type == 'moves':
      self.planner.time = int(value)
      return self.planner.path()

  def set(self, type, value):
    if type == 'player_names':
      self.player_names = value.split(',')

    elif type == 'your_bot':
      self.game.own.name = value
      self.game.opp.name = next(x for x in self.player_names if x != value)

    elif type == 'field_width':
      self.game.own.width = int(value)
      self.game.opp.width = int(value)

    elif type == 'field_height':
      self.game.own.height = int(value)
      self.game.opp.height = int(value)

    elif type == 'timebank':
      self.game.timebank = int(value)

    elif type == 'time_per_move':
      self.game.time_per_move = int(value)

  def update(self, player, type, value):
    if player == 'game':
      self.update_game(type, value)
    else:
      self.update_player(player, type, value)

  def update_game(self, type, value):
    if type == 'this_piece_position':
      self.game.piece_position = tuple(int(x) for x in value.split(','))[::-1]

    elif type == 'this_piece_type':
      self.game.piece = piece_letters[value]

    elif type == 'next_piece_type':
      self.game.next_piece = piece_letters[value]

    elif type == 'round':
      self.game.round = int(value)

  def update_player(self, player, type, value):
    player = self.game.own if player == self.game.own.name else self.game.opp

    if type == 'field':
      cells = [[int(x) for x in row.split(',')] for row in value.split(';')]
      player.field.reset(cells)

    elif type == 'combo':
      player.combo = int(value)

    elif type == 'row_points':
      player.points = int(value)

    elif type == 'skips':
      player.skips = int(value)


class Bot(object):
  def __init__(self, game, planner_cls, **planner_args):
    self.game = game
    self.planner = planner_cls(self.game, **planner_args)
    self.parser = Parser(self.game, self.planner)

  def run(self):
    for line in self.input():
      try:
        moves = self.parser.execute(line)
        if not moves:
          continue
        self.output(','.join(moves) + '\n')
      except Exception:
        logger().exception('exception')

  def input(self):
    while not stdin.closed:
      try:
        line = stdin.readline().strip()
        if line:
          logger().info('input "%s"', line)
          yield line
      except EOFError:
        break

  def output(self, string):
    logger().info('output "%s"', string)
    stdout.write(string)
    stdout.flush()


def run(planner_cls = Planner, **planner_args):
  Bot(Game(), planner_cls, **planner_args).run()


def main():
  logging.basicConfig(level = logging.WARNING)
  book_dir = os.path.dirname(os.path.abspath(__file__))
  run(SearchPlanner,
      feature_weights = {'ceiling2': -0.0041221927846757751,
                          'combo': 0.1571798812494129,
                          'heights': 0.014573390080121143,
                          'holes+': (-0.6894840769652173,
                                     -0.021916193769641212),
                          'landing': 0.13924007492725471,
                          'lines': -0.45717727665593205,
                          'points': 0.12979604317378571,
                          'tspin': 0.48444853402200727,
                          'wells': -0.13033725587749273},
      book = pickle.load(open(os.path.join(book_dir, 'book.bin'), 'rb')))


# v9: holes = -1, landing = 1
# v10: holes = -0.96, landing = 0.27
# v11: holes = -0.974, landing = 0.226
# v12: holes = -0.84970604324672094,
#      landing = 0.095541495548048416,
#      lines = -0.49307883427483001,
#      points = 0.10422156286330667,
#      wells = -0.12199422823620927
# v14: v12 + bugfix
# v15: v14 + better time management
# v16: {'cap': -0.0074236320980317948,
#       'holes': -0.8202055736166215,
#       'landing': 0.16784965913721128,
#       'lines': -0.51131252702393881,
#       'points': 0.11291272027172526,
#       'wells': -0.15762111511931681}
# v17: {'cap': -0.076043942446028689,
#       'holes': -0.79223579987351411,
#       'landing': 0.16492040422921378,
#       'lines': -0.54500983747028475,
#       'points': 0.10162649659157767,
#       'wells': -0.17608480621117104,
#       ('holes', 'lines'): 0.031803926764180368}
# v18: {'ceiling2': -0.0070723872983686981,
#       'heights': 0.016581397766005184,
#       'holes': -0.80150107826478201,
#       'landing': 0.15990795093849816,
#       'lines': -0.48696135388324074,
#       'points': 0.13618890572364872,
#       'tspin': 0.23119656581943784,
#       'wells': -0.1502327430634379}
# v19: {{'ceiling2': -0.0051801887076170089,
#        'combo': 0.1571798812494129,
#        'heights': 0.014573390080121143,
#        'holes': -0.69147032540185427,
#        'landing': 0.13924007492725471,
#        'lines': -0.45717727665593205,
#        'points': 0.12979604317378571,
#        'tspin': 0.48444853402200727,
#        'wells': -0.13033725587749273}}
# v20: {'ceiling2': -0.0041221927846757751,
#       'combo': 0.1571798812494129,
#       'heights': 0.014573390080121143,
#       'holes+': (-0.6894840769652173,
#                  -0.021916193769641212),
#       'landing': 0.13924007492725471,
#       'lines': -0.45717727665593205,
#       'points': 0.12979604317378571,
#       'tspin': 0.48444853402200727,
#       'wells': -0.13033725587749273}
# v21: v20 + better time management, tspin 0.5 bonus, landing col bonus
# v22: v21 + tspin 1 bonus
# v23: v21 + tspin line completion bonus
# v25: v23 + beam search time management, tspin ceiling fix
# v26: v25 + tspin line completion bonus squared
# v29: v26 + opening book


if __name__ == '__main__':
    main()
