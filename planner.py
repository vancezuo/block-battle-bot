#!/usr/bin/env python

from __future__ import print_function, division

import random
import search
import game
import pickle

import logging
def logger():
  return logging.getLogger(__name__)

import sys
if sys.version_info[0] < 3:
    range = xrange


class Planner(object):
  def __init__(self, game):
    self.game = game
    self.time = 10000

  def move(self):
    moves = self.game.own.field.moves(self.game.current_placement)
    return random.choice(moves) if moves else None

  def path(self, move = None):
    start = self.game.current_placement
    end = move if move else self.move()
    default = ['skip'] if self.game.own.skips else ['no_moves']
    logger().info('end: %s', end)
    return self.game.own.field.path(start, end) if end else default


class SearchPlanner(Planner):
  def __init__(self, game, depth = 2, feature_weights = {}, book = None):
    Planner.__init__(self, game)

    self.depth = depth
    self.feature_weights = feature_weights
    self.book = book

  def placements(self):
    placements = []
    depth = self.depth

    for i in range(depth):
      if i == 0:
        placements.append((self.game.current_placement,))
      elif i == 1:
        placements.append((self.game.next_placement,))
      else:
        placements.append(self.game.further_placements)
    return placements

  def check_book(self):
    if self.book is None:
      return None
    return self.book.find(self.game.own.field,
                          self.game.current_placement.piece,
                          self.game.next_placement.piece)


  def move(self):
    weights = self.feature_weights
    placements = self.placements()
    beam = 1

    move = self.check_book()
    if move:
      logger().info('move: %s (book)', move)
      return move

    if self.time < 2000:
      placements = placements[:1]
      logger().warning('round %s: truncating depth (time: %s)',
                       self.game.round, self.time)
    elif self.time < 4000:
      beam = 0.65 - 0.4 * (self.time < 3000)
      logger().warning('round %s: truncating width (time: %s)',
                       self.game.round, self.time)

    args = (self.game, placements, weights)
    move, score = search.expectimax(*args, beam = beam)

    logger().info('move: %s (%s)', move, score)
    return move
