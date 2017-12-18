#!/usr/bin/env python

from __future__ import print_function, division
from evaluate import Evaluation

import math
import heapq

import logging
def logger():
  return logging.getLogger(__name__)


def expectimax(game, placements, feature_weights, beam = 1, return_all = False):
  eval = Evaluation(game, feature_weights)
  if not placements:
    return None, eval.value()

  def _expectimax(game, placements):
    if not placements:
      return eval.value()
    value = 0
    for p in placements[0]:
      best = float('-inf')
      moves = game.own.field.moves(p)
      if moves and game.own.skips:
        moves.append(None)
      for m in moves:
        eval.update(m, *game.own.move(m))
        v = _expectimax(game, placements[1:])
        eval.rollback(*game.own.undo())
        if v > best:
          best = v
      value += best
    return value / len(placements[0])

  best = None, float('-inf')
  all = [] if return_all else None

  moves = game.own.field.moves(placements[0][0])
  if moves and game.own.skips:
    moves.append(None)

  if beam < 1 and len(placements) > 1:
    def _snap_eval(m):
      eval.update(m, *game.own.move(m))
      v = eval.value()
      eval.rollback(*game.own.undo())
      return v
    num_beam = int(math.ceil(beam * len(moves)))
    moves = heapq.nlargest(num_beam, moves, key=_snap_eval)

  for m in moves:
    eval.update(m, *game.own.move(m))
    v = _expectimax(game, placements[1:])
    eval.rollback(*game.own.undo())
    if v > best[1]:
      best = m, v

    if all is not None:
      all.append((m, v))

  return (best, all) if return_all else best