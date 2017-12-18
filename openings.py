#!/usr/bin/env python

from __future__ import print_function, division
from game import Game, Placement, pieces, letter_pieces, piece_letters
from collections import deque
from copy import copy

import numpy as np
import planner

class Book(dict):
  def prepare(self, planner, depth = 1):
    for piece in pieces:
      self._prepare(planner, piece, depth)

  def _prepare(self, planner, piece, depth):
    if depth <= 0:
      return

    field = planner.game.own.field
    key0 = field.cells[field.ceiling:field.real_height,:].tostring()
    # key0 = np.packbits(planner.game.own.field.cells.view(np.uint8)).tostring()

    if not key0 in self:
      self[key0] = {}

    for next_piece in pieces:
      planner.game.piece = piece
      planner.game.next_piece = next_piece

      key1 = (letter_pieces[piece], letter_pieces[next_piece])

      if not key1 in self[key0]:
        move = planner.move()
        letter = letter_pieces[move.piece]
        rotation = move.rotation
        row = move.row
        col = move.col

        self[key0][key1] = (letter, rotation, row, col)
        print('({}, {}): {},'.format(key0, key1, self[key0][key1]))
      else:
        letter, rotation, row, col = self[key0][key1]
        move = Placement(piece_letters[letter], rotation, row, col)

      if not move:
        continue

      planner.game.own.move(move)
      self._prepare(planner, next_piece, depth - 1)
      planner.game.own.undo()

  def find(self, field, piece, next_piece):
    try:
      key0 = field.cells[field.ceiling:field.real_height,:].tostring()
      key1 = (letter_pieces[piece], letter_pieces[next_piece])
      letter, rotation, row, col = self[key0][key1]
      return Placement(piece_letters[letter], rotation, row, col)
    except KeyError:
      return None
