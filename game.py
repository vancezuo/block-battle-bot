#!/usr/bin/env python

from __future__ import print_function, division
from collections import deque
from util import cached_property, memoized, nop
from copy import copy

import numpy as np
import random

import sys
if sys.version_info[0] < 3:
    range = xrange

class Cell():
  empty = 0
  shape = 1
  block = 2
  solid = 3


class Action():
  left = 'left'
  right = 'right'
  turn_left = 'turnleft'
  turn_right = 'turnright'
  down = 'down'
  up = 'up'


class Points():
  line = [0, 0, 3, 6, 10]
  tspin = [0, 5, 10]
  perfect = 18


class Piece(object):
  def __init__(self, value, name):
    self.value = value
    self.name = 'Piece.{}'.format(name)

  def __repr__(self):
    return self.name

  def __getitem__(self, index):
    return self.value[index]

  @cached_property
  def indexes(self):
    return range(0, len(self.value))

  @cached_property
  def max_height(self):
    return 1 + max(max(x[0]) for x in self.value)

  @cached_property
  def max_width(self):
    return 1 + max(max(x[1]) for x in self.value)

  @cached_property
  def num_rotations(self):
    return len(self.value)

  @cached_property
  def offsets(self):
    return self.value

  @cached_property
  def pairs(self):
    return tuple(tuple(zip(*x)) for x in self.value)

  @cached_property
  def rows(self):
    return tuple(x[0] for x in self.value)

  @cached_property
  def cols(self):
    return tuple(x[1] for x in self.value)

  @cached_property
  def heights(self):
    return tuple(1 + max(x[0]) for x in self.value)

  @cached_property
  def widths(self):
    return tuple(1 + max(x[1]) for x in self.value)

  @staticmethod
  def tspin_offsets():
    return ((0, 0, 2, 2), (0, 2, 0, 2))

Piece.L = Piece([((0, 1, 1, 1), (2, 0, 1, 2)), ((0, 1, 2, 2), (1, 1, 1, 2)),
                 ((1, 1, 1, 2), (0, 1, 2, 0)), ((0, 0, 1, 2), (0, 1, 1, 1))],
                 'L')
Piece.O = Piece([((0, 0, 1, 1), (0, 1, 0, 1))],
                 'O')
Piece.I = Piece([((1, 1, 1, 1), (0, 1, 2, 3)), ((0, 1, 2, 3), (2, 2, 2, 2))],
                 'I')
Piece.J = Piece([((0, 1, 1, 1), (0, 0, 1, 2)), ((0, 0, 1, 2), (1, 2, 1, 1)),
                 ((1, 1, 1, 2), (0, 1, 2, 2)), ((0, 1, 2, 2), (1, 1, 0, 1))],
                 'J')
Piece.S = Piece([((0, 0, 1, 1), (1, 2, 0, 1)), ((0, 1, 1, 2), (1, 1, 2, 2))],
                 'S')
Piece.T = Piece([((0, 1, 1, 1), (1, 0, 1, 2)), ((0, 1, 1, 2), (1, 1, 2, 1)),
                 ((1, 1, 1, 2), (0, 1, 2, 1)), ((0, 1, 1, 2), (1, 0, 1, 1))],
                 'T')
Piece.Z = Piece([((0, 0, 1, 1), (0, 1, 1, 2)), ((0, 1, 1, 2), (2, 1, 2, 1))],
                 'Z')

pieces = [Piece.L, Piece.O, Piece.I, Piece.J, Piece.S, Piece.T, Piece.Z]
# pieces = [Piece.O, Piece.S, Piece.T, Piece.Z]
# pieces = [Piece.O, Piece.I]
# pieces = [Piece.I]

piece_letters = {
  'L': Piece.L,
  'O': Piece.O,
  'I': Piece.I,
  'J': Piece.J,
  'S': Piece.S,
  'T': Piece.T,
  'Z': Piece.Z
}

letter_pieces = {
  Piece.L: 'L',
  Piece.O: 'O',
  Piece.I: 'I',
  Piece.J: 'J',
  Piece.S: 'S',
  Piece.T: 'T',
  Piece.Z: 'Z'
}


class Placement(object):
  cache = {}

  def __new__(cls, piece, rotation, row, col, *args, **kwargs):
    key = (piece, rotation, row, col)
    return Placement.cache.setdefault(key, super(Placement, cls).__new__(cls))

  def __init__(self, piece, rotation, row, col):
    self.piece = piece
    self.rotation = rotation
    self.row = row
    self.col = col

  def __repr__(self):
    args = type(self).__name__, self.piece, self.rotation, self.row, self.col
    return '{}(piece={}, rotation={}, row={}, col={})'.format(*args)

  def _replace(self, piece = None, rotation = None, row = None, col = None):
    if piece is None:    piece = self.piece
    if rotation is None: rotation = self.rotation
    if row is None:      row = self.row
    if col is None:      col = self.col
    return Placement(piece, rotation, row, col)

  @property
  def to_tuple(self):
    return (self.piece, self.rotation, self.row, self.col)

  @cached_property
  def left(self):
    return self._replace(col = self.col - 1)

  @cached_property
  def right(self):
    return self._replace(col = self.col + 1)

  @cached_property
  def up(self):
    return self._replace(row = self.row - 1)

  @cached_property
  def down(self):
    return self._replace(row = self.row + 1)

  @cached_property
  def turn_left(self):
    r = self.rotation - 1
    return self._replace(rotation = r) if r >= 0 else None

  @cached_property
  def turn_right(self):
    r = self.rotation + 1
    l = self.piece.num_rotations
    return self._replace(rotation = r) if r < l else None

  @cached_property
  def rows(self):
    return tuple(r + self.row for r in self.piece.rows[self.rotation])

  @cached_property
  def cols(self):
    return tuple(c + self.col for c in self.piece.cols[self.rotation])

  @cached_property
  def row_bounds(self):
    rows = self.rows
    return (min(rows), max(rows) + 1)

  @cached_property
  def col_bounds(self):
    cols = self.cols
    return (min(cols), max(cols) + 1)

  @cached_property
  def offsets(self):
    return (self.rows, self.cols)

  @cached_property
  def pairs(self):
    return tuple((r + self.row, c + self.col)
                 for r, c in self.piece.pairs[self.rotation])

  @cached_property
  def nonnegative_offsets(self):
    return tuple(zip(*(x for x in self.pairs if x[0] >= 0))) # and x[1] >= 0

  @cached_property
  def tspin_offsets(self):
    offsets = Piece.tspin_offsets()
    rows = tuple(r + self.row for r in offsets[0])
    cols = tuple(c + self.col for c in offsets[1])
    return (rows, cols)

  @memoized
  def is_inside(self, row_bounds, col_bounds):
    return self.is_inside_rows(row_bounds) and self.is_inside_cols(col_bounds)

  @memoized
  def is_inside_rows(self, row_bounds):
    if row_bounds is None:
      return True
    return all(row_bounds[0] <= r < row_bounds[1] for r in self.rows)

  @memoized
  def is_inside_cols(self, col_bounds):
    if col_bounds is None:
      return True
    return all(col_bounds[0] <= c < col_bounds[1] for c in self.cols)

  @cached_property
  def moves(self):
    actions = ['left', 'right', 'turnleft', 'turnright', 'down']
    placements = [self.left, self.right, self.turn_left,
                  self.turn_right, self.down]
    return tuple((a, p) for a, p in zip(actions, placements) if p)

  @cached_property
  def backward_moves(self):
    actions = ['up', 'turnleft', 'turnright', 'left', 'right']
    placements = [self.up, self.turn_left, self.turn_right,
                  self.left, self.right]
    return tuple((a, p) for a, p in zip(actions, placements) if p)

  def generate():
    while True:
      yield Placement(random.choice(pieces), 0, -1, 4)


class Field(object):
  move_cache = {}

  def __init__(self, width = 10, height = 20, cells = None, history = True):
    self.width = width
    self.height = height

    self.reset_cells(cells)
    self.reset_ceiling()
    self.reset_heights()
    self.reset_solid_lines()

    self.history = [] if history else nop

  def __repr__(self):
    args = (type(self).__name__, self.width, self.height,
            self.cells, self.history)
    return '<{}: width={}, height={}, cells={}, history={}>'.format(*args)

  def blank_cells(self):
    return np.zeros((self.height, self.width), dtype = bool)

  def reset_cells(self, cells):
    if cells is None:
      self.cells = self.blank_cells()
    else:
      self.cells = np.array(cells) >= Cell.block

  def reset_ceiling(self):
    for i in range(self.height):
      if np.any(self.cells[i,:]):
        break
    self.ceiling = i

  def reset_solid_lines(self):
    for i in range(self.height, 0, -1):
      if not np.all(self.cells[i - 1,:]):
        break
    self.solid_lines = self.height - i

  def reset_heights(self):
    self.heights = self.calculate_heights()

  def calculate_heights(self, slice = slice(None,None,None)):
    if self.height == self.ceiling:
      return np.zeros((self.width,), dtype=np.int8)[slice]
    arr = self.cells[self.ceiling:,slice]
    top = ~arr[0,:]
    heights = np.argmax(arr, axis=0)
    zeros = np.where(heights == 0)
    heights[zeros] = top[zeros] * arr.shape[0]
    return self.height - self.ceiling - heights

  def reset(self, cells):
    self.reset_cells(cells)
    self.reset_ceiling()
    self.reset_heights()
    self.reset_solid_lines()
    self.history[:] = []

  @property
  def floor(self):
    return self.height - min(self.heights)

  @property
  def second_floor(self):
    return self.height - np.partition(self.heights, 1)[1]

  @property
  def fourth_floor(self):
    return self.height - np.partition(self.heights, 3)[3]

  @property
  def real_height(self):
    return self.height - self.solid_lines

  @memoized
  def can_contain(self, placement):
    return placement.is_inside((-1, self.height), (0, self.width))

  def can_fit(self, placement):
    return (self.can_contain(placement) and
            not np.any(self.cells[placement.nonnegative_offsets]) )

  def can_base(self, placement):
    try:
      return (placement.row >= 0 and
              np.any(self.cells[1:,:][placement.nonnegative_offsets]) )
    except IndexError:
      return True

  def can_place(self, placement):
    return self.can_fit(placement) and self.can_base(placement)

  def ceiling_start(self, placement):
    row = self.ceiling - placement.piece.max_height
    return placement._replace(row = row) if row > placement.row else placement

  def local_moves(self, placement):
    return ((a, p) for a, p in placement.moves if self.can_contain(p))

  def moves(self, placement):
    placement = self.ceiling_start(placement)

    # This was tested -- but found no speedup benefit
    # key = self.heights.tostring()
    # try:
    #   return Field.move_cache[(key, placement)]
    # except KeyError:
    #   pass

    moves = []

    if not self.can_fit(placement):
      return moves

    visited = set([placement])
    queue = deque([placement])
    while queue:
      placement = queue.popleft()
      if self.can_base(placement):
        moves.append(placement)
      for _, neighbor in self.local_moves(placement):
        if neighbor in visited or not self.can_fit(neighbor):
          continue
        visited.add(neighbor)
        queue.append(neighbor)

    # Field.move_cache[(key, placement)] = moves
    return moves

  def drops(self, placement):
    moves = []

    placement = self.ceiling_start(placement)

    for rotation in placement.piece.indexes:
      for col in range(-2, self.width):
        p = placement._replace(rotation = rotation, col = col)
        if not self.can_fit(p):
          continue
        while not self.can_base(p):
          p = p.down
        moves.append(p)

    return moves

  def path(self, start_placement, end_placement):
    if not self.can_fit(start_placement):
      return None

    def extract_path(paths, placement):
      path = ['drop']
      while True:
        parent, action = paths[placement]
        if parent is None:
          break
        placement = parent
        if len(path) <= 1 and action == 'down':
          continue
        path.append(action)
      path.reverse()
      return path

    paths = {start_placement: (None, None)}
    queue = deque([start_placement])
    while queue:
      placement = queue.popleft()
      for action, neighbor in self.local_moves(placement):
        if neighbor in paths or not self.can_fit(neighbor):
          continue
        paths[neighbor] = (placement, action)
        if neighbor == end_placement:
          return extract_path(paths, end_placement)
        queue.append(neighbor)

    return None

  def check_empty(self):
    return self.ceiling == self.height - self.solid_lines

  def check_tspin(self, placement):
    if ( placement.piece is not Piece.T or
         not (0 <= placement.row < self.height - 2 and
              0 <= placement.col < self.width - 2) ):
      return False
    offsets = placement.tspin_offsets
    empties = np.where(self.cells[offsets] == 0)[0]
    if empties.shape[0] != 1:
      return False
    index = empties[0]
    row, col = offsets[0][index], offsets[1][index]
    drow = 1 if index <= 1 else -1
    dcol = -1 if index & 1 else 1
    return not np.all(self.cells[[row, row + drow], [col + dcol, col]])

  def place(self, placement, check = False):
    if check and not self.can_place(placement):
      raise Exception('illegal placement')
    self.cells[placement.offsets] = True
    self.ceiling = min(self.ceiling, placement.row_bounds[0])

  def squash(self, rows):
    num_rows = len(rows)
    keep = [r for r in range(self.height) if r not in rows]

    old_cells = self.cells
    self.cells = self.blank_cells()
    self.cells[num_rows:,:] = old_cells[keep,:]
    self.ceiling += num_rows

  def update_heights(self, placement, filled_rows = None):
    if filled_rows:
      self.heights = self.calculate_heights()
    else:
      col_slice = slice(*placement.col_bounds)
      self.heights = copy(self.heights)
      self.heights[col_slice] = self.calculate_heights(col_slice)

  def move(self, placement, check = False):
    ceiling = self.ceiling # save for history
    heights = self.heights

    self.place(placement, check)

    self.history.append((placement, ceiling, heights, self.cells))

    row_bounds = placement.row_bounds
    filled = np.all(self.cells[slice(*row_bounds),:], axis = 1)
    if filled.any():
      filled_rows = [r for r, c in zip(range(*row_bounds), filled) if c]
      tspin = self.check_tspin(placement)

      self.squash(filled_rows)

      self.update_heights(placement, filled_rows)
      return filled_rows, tspin
    else:
      self.update_heights(placement)
      return (), False

  def drop(self, placement):
    if not self.can_fit(placement):
      return False
    while not self.can_place(placement):
      placement = placement.down
    return self.move(placement)

  def undo(self):
    if not self.history:
      raise Exception('field undo no history')
    placement, self.ceiling, self.heights, self.cells = self.history.pop()
    self.cells[placement.offsets] = False
    return ()

  def push(self, row):
    if self.cells[0,:].any():
      return False
    num_rows = self.height - self.solid_lines
    self.cells[0:num_rows - 1,:] = self.cells[1:num_rows,:]
    self.cells[num_rows - 1,:] = row
    self.ceiling -= 1
    self.reset_solid_lines()
    self.reset_heights()
    return True


class Player(object):
  def __init__(self, name = None, points = 0, combo = 0, width = 10,
               height = 20, cells = None, history = True):
    self.name = name
    self.points = points
    self.combo = combo
    self.skips = 0

    self.field = Field(width, height, cells, history)

    self.history = [] if history else nop

  def __repr__(self):
    template = '<{}: name={}, points={}, combo={}, field={}, history={}>'
    args = (type(self).__name__, self.name, self.points, self.combo,
            self.field, self.history)
    return template.format(*args)

  def update_score(self, filled_rows, tspin = False):
    num_filled = len(filled_rows)
    if num_filled == 0:
      self.combo = 0
      return

    if self.field.check_empty():
      points = Points.perfect
    elif tspin:
      points = Points.tspin[num_filled]
      self.skips += num_filled == 2
    else:
      points = Points.line[num_filled]
      self.skips += num_filled == 4

    self.points += points + self.combo
    self.combo += (points > 0)

  def move(self, placement, check = False):
    skip = placement is None
    self.history.append((self.points, self.combo, self.skips, skip))
    result = self.field.move(placement, check) if not skip else ((), False)
    self.skips -= skip
    self.update_score(*result)
    return result

  def undo(self):
    if not self.history:
      raise Exception('player undo no history')
    self.points, self.combo, self.skips, skip = self.history.pop()
    if not skip:
      self.field.undo()
    return ()


class Game(object):
  def __init__(self, timebank = 10000, time_per_move = 500, own = None,
               opp = None, piece = None, piece_position = (-1, 3),
               next_piece = None, round = 1, solid_rate = 15, garbage_rate = 0):
    self.timebank = timebank
    self.time_per_move = time_per_move

    self.own = own if own else Player('player1')
    self.opp = opp if opp else Player('player2')

    self.piece = piece if piece else random.choice(pieces)
    self.piece_position = piece_position
    self.next_piece = next_piece if next_piece else random.choice(pieces)
    self.round = round

    self.solid_rate = solid_rate
    self.garbage_rate = garbage_rate
    self.garbage_amnt = 1

  def __repr__(self):
    template = ('<{}: timebank={}, time_per_move={}, own={}, opp={}, '
                'piece={}, piece_position={}, next_piece={}, round={}>')
    args = (type(self).__name__, self.timebank, self.time_per_move,
            self.own, self.opp, self.piece, self.piece_position,
            self.next_piece, self.round)
    return template.format(*args)

  @property
  def current_placement(self):
    row, col = self.piece_position
    return Placement(self.piece, 0, row, col)

  @property
  def next_placement(self):
    row, col = self.piece_position
    return Placement(self.next_piece, 0, row + self.pending_solid_row, col)

  @property
  def further_placements(self):
    row, col = self.piece_position
    return tuple(Placement(p, 0, row + self.pending_solid_row, col)
                 for p in pieces)

  @property
  def pending_solid_row(self):
    return self.solid_rate and self.round % self.solid_rate == 0

  @property
  def pending_garbage(self):
    return self.garbage_rate and self.round % self.garbage_rate == 0

  def advance(self, next_piece = None, garbage = None):
    self.piece = self.next_piece
    self.next_piece = next_piece if next_piece else random.choice(pieces)
    self.round += 1
    if self.pending_garbage:
      width = self.own.field.width
      garbage = np.ones((width,), dtype=bool)
      garbage[random.sample(range(width), self.garbage_amnt)] = False
      self.garbage_amnt = 3 - self.garbage_amnt
      if not self.own.field.push(garbage):
        return False
      width = self.own.field.width
      garbage = np.ones((width,), dtype=bool)
      garbage[random.sample(range(width), self.garbage_amnt)] = False
      self.garbage_amnt = 3 - self.garbage_amnt
      if not self.own.field.push(garbage):
        return False
    if self.pending_solid_row and not self.own.field.push(1):
      # row, col = self.piece_position
      # self.piece_position = row + 1, col
      return False
    return True

  @property
  def height_penalty(self):
    return self.piece_position[0] + 1
