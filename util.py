#!/usr/bin/env python

from __future__ import print_function, division
import functools


class cached_property(object):
  """ A property that is only computed once per instance and then replaces
      itself with an ordinary attribute. Deleting the attribute resets the
      property.

      Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
      """

  def __init__(self, func):
    self.__doc__ = getattr(func, '__doc__')
    self.func = func

  def __get__(self, obj, cls):
    if obj is None:
        return self
    value = obj.__dict__[self.func.__name__] = self.func(obj)
    return value


class memoized(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated.
   """

   def __init__(self, func):
      self.func = func
      self.cache = {}

   def __call__(self, *args):
      try:
         return self.cache[args]
      except KeyError:
         value = self.func(*args)
         self.cache[args] = value
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args)

   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__

   def __get__(self, obj, objtype):
      """Support instance methods."""
      return functools.partial(self.__call__, obj)


class Nop(object):
  def __bool__(self): return False
  def __copy__(self): return self
  def __deepcopy__(self, _): return self
  def __getattr__(self, _): return self
  def __call__(self, *args, **kwargs): return self

nop = Nop()
