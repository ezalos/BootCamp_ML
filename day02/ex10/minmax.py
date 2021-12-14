import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Minmax():
	def __init__(self):
		self.min = 0.
		self.max = 0.

	def fit(self, X):
		self.min = X.min(axis=0)
		self.max = X.max(axis=0)
		return self

	def apply(self, X):
		e = 1e-20
		mnmx = (X - self.min) / (self.max - self.min + e)
		return mnmx

	def unapply(self, X):
		e = 1e-20
		return (X * (self.max - self.min + e)) + self.min