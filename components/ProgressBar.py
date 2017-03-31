import sys
from time import sleep


class ProgressBar:
	def __init__(self, width, total, title, newline=False):
		self.total = total
		self.curr = 0
		self.width = width
		self.title = title
		self.lastline = ""
		self.newline = newline
		self.update(0)

	def update(self, val):
		if not self.newline:
			sys.stdout.write("\r")
		line = self.title + " ["
		progress = val*self.width/self.total
		line += "=" * progress
		line += " " * (self.width - progress) + "] " + str(progress * 100 / self.width) + "% : " + str(val) + "/" + str(self.total)
		sys.stdout.write(line)
		sys.stdout.flush()
		self.lastline = line

