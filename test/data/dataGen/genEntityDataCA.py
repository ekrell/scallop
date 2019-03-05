#!/usr/bin/python

"""
Simple CA that generates data with a very simple
scheme using growth, decay rates.
Also supports scripted events.

Warning: several hard-coded specifications throughout code..
"""

__package__ = 'cage'

import cage
import random
import curses
from PIL import Image
from numpy.random import choice
import pandas as pd
import numpy as np
import os

# 0 represents no entity
NSEMPTY = 0

def initPattern (rows = 3, cols = 3, initPoints = []):
    '''Generate a 2D initial starting grid

    Args:
        rows (int): Number of rows of pattern.
            Defaults to 3.
        cols (int): Number of cols of pattern.
            Defaults to 3.
        initPoints (List of tuples): tuple containing:
            (int) row of location to assign 1.
            (col) col of location to assign 1.
            Defaults to empty list.
    '''

    grid = []

    # Initialize grid to zeros
    for r in range (rows):
	grid = grid + [[]]
	for c in range (cols):
	    grid[r] = grid[r] + [0]

    # Place 1 at each point
    for p in initPoints:
	grid[p[0]][p[1]] = 1

    return grid


class BirthRule (cage.Rule):
	"""
	N = number of neighbors
        -----                                    -----
        |1  |                                    |1  |
        |   |  --- (N / 10) chance of birth ---> | 1 |
        |   |      at the middle position        |   |
        -----                                    -----
	"""

	def populate (self, address, entity):
		# Initialize neighboorhood table
		self.table = self.map.states (address)
		self.entity = entity
		self.weight = float (entitiesTable[str (entity)][0])

	def rule (self, address):
		n = float (self.table.count (self.entity))
		probBirth = (n / 10.) * self.weight
		newValue = choice ( [NSEMPTY, self.entity], 1, p = [1 - probBirth, probBirth])[0]
		return newValue

class DeathRule (cage.Rule):
	"""
	N = number of neighbors
        -----                                     -----
        |1  |                                     |1  |
        | 1 |  -- ((9-N)/20) chance of death --> | 0 |
        |   |     at the middle position          |   |
        -----                                     -----
	"""
	def populate (self, address, entity):
		# Initialize neighboorhood table
		self.table = self.map.states (address)
		self.entity = entity
		self.weight = entitiesTable[str (entity)][1]

	def rule (self, address):
		n = float (self.table.count (self.entity))
		probDeath = ((9 - n) / 20) * self.weight
		newValue = choice ( [NSEMPTY, self.entity], 1, p = [probDeath, 1 - probDeath])[0]
		return newValue


class SubjectRule (BirthRule, DeathRule):

	def rule (self, address):

		# Get state of self
		state = self.map.get(address)
		newState = state
		for e in entityTypes:
			if (state == NSEMPTY):
				# Is the cell born?
				BirthRule.populate (self, address, e)
				newState = BirthRule.rule (self, address)
				if (newState != NSEMPTY):
					return newState
			elif (state == e):
				# Does the cell die?
				DeathRule.populate (self, address, e)
				newState = DeathRule.rule (self, address)
				if (newState == 0):
					return newState
		return newState


class SubjectAutomaton (cage.SynchronousAutomaton, SubjectRule):

	def __init__ (self, size):
		cage.SynchronousAutomaton.__init__ (self, cage.MooreMap (size))
		SubjectRule.__init__ (self)

	def update (self):
		cage.SynchronousAutomaton.update (self)

class NaschImagePlayer(cage.Player):

	def __init__(self, width, height, numIterations, directory, prefix):
		assert Image , "WARNING: no Image library loaded"
		cage.Player.__init__(self)
		self.width = width
		self.height = height
		self.image = Image.new('RGB', (width, height), (255, 255, 255))
		self.size = (width, height)
		self.inited = 0
		self.iteration = 1
		self.stop = int(numIterations)
                self.directory = directory
                self.prefix = prefix

	def display(self):
		map = self.automaton.map
		col = (0, 0, 0)
		for x in range (map.width):
			for y in range (map.height):
				val = map.get((x,y))
				if val == NSEMPTY:  #empty cell: white
					col = (255, 255, 255)
				elif val == 0:
					col = (255, 0, 0)
				elif val == 1:
					col = (0, 106, 106)
				elif val == 2:
					col = (129, 50, 234)
				elif val == 3:
					col = (65, 167, 234)
				elif val == 4:
					col = (225, 106, 106)
				elif val == 5:
					 col = (0, 255, 0)
				else:   #invalid: black
					col = (0, 0, 0)
				self.image.putpixel((x, y), col)

	def printImage (self):
                img_temp = self.image.copy()
                self.image = self.image.rotate(-90, expand = True) # Correct orientation
                self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT) # Correct orientation
		self.image.save (self.directory + '/' + self.prefix + '_' + str (self.iteration) + '.png')
                self.image = img_temp.copy()

	def main(self, automaton):
		cage.Player.main(self, automaton)
		assert self.automaton is not None
		assert self.automaton.map.dimension == 2 ###
		while (self.iteration <= self.stop and self.automaton.running()):
		        self.display()
			self.automaton.update()
			currentEvents = checkEvents (self.iteration)
			exeEvents (currentEvents, self.automaton.map)
			self.automaton.between()
			self.printImage ()
			self.iteration = self.iteration + 1
		self.finish()

	def finish(self):
		#self.image.show()
		True == True


def checkEvents (iteration):
	currentEvents = []

	if (len (events) == 0):
		return currentEvents
	i = 0
	event = events[i]
	while (int (event.split (',')[0]) == iteration):
		currentEvents.append (event)
		events.pop (0)
		if (len (events) == 0):
			break
		event = events[i]
	return currentEvents

def exeEvents (events, gridMap):
	for event in events:
		event = event.split (',')
		iteration = event[0]
		command = event[1]
		args = event[2:]

		if (command == "seed"):
			entity = int (args[0])
			col = int (args[1])
			row = int (args[2])
			gridMap.set ((col, row), entity)

		elif (command == "hole"):
			col = int (args[0])
			row = int (args[1])
			radius = int (args[2])
			prob = float (args[3])

			points = radMask ( (col, row), radius, np.random.random ( (gridMap.height-1, gridMap.width-1)))
			for point in points:
				doesDie = choice ( [0, 1], 1, p = [1 - prob, prob])[0]
				if (doesDie == 1):
					gridMap.set (point, 0)

# Utility functions

def radMask (index, radius, array):
	a, b = index
	nx, ny = array.shape
	y, x = np.ogrid[-a:nx-a, -b:ny-b]
	mask = x*x + y*y <= radius * radius
	points = []
	for y in range (0, mask.shape[0]):
		for x in range (0, mask.shape[1]):
			if (mask[y][x] == True):
				points.append ( (y, x) )
	return points


def initEntitiesByFile (filename):
	entities = pd.read_csv (filename)
	return entities

def initMapByFile (filename):
	with open(filename, "r") as file:
    		grid = [[int(x) for x in line.split()] for line in file]
	return grid

def initMapByImage (filename):
	from PIL import Image
	im = Image.open (filename)
	im = im.convert ('RGB')
	grid = [[0 for x in range (im.size[1])] for y in range (im.size[0])]

	for row in range (im.size[0]):
		for col in range (im.size[1]):

			p = im.getpixel ( (row, col) )

			if (p == (0, 106, 106)):
				grid[row][col] = 1

			elif (p == (129, 50, 234)):
				grid[row][col] = 2

			elif (p == (65, 167, 234)):
				grid[row][col] = 3

			elif (p == (225, 106, 106)):
				grid[row][col] = 4
	return grid


def initEventsByFile (filename):
	events = open (filename).read ().splitlines ()
	events = events[:-1]
	return events


entityTypes = None
entitiesTable = None
events = None


def parseOptions():
	from optparse import OptionParser

	# Define options
	parser = OptionParser()
	parser.add_option("-e", "--entities", dest = "entities", metavar = "ENTITIES",
		help = "Path to entities csv file")
	parser.add_option("-s", "--events", dest = "events", metavar = "EVENTS",
		help = "Path to events script csv file")
	parser.add_option("-i", "--iterations", dest = "iterations", metavar = "ITERATIONS",
		help = "Number of iterations")
	parser.add_option("-m", "--map", dest = "map", metavar = "MAP",
		help = "Path to entities map image")
        parser.add_option("-d", "--direction", dest = "directory", metavar = "DIRECTORY",
                help = "Directory to store resulting maps", default = None)
        parser.add_option("-p", "--prefix", dest = "prefix", metavar = "PREFIX",
                help = "File prefix for resulting map file names", default = "")

	 # Get options
	(options, args) = parser.parse_args()

	# Verify correct options
	if options.entities   is None or \
	   options.events     is None or \
	   options.iterations is None or \
	   options.map        is None:
		(options, args) = None, None

	return options, args

def main ():
	#### Parameters
	###numIterations = 50
	###entitiesFile = "experiments/EXP_R1_S2_entities.csv"
	###eventsFile = "experiments/EXP_R1_S2_events.csv"

	#### Using "real" image map (2000 x 1000)
	###gridHeight = 2000
	###gridWidth = 1000
	###mapImageFile = "experiments/scenario1_e1234_t.bmp"
	###grid = initMapByImage (mapImageFile)
	###
	#### Using simple square ascii map (100 x 100)
	####gridHeight = 100
	####gridWidth = 100
	####mapFile = "experiments/sample_map.txt"
	####grid = initMapByFile (mapFile)

        # Parameters
	(options, args) = parseOptions()
	if options is None:
		exit()
	numIterations = options.iterations
	entitiesFile  = options.entities
	eventsFile    = options.events
	mapImageFile  = options.map

	grid = initMapByImage (mapImageFile)
	gridHeight = len(grid)
	gridWidth  = len(grid[0])

	# Initialize entities and existing map
	global entitiesTable
	global entityTypes
	global events

	entitiesTable = initEntitiesByFile (entitiesFile)
	numEntities = entitiesTable.shape[1] - 1 # Number of columns - 1
	print ("Entities Table")
	print (entitiesTable)
	print ("----------")

	entityTypes = entitiesTable.columns.tolist()[1:]
	entityTypes = [int (e) for e in entityTypes]

	events = initEventsByFile (eventsFile)
	print ("Events")
	print (events)
	print ("----------")


	# Build CA infrustructure

        if options.directory is not None:
            directory = options.directory
        else:
            directory = os.getcwd()

	player = NaschImagePlayer (gridWidth, gridHeight, numIterations, directory, options.prefix)
	automaton = SubjectAutomaton (player.size)
	cage.PatternInitializer (grid).initialize (automaton)

	# Begin CA
	print ("Begin automaton")
	player.main (automaton)
	print ("End automaton")

	# Clean CA
	player.done ()

if __name__ == '__main__': main ()
