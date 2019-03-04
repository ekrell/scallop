# This script stores fitness functions for scoring the grid elements
# of a grid of scientific entities. 

import yaml
from math import ceil

def initLogbook (tWeight, entities, weights, colorcodes, geotiffs,
                 weightDensity, weightSpeed, weightAcc,
                 defaultMargin, defaultRoam):

	# The 'logbook' is a data storage for the analyst.
	# The analyst may reference this log when determining 
	# "interestingness"

	# Guide:
	# 'entities' : A list of N entities, where the index is the ID and the value is a string name.
	#     EX: Entity '1' has name "spartina" ---> entities[1] == "spartina"
	# 'weights' : A list of N weights of the interest level of each entity.
	#     EX: Entity '1' has weight '1' ---> weights[1] == 1

	# To deal with 0-based counting, the '0'th entity is 'No entity'. Has weight of 0.

	logbook = { 'tWeight'       : 0,
	            'entities'      : [], 
	            'weights'       : [],
	            'weightDensity' : weightDensity,
	            'weightSpeed'   : weightSpeed,
	            'weightAcc'     : weightAcc,
	            'colorcodes'    : {},
	            'geotiffs'      : [],
                    'defaultMargin' : defaultMargin,
                    'defaultRoam'   : defaultRoam,
	}

	logbook['tWeight'] = tWeight

	# Set 0th entry as 'No entity' with weight == 0
	logbook['entities'].append("NULL")
	logbook['weights'].append(0)
	logbook['geotiffs'].append(None)

	logbook['colorcodes'] = colorcodes
	logbook['colorcodes'][255] = 0

	for e in entities:
		logbook['entities'].append(e)
	for w in weights:
		logbook['weights'].append(w)
	for g in geotiffs:
		logbook['geotiffs'].append(g)

	return logbook

def initLogbookByFile (logbookFile):

	if logbookFile == "None":
		logbook = None
		return logbook

	with open(logbookFile) as f:
		config = yaml.safe_load(f)

	tWeight      = config['logbook']['tWeight']
	entities      = config['logbook']['entities']
	weights       = config['logbook']['weights']
	colorcodes    = config['logbook']['colorcodes']
	geotiffs      = config['logbook']['geotiffs']
	weightDensity = config['logbook']['weightDensity']
	weightSpeed   = config['logbook']['weightSpeed']
	weightAcc     = config['logbook']['weightAcc']
        defaultMargin = config['logbook']['targetDefaults']['margin']
        defaultRoam   = config['logbook']['targetDefaults']['roam']

	logbook = initLogbook (tWeight, entities, weights, colorcodes,
	    geotiffs, weightDensity, weightSpeed, weightAcc,
            defaultMargin, defaultRoam)

	return logbook

def logbook2html (logbook):
	from yattag import Doc
	import json
	import pandas as pd

	doc, tag, text, = Doc ().tagtext ()

	# Non-tabular
	with tag ('p'):
		text ('Global weight of target influence: ' + str (logbook['tWeight'])) 


	# Tabular
	colorcodes = [logbook['colorcodes'].keys()[logbook['colorcodes'].values().index(e)] for e in range (0,len(logbook['entities']))]
	entities_tbl = pd.DataFrame (
		{'NAME': logbook['entities'],
		 'WEIGHT': logbook['weights'],
		 'COLOR_CODE': colorcodes, 
		})
	entities_tbl.columns.name = 'ENTITY'

	entities_html = entities_tbl.to_html ()
	
	doc.asis (entities_html)

	return doc.getvalue ()


def calcCellReward (grid, row, col, logbook, fitfun = "rewardEntity"):

	# NOTE: Currently, applies a single fitness function using 'fitfun' param.
	#    But what if a list of fitness functions to apply sequentially was provided?

	#####################
	# Fitness Functions #
	#####################

	def dummy ():
		# Cell fitness function: 'dummy'.
		# For testing. Always returns 0.
		return 0

	def rewardEntity (e, logbook):
		# Cell fitness function: 'rewardEntity'
		# Cell's reward is based on the cell's entity value.
		# Looks up the weight of the entity found (if any) in logbook.
		r = logbook['weights'][e]
		return r

	def rewardSystem(e, densities, speeds, accs, logbook):
		r = rewardEntity(e, logbook)
		for ei in range(1, len(logbook["entities"])):
			r = r + abs(densities[ei - 1]) * logbook["weightDensity"] * logbook["weights"][ei]
			r = r + abs(speeds   [ei - 1]) * logbook["weightSpeed"]   * logbook["weights"][ei]
			r = r + abs(accs     [ei - 1]) * logbook["weightAcc"]     * logbook["weights"][ei]
		return r

	# Init reward to 1
	r = 0
	# Get cell value
	c = grid[row][col]
	# convert cell value to entity
	e = logbook['colorcodes'][c]

	#
	densities = [0 for e in range(len(logbook["entities"]))]
	speeds    = [0 for e in range(len(logbook["entities"]))]
	accs      = [0 for e in range(len(logbook["entities"]))]

	for ei in range(1, len(logbook["entities"])):
		densities[ei] = logbook["grids"][ei - 1]["density"][row][col]
		speeds[ei]    = logbook["grids"][ei - 1]["speed"][row][col]
		accs[ei]      = logbook["grids"][ei - 1]["acc"][row][col]

	# Apply selected fitness function
	if fitfun == "dummy":
		r = dummy ()	
	elif fitfun == "rewardEntity":
		r = rewardEntity (e, logbook)
	elif fitfun == "rewardSystem":
		r = rewardSystem(e, densities, speeds, accs, logbook)
	return r


def getRewardAlongLine (x0, y0, x1, y1, grid, xLimit, yLimit, logbook):
	from bresenham import bresenham

	reward = 0
	x0 = int (ceil (x0))
	y0 = int (ceil (y0))
	x1 = int (ceil (x1))
	y1 = int (ceil (y1))
	
	# Get list of cells that are intersected by line segment
	b = list(bresenham(x0, y0, x1, y1))
	# Iterate over those cells
	for p in b:
		reward = reward + calcCellReward (grid, p[0], p[1], logbook, fitfun = "rewardSystem")

	return reward
		


def rewardLine (path, xStart, yStart, xStop, yStop,
	xLimit, yLimit, gridTargets, logbook):
	
	reward = { 'total' : 0.0, 'segments' : [] }
	
	if gridTargets is None:
		return reward

	xi = path[0]
	yi = path[0]
	xj = None
	yj = None

	# First segment is between start location and first waypoint
	reward['segments'].append (getRewardAlongLine (xStart, yStart, xi, yi,
		gridTargets, xLimit, yLimit, logbook))

	# Loop over interior waypoint segments
	count = 2
	iterations = len (path) / 2 - 1
	for i in range (0, iterations):
		xj = path[count]
		count = count + 1
		yj = path[count]
		count = count + 1

		reward['segments'].append (getRewardAlongLine (xi, yi, xj, yj,
			gridTargets, xLimit, yLimit, logbook))

		xi = xj
		yi = yj

	# Final segment is between last waypoint and goal location
	reward['segments'].append (getRewardAlongLine (xj, yj, xStop, yStop,
		gridTargets, xLimit, yLimit, logbook))

	reward['total'] = sum (reward['segments'])

	return reward



