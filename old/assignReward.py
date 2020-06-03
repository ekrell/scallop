#!/usr/bin/python

'''
Creates a grid of reward based on initial reward and statistics of
reward within each subgrid region.
'''
from osgeo import gdal
import os
import sys
import pickle
import numpy as np
import pandas as pd
from path import Path
import re
from statistics import median
import entityFitness
from PIL import Image

def parseOption ():
	from optparse import OptionParser
	parser = OptionParser ()
        # Number of subgrids to divide region into
        parser.add_option('-n', "--num_subgrids", dest="num_subgrids",
                help = "Number of subgrids to divide region", default = 100)
        # Path to entity book
        parser.add_option('-e', "--entitybook", dest="entitybook",
                help = "Path to entity book (YaML)")
        # The geotiff that represents the region under study.
        # Used to tie correct coordinates, etc to entities.
	parser.add_option('-r', "--region", dest='regionFile',
		help = "Base region geotiff", metavar='REGION_FILE')
        parser.add_option ('-o', "--outdirectory", dest='outdirectory',
                help = "Output directory")
        parser.add_option ('-l', "--outprefix", dest="outprefix",
                help = "Prefix for naming output files")
        # Directory where images representing a single state of entities are stored.
	parser.add_option ('-d', "--directory", dest='directory',
		help = "Directory of image iterations", metavar='DIR')
        # Prefix to select image iterations in directory
        parser.add_option ('-p', "--prefix", dest='prefix',
                help = "Prefix to select image iterations in directory",
                metavar='DIR', default = "")
        # Directory where results of this program are stored, so that subsequent executions can do less work.
	parser.add_option ('-c', "--cachedirectory", dest='cachedirectory', default = "",
		help = "Use this directory where existing subgrids are stored", metavar='CACHE')
	return parser.parse_args ()

def initMapByImage (filename, entityBook, nSubgrids):
	from PIL import Image
	im = Image.open (filename)
	im = im.convert ('RGB')
	grid = np.empty (shape = [im.size[0], im.size[1]])
	points = [[] for e in range(len(entityBook["entities"]))]

	for row in range (im.size[0]):
		for col in range (im.size[1]):
			p = im.getpixel ( (row, col) )
                        eidx = entityBook["colorcodes"][p[0]]
                        grid[row][col] = eidx
                        points[eidx].append([row, col])


        rowpadding = 0
        colpadding = 0
        while ((im.size[0] + rowpadding) % nSubgrids != 0):
            rowpadding = rowpadding + 1
        while ((im.size[1] + colpadding) % nSubgrids != 0):
            colpadding = colpadding + 1

	grid = np.pad(grid, [(0, rowpadding), (0, colpadding)],
                      mode='constant', constant_values=0)
	return grid, points, rowpadding, colpadding

##def drawGrids (filename, nSubgrids = 10):
##	import pylab as plt
##	from PIL import Image, ImageDraw
##
##	# Load image
##	img = Image.open(filename)
##	width, height = img.size
##
##	# Draw lines
##	draw = ImageDraw.Draw (img)
##	y_start = 0
##	y_end = img.height
##	step_size = int (img.width / nSubgrids)
##
##	for x in range (0, img.width, step_size):
##		line = ((x, y_start), (x, y_end))
##		draw.line (line, fill = 129)
##
##	x_start = 0
##	x_end = img.width
##
##	for y in range (0, img.height, step_size):
##		line = ((x_start, y), (x_end, y))
##		draw.line (line, fill = 129)
##
##	img.save ('grid.png')
##
##	del draw



def getRegionStats (region, neighbors, nEntities):
	stat = { 'density'     : None,
	         'rdensity'    : None,
	         'internality' : None }
	density     = []
	rdensity    = []
	internality = []
	size = region.shape[0] * region.shape[1]
	for e in range (1, nEntities + 1):
		presence = float ((e == region).sum())
		d = presence / size
		density.append (d)
		rdensity.append (None)
		internality.append (None)
	stat['density'] = density
	stat['rdensity'] = rdensity
	stat['internality'] = internality
	return stat


def blockshaped(arr, nrows, ncols):
    """
    SOURCE: https://stackoverflow.com/a/16873755

    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape

    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    SOURCE: https://stackoverflow.com/a/16873755

    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def iteration (filename, outTag, nSubgrids, nEntities, entityBook):
	# Show the grid lines
	# drawGrids (filename, pow(nSubgrids, 0.5))


	# Get image as a grid
	# Where numbers refer to entity presence
	# and '0' means absence of any entity
	(grid, points, rowpadding, colpadding) = \
                initMapByImage (filename, entityBook, nSubgrids)

	nYgrids = grid.shape[0] / nSubgrids
	nXgrids = grid.shape[1] / nSubgrids

	# Allocate a grid to track the subgrids
	subtable = np.empty (shape = [nYgrids, nXgrids])
	subtable = np.array ([[dict() for x in range (0, nXgrids)] for y in range (0, nYgrids)])

	# Split into subgrids
	gridSplit = blockshaped (grid, nSubgrids, nSubgrids)
	gridStats = []

	for g in gridSplit:
		gridStats.append (getRegionStats (g, None,
                    len(entityBook["entities"])))

	count = 0
	for row in range (0, nYgrids):
		for col in range (0, nXgrids):
			sub = { 'grid' : gridSplit[count],
			        'stat' : gridStats[count] }
			subtable[row][col] = sub
			count = count + 1

	return subtable


def getSampleImageFile(directory, prefix):
	p = Path(directory)
	imageFile = p.files(prefix + '*.png')[0]
	return imageFile


def getSubgridsByDirectory(directory, prefix, outdirectory, outprefix,
                           nSubgrids, nEntities, entityBook):
	p = Path(directory)
	images = []
	for f in p.files(prefix + '*.png'):
		images.append (f)

	subtables = []

	for i in images:
		m = re.search('[0-9]*.png', i)
		m = re.search('[0-9]*', m.group ())
		idx = int(m.group ())
		subtables.append(iteration(i, idx, nSubgrids, nEntities, entityBook))

		# Save the table of subgrids as a python object
		pickle.dump (subtables[len (subtables) - 1],
                        open (outdirectory + "/" + outprefix + "_gridstats_" \
                                + str (idx) + ".pickle", "wb"))

	return images, subtables

def loadSubgridsByDirectory(directory, outprefix):
	p = Path (directory)
	pickles = []
	for f in p.files (pattern= outprefix + '_gridstats_*.pickle'):
		pickles.append (f)

	nTables = len (pickles)

	subTables = []

	# Sort by iteration number.
	# Note that this is __dependent__ on the filename pattern..
	# Example: 'gridstats/EXP_R1_S2/gridstats_1.pickle'
	pickles.sort(key = lambda x:int(x.split ('_')[-1].split('.')[0]))

	for p in pickles:
		subTables.append (pickle.load (open (p, "rb")))

	return subTables

def getResultGrids(results, grid2img, nEntities):

	entity2grids = [ { "densityGrid" : None,
	                   "speedGrid"   : None,
	                   "accGrid"     : None,
	                  } for e in range(nEntities)]

        for e in range(nEntities):
		entity2grids[e]["densityGrid"] = np.array([[0.0 for col in range(grid2img.shape[1])] \
			for row in range(grid2img.shape[0])])
		entity2grids[e]["speedGrid"]   = np.array([[0.0 for col in range(grid2img.shape[1])] \
			for row in range(grid2img.shape[0])])
		entity2grids[e]["accGrid"]     = np.array([[0.0 for col in range(grid2img.shape[1])] \
			for row in range(grid2img.shape[0])])

		idx = 0
		for row in range(len(results)):
			for col in range(len(results[0])):
				entity2grids[e]["densityGrid"][grid2img == idx] = \
                                    100 * results[row][col][e]["results"]["mean"]
				entity2grids[e]["speedGrid"]  [grid2img == idx] = \
                                    100 * results[row][col][e]["results"]["speed"]
				entity2grids[e]["accGrid"]    [grid2img == idx] = \
                                    100 * results[row][col][e]["results"]["acc"]
				idx = idx + 1
	return entity2grids

def analyzeSequence (subtables, nEntities):

	# Look at first table to get dimensions
	nYgrids = subtables[0].shape[0]
	nXgrids = subtables[0].shape[1]

        # Initialize empty results table
	results = np.array ([[ [dict( { 'densities' : [] } ) for e in range (0, nEntities  )]
                  for x in range (0, nXgrids)] for y in range (0, nYgrids)])

	res = { 'mean'   : None,
	        'min'    : None,
	        'max'    : None,
	        'median' : None,
	        'start'  : None,
	        'stop'   : None,
	        'diff'   : None,
	        'speed'  : None,
		'acc'    : None,
 }


	# Load the analysis grid with data points
	for st in subtables:
		for row in range (0, nYgrids):
			for col in range (0, nXgrids):
				for e in range (nEntities):
					results[row][col][e]['densities'].append (st[row][col]['stat']['density'][e])

	# Analyze those data points
	for row in range (0, nYgrids):
		for col in range (0, nXgrids):
			for e in range (0, nEntities):
                                # Mean density
				res['mean']   = float ( sum (results[row][col][e]['densities']) / len (results[row][col][e]['densities']) )
                                # Min density
				res['min']    = min (results[row][col][e]['densities'])
                                # Max density
				res['max']    = max (results[row][col][e]['densities'])
                                # Median density
				res['median'] = median (results[row][col][e]['densities'])
                                # First density value
				res['start']  = results[row][col][e]['densities'][0]
                                # Last density value
				res['stop']   = results[row][col][e]['densities'][-1]
                                # Change in densities
				res['diff']   = res['stop'] - res['start']

				# Get average speed, acceleration
				prevDensity   = res["start"]
				diffs = []
				for density in results[row][col][e]['densities'][1:]:
					diffs.append(density - prevDensity)
					prevDensity = density
				# Density avg speed
                                res["speed"]  = sum(diffs) / len(diffs)
				accs = []
				prevDiff = diffs[0]
				for diff in diffs[1:]:
					accs.append(diff - prevDiff)
					prevDiff = diff
				# Density avg acceleration
				res["acc"]    = sum(accs) / len(accs)


				results[row][col][e]['results'] = res.copy ()

	return results


def getGrid2img(subtables, results, imgfile, rowpadding, colpadding):

	# Creates a mapping between the original 2D map
	# and the subgrids.
	# The value of the map's cell indexes the
	# corresponding subgrid result

	img = Image.open (imgfile)
        h = img.size[0] + rowpadding
	w = img.size[1] + colpadding

	table = subtables[0].copy()

	gridSplit = []
	idx = 0
	for row in range(len(results)):
		for col in range(len(results[0])):
			gridSplit.append(table[row][col]["grid"].copy())
			gridSplit[idx][:,:] = idx
			idx = idx + 1
	gridSplit = np.array(gridSplit)
	grid2img = unblockshaped(gridSplit, h, w)

        rowfix = -1 * rowpadding
        colfix = -1 * colpadding
        if (rowfix < 0 and colfix < 0):
	    grid2img = grid2img[:rowfix, :colfix]
        elif (rowfix < 0):
            grid2img = grid2img[:rowfix, :]
        else:
            grid2img = grid2img[:, :colfix]

	return grid2img

def getResultGeotiffs(entity2grids, entityGeotiffFilenames, regionFile):
	bands = ["densityGrid", "speedGrid", "accGrid"]

	nEntities = len(entity2grids)
	nBands    = len(bands)
        nRows = entity2grids[0]["densityGrid"].shape[0]
	nCols = entity2grids[0]["densityGrid"].shape[1]

	region = gdal.Open(regionFile)

	drv = gdal.GetDriverByName ('GTiff')
	geotransform = region.GetGeoTransform()
        minx = geotransform[0]
	maxy = geotransform[3]
	miny = maxy + geotransform[5] * nRows
	geotransform = [minx, geotransform[1], 0, miny, 0, -geotransform[5]]


	for e in range(nEntities):
		# Init raster for entity
		raster = drv.Create(entityGeotiffFilenames[e],
		         nCols, nRows, nBands,
		         gdal.GDT_Float32)
		raster.SetGeoTransform(region.GetGeoTransform())
		raster.SetProjection(region.GetProjection())

		for b in range(nBands):
			# Fill band data from grid
			band = raster.GetRasterBand(b + 1) # Gdal starts at 1, not 0
			band.SetNoDataValue(0)

			band.WriteArray(entity2grids[e][bands[b]], 0, 0)
		# Close
		raster.FlushCache()


########
# MAIN #
########

(options, args) = parseOption()

nSubgrids = int(options.num_subgrids)

entityBook = entityFitness.initEntitybookByFile(options.entitybook)

nEntities = len(entityBook["entities"])

imageFile = getSampleImageFile(options.directory, options.prefix)
im = Image.open (imageFile)

rowpadding = 0
colpadding = 0
while ((im.size[0] + rowpadding) % nSubgrids != 0):
    rowpadding = rowpadding + 1
while ((im.size[1] + colpadding) % nSubgrids != 0):
    colpadding = colpadding + 1


entityGeotiffFilenames = []
for i in range(nEntities):
    entityGeotiffFilenames.append(options.outdirectory + "/" + options.outprefix \
            + "_analysis_" + entityBook["entities"][i] + ".tiff")

if (options.cachedirectory == ""):
	(files, subtables) = getSubgridsByDirectory(options.directory,
            options.prefix, options.outdirectory, options.outprefix, nSubgrids,
            nEntities, entityBook)
	#for f in files:
		#drawGrids (f)
else:
	subtables = loadSubgridsByDirectory(options.cachedirectory, options.outprefix)

exit(0)

# Calculate information on each grid region
analyzed = analyzeSequence(subtables, nEntities)
# Generate an image where each cell indexes a larger grid region
grid2img = getGrid2img(subtables, analyzed, imageFile, rowpadding, colpadding)
# Generate a set of grids based on the analysis results
entity2grid = getResultGrids(analyzed, grid2img, nEntities)
# Convert the entity2grids structure to a set of GDAL files
# (One file per entity, one band per grid type)
entity2geotiffs = getResultGeotiffs(entity2grid, entityGeotiffFilenames, options.regionFile)
