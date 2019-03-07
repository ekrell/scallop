#!/usr/bin/python

def parseOption():
    parser = OptionParser()
    parser.add_option('-c', "--config",       dest    = "config",
        help = "Config file.",                metavar = "CONFIG")
    parser.add_option('-r', "--regionFile",   dest    = "regionFile",
        help = "Region GDAL file.",           metavar = "REGION_FILE")
    parser.add_option('-e', "--entitiesFile", dest    = "entitiesFile",
        help = "Image (png) of entities.",    metavar = "ENTITIES_FILE")
    parser.add_option('-n', "--numTargets",   dest    = "numTargets",
        help = "Number targets to select.",   metavar = "NUM_TARGETS",  default = 10)
    parser.add_option('-o', "--outTable",     dest    = "outTable",
        help = "Output targets CSV.",         metavar = "OUT_TABLE")
    parser.add_option('-p', "--pickle",       dest    = "pickle",
        help = "Output pickle file.",         metavar = "PICKLE")
    return parser.parse_args()

def getArchiveByGrid (row, col, grid, transform):
    archivePoint = { "row" : row, "col" : col, "lat" : None, "lon" : None, "elem" : None }
    worldPoint = grid2world (archivePoint["row"], archivePoint["col"], transform, len(grid))
    archivePoint["lat"] = worldPoint[0]
    archivePoint["lon"] = worldPoint[1]
    elem = getElemByGrid (archivePoint["row"], archivePoint["col"], grid)
    archivePoint["elem"] = elem
    # A flipped row for when you have the origin at lower left.
    archivePoint["rowFromBottom"] = (-1) * archivePoint["row"] + len(grid)
    return archivePoint

def makeDataDict(config, regionFile, entitiesFile):
    # initialize logbook
    logbook = TargetFitness.initLogbookByFile(config)
    logbook["rasters"] = [None] + [gdal.Open(g) for g in logbook["geotiffs"][1:]]
    logbook["grids"] = \
        [{ "density" : logbook["rasters"][e].GetRasterBand(1).ReadAsArray(),
           "speed"   : logbook["rasters"][e].GetRasterBand(2).ReadAsArray(),
           "acc"     : logbook["rasters"][e].GetRasterBand(3).ReadAsArray() } \
            for e in range(1, len(logbook["rasters"]))]
    logbook["region"]         = gdal.Open(regionFile)
    logbook["grid"]           = logbook["region"].GetRasterBand(1).ReadAsArray()
    logbook["regionEntities"] = gdal.Open(entitiesFile)
    logbook["gridEntities"]   = logbook["regionEntities"].GetRasterBand(1).ReadAsArray()
    return logbook

def makeParamsDict(maxDepth, maxCells, minCells):
    params = { "maxDepth" : maxDepth,
               "maxCells" : maxCells,
               "minCells" : minCells,
    }
    return params

def statNode(node, depth, data, params):
    # Get information about a node
    info = { "numCells"     : node["height"] * node["width"],
             "depth"        : depth,
             "reward"       : None,
             "numObstacles" : data["grid"][node["upper_coords_absolute"]["y"] : \
                                           node["lower_coords_absolute"]["y"],  \
                                           node["upper_coords_absolute"]["x"] : \
                                           node["lower_coords_absolute"]["x"]].sum(),
    }
    return info

def checkBaseCase(node, depth, data, params):
    # See if the current node's properties
    # are such that it has reached the base case
    # for quadgrid), quadgridRecurse
    base = 0

    if params["maxDepth"]  < depth                                      or \
       node["measures"]["numCells"] == node["measures"]["numObstacles"] or \
       (params["maxCells"] > node["measures"]["numCells"]                  \
          and node["measures"]["numObstacles"] == 0)                    or \
       params["minCells"]  > node["measures"]["numCells"]:
        return True
    else:
        return False

def printNodeList(nodes):
    for node in nodes:
        print ("Measures:", node["measures"], "Leaf?", node["leaf"])

def filterNodeList(nodes):
    nodesFiltered = []
    for node in nodes:
        if node["measures"]["numCells"] > node["measures"]["numObstacles"] and \
           node["measures"]["reward"]   > 0:
            nodesFiltered.append(node)
    return nodesFiltered

def rankNodeList(nodes):
    nodesRanked = sorted(nodes,       key=lambda k: k["measures"]['reward'], reverse = True)
    #nodesRanked = sorted(nodesRanked, key=lambda k: k["measures"]['numObstacles'])
    return nodesRanked

def traverseTreeRecurse(node, numNodes, verbose):
    if verbose == True:
        print ("Measures:", node["measures"])
    if len(node["children"]) == 0:
        return numNodes
    else:
        numNodes = numNodes + 4
        for c in node["children"]:
            numNodes = traverseTreeRecurse(c, numNodes, verbose)
    return numNodes

def traverseTree(tree, verbose = True):
    node = tree["root"]
    numNodes = 1
    numNodes = traverseTreeRecurse(node, numNodes, verbose)
    return numNodes

def getLeavesRecurse(node, leaves):
    if len(node["children"]) == 0:
        leaves.append(node)
        return
    for c in node["children"]:
        getLeavesRecurse(c, leaves)

def getLeaves(tree):
    leaves = []
    node = tree["root"]
    getLeavesRecurse(node, leaves)
    return leaves

def quadgridRecurse(parent, depth, data, params):

    # Check base case
    if checkBaseCase(parent, depth, data, params) == True:
        parent["leaf"] = True
        return depth    # Depth at iteration when base cas hit
    else:
        parent["leaf"] = False

    # Increase depth
    depth = depth + 1

    # Initialize 4 empty nodes
    nodes = [copy.deepcopy(nodeTemplate) for i in range(4)]

    # Divide image into four equal quadrants.
    # The four nodes are labelled 0..3 is the following configuration:
    # ---------
    # | 0 | 1 |
    # ---------
    # | 2 | 3 |
    # ---------

    nodes[0]['upper_coords_relative']['x'] =  \
        parent["upper_coords_relative"]["x"]
    nodes[0]['upper_coords_relative']['y'] =  \
        parent["upper_coords_relative"]["y"]
    nodes[0]['lower_coords_relative']['x'] =  \
        parent["upper_coords_relative"]["x"]  + (parent["width"]  / 2)
    nodes[0]['lower_coords_relative']['y'] =  \
        parent["upper_coords_relative"]["y"]  + (parent["height"] / 2)
    nodes[1]['upper_coords_relative']['x'] =  \
        parent["upper_coords_relative"]["x"]  + (parent["width"]  / 2)
    nodes[1]['upper_coords_relative']['y'] =  \
        parent["upper_coords_relative"]["y"]
    nodes[1]['lower_coords_relative']['x'] =  \
        parent["lower_coords_relative"]["x"]
    nodes[1]['lower_coords_relative']['y'] =  \
        parent["upper_coords_relative"]["y"]  + (parent["height"] / 2)
    nodes[2]['upper_coords_relative']['x'] =  \
        parent["upper_coords_relative"]["x"]
    nodes[2]['upper_coords_relative']['y'] =  \
        parent["upper_coords_relative"]["y"]  + (parent["height"] / 2)
    nodes[2]['lower_coords_relative']['x'] =  \
        parent["upper_coords_relative"]["x"]  + (parent["width"]  / 2)
    nodes[2]['lower_coords_relative']['y'] =  \
        parent["lower_coords_relative"]["y"]
    nodes[3]['upper_coords_relative']['x'] =  \
        parent["upper_coords_relative"]["x"]  + (parent["width"]  / 2)
    nodes[3]['upper_coords_relative']['y'] =  \
        parent["upper_coords_relative"]["y"]  + (parent["height"] / 2)
    nodes[3]['lower_coords_relative']['x'] =  \
        parent["lower_coords_relative"]["x"]
    nodes[3]['lower_coords_relative']['y'] =  \
        parent["lower_coords_relative"]["y"]

    nodes[0]['upper_coords_absolute']['x'] =  \
        parent['upper_coords_absolute']['x']
    nodes[0]['upper_coords_absolute']['y'] =  \
        parent['upper_coords_absolute']['y']
    nodes[0]['lower_coords_absolute']['x'] =  \
        parent['upper_coords_absolute']['x']  + (parent["width"]  / 2)
    nodes[0]['lower_coords_absolute']['y'] =  \
        parent['upper_coords_absolute']['y']  + (parent["height"] / 2)
    nodes[1]['upper_coords_absolute']['x'] =  \
        parent['upper_coords_absolute']['x']  + (parent["width"]  / 2)
    nodes[1]['upper_coords_absolute']['y'] =  \
        parent['upper_coords_absolute']['y']
    nodes[1]['lower_coords_absolute']['x'] =  \
        parent['lower_coords_absolute']['x']
    nodes[1]['lower_coords_absolute']['y'] =  \
        parent['upper_coords_absolute']['y']  + (parent["height"] / 2)
    nodes[2]['upper_coords_absolute']['x'] =  \
        parent['upper_coords_absolute']['x']
    nodes[2]['upper_coords_absolute']['y'] =  \
        parent['upper_coords_absolute']['y']  + (parent["height"] / 2)
    nodes[2]['lower_coords_absolute']['x'] =  \
        parent['upper_coords_absolute']['x']  + (parent["width"]  / 2)
    nodes[2]['lower_coords_absolute']['y'] =  \
        parent['lower_coords_absolute']['y']
    nodes[3]['upper_coords_absolute']['x'] =  \
        parent['upper_coords_absolute']['x']  + (parent["width"]  / 2)
    nodes[3]['upper_coords_absolute']['y'] =  \
        parent['upper_coords_absolute']['y']  + (parent["height"] / 2)
    nodes[3]['lower_coords_absolute']['x'] =  \
        parent['lower_coords_absolute']['x']
    nodes[3]['lower_coords_absolute']['y'] =  \
        parent['lower_coords_absolute']['y']

    nodes[0]["height"] = abs(nodes[0]["lower_coords_relative"]["y"] \
                           - nodes[0]["upper_coords_relative"]["y"])
    nodes[1]["height"] = abs(nodes[1]["lower_coords_relative"]["y"] \
                           - nodes[1]["upper_coords_relative"]["y"])
    nodes[2]["height"] = abs(nodes[2]["lower_coords_relative"]["y"] \
                           - nodes[2]["upper_coords_relative"]["y"])
    nodes[3]["height"] = abs(nodes[3]["lower_coords_relative"]["y"] \
                           - nodes[3]["upper_coords_relative"]["y"])
    nodes[0]["width"]  = abs(nodes[0]["lower_coords_relative"]["x"] \
                           - nodes[0]["upper_coords_relative"]["x"])
    nodes[1]["width"]  = abs(nodes[1]["lower_coords_relative"]["x"] \
                           - nodes[1]["upper_coords_relative"]["x"])
    nodes[2]["width"]  = abs(nodes[2]["lower_coords_relative"]["x"] \
                           - nodes[2]["upper_coords_relative"]["x"])
    nodes[3]["width"]  = abs(nodes[3]["lower_coords_relative"]["x"] \
                           - nodes[3]["upper_coords_relative"]["x"])

    # Assign these nodes to the parent as children
    parent["children"] = nodes

    # Stat each node
    depthOuts = [0 for c in parent["children"]]
    for ci in range(len(parent["children"])):
        parent["children"][ci]["measures"] = statNode(parent["children"][ci], depth, data, params)

        depthOuts[ci] = quadgridRecurse(parent["children"][ci], depth, data, params)

    return max(depthOuts)

def quadgrid(tree, data, params):
    height, width = data["grids"][0]["density"].shape

    # Init tree depth
    depth = 0

    # Create head node
    root = copy.deepcopy(nodeTemplate)
    root["upper_coords_absolute"]["y"] = 0
    root["upper_coords_absolute"]["x"] = 0
    root["lower_coords_absolute"]["y"] = height
    root["lower_coords_absolute"]["x"] = width
    root["upper_coords_relative"]["y"] = 0
    root["upper_coords_relative"]["x"] = 0
    root["lower_coords_relative"]["y"] = height
    root["lower_coords_relative"]["x"] = width
    root["height"] =  abs(root["lower_coords_relative"]["y"] - root["upper_coords_relative"]["y"])
    root["width"]  =  abs(root["upper_coords_relative"]["x"] - root["lower_coords_relative"]["x"])
    # Add quality measures
    root["measures"] = statNode(root, depth, data, params)
    # Set as root
    tree["root"] = root

    # Begin recursive quadtree
    depth = quadgridRecurse(tree["root"], depth, data, params)

    return depth


def calcNodeReward(node, data, params):
    reward = 0
    for row in range(node["upper_coords_absolute"]["y"],
                     node["lower_coords_absolute"]["y"]):
        for col in range (node["upper_coords_absolute"]["x"],
                          node["lower_coords_absolute"]["x"]):

            # Get cell entity
            c = int(data["gridEntities"][row][col])
            e = data["colorcodes"][c]
            r = data["weights"][e]
            reward = reward + r
            rew = reward
            # Get cell density, speed, acc
            for e in range(1, len(data["entities"])):
                reward = reward + data["grids"][e - 1]["density"][row][col] \
                                * data["weightDensity"] * data["weights"][e]
                reward = reward + data["grids"][e - 1]["speed"]  [row][col] \
                                * data["weightSpeed"]   * data["weights"][e]
                reward = reward + data["grids"][e - 1]["acc"]    [row][col] \
                                * data["weightAcc"]     * data["weights"][e]
    #print (rew, reward)
    return reward


def getNodeListReward(nodes, data, params):
    # In-place modification of a node attribute
    for node in nodes:
        node["measures"]["reward"] = calcNodeReward(node, data, params)

def getNodeDesc(node, data, params):
    # Conver a node as "row, col" to a world
    # representation suitable for MissionPlanner

    desc = { "center"  : { "lat" : None, "lon" : None,
                           "row" : None, "col" : None, },
             "width"   : None,
             "length"  : None,
             "xpoints" : None,
             "ypoinys" : None,
             "margin"  : None,
             "roam"    : None,
    }

    desc["ypoints"] = abs(node["lower_coords_absolute"]["y"] - \
                          node["upper_coords_absolute"]["y"])
    desc["xpoints"] = abs(node["lower_coords_absolute"]["x"] - \
                          node["upper_coords_absolute"]["x"])

    desc["length"]  = desc["ypoints"]
    desc["width"]   = desc["xpoints"]

    desc["center"]["row"] = node["upper_coords_absolute"]["y"] + \
                            int(0.5 * desc["length"])
    desc["center"]["col"] = node["upper_coords_absolute"]["x"] + \
                            int(0.5 * desc["width"])

    centerArchive = getArchiveByGrid(desc["center"]["row"],
                                              desc["center"]["col"],
                     data["grid"], data["region"].GetGeoTransform())
    desc["center"]["lat"] = centerArchive["lat"]
    desc["center"]["lon"] = centerArchive["lon"]

    desc["margin"] = data["defaultMargin"]
    desc["roam"]   = data["defaultRoam"]

    return desc

def describeNodesList(nodes, data, params):
    for node in nodes:
        node["desc"] = getNodeDesc(node, data, params)

def getTargetsTable(nodes, data, params):
    numNodes = len(nodes)
    ran = range(len(nodes))

    IDs       = range(1, len(nodes) + 1)
    scores    = [100 for i in ran]
    lengths   = [0   for i in ran]
    widths    = [0   for i in ran]
    ypoints   = [0   for i in ran]
    xpoints   = [0   for i in ran]
    margins   = [0.0 for i in ran]
    roams     = [0.0 for i in ran]
    lats      = [0.0 for i in ran]
    lons      = [0.0 for i in ran]

    for ni in ran:
        lengths  [ni] = nodes[ni]["desc"]["length"]
        widths   [ni] = nodes[ni]["desc"]["width"]
        ypoints  [ni] = nodes[ni]["desc"]["ypoints"]
        xpoints  [ni] = nodes[ni]["desc"]["xpoints"]
        lats     [ni] = nodes[ni]["desc"]["center"]["lat"]
        lons     [ni] = nodes[ni]["desc"]["center"]["lon"]
        margins  [ni] = nodes[ni]["desc"]["margin"]
        roams    [ni] = nodes[ni]["desc"]["roam"]

    targetsTable = pd.DataFrame({
        "ID"      : IDs,
        "Score"   : scores,
        "Lon"     : lons,
        "Lat"     : lats,
        "Width"   : widths,
        "Length"  : lengths,
        "Xpoints" : xpoints,
        "Ypoints" : ypoints,
        "Margin"  : margins,
        "Roam"    : roams
    })

    return targetsTable

# MAIN
import pickle
import yaml
import statistics
import copy
import time
import os
import sys
import pandas   as pd
import numpy    as np
from   osgeo    import gdal
from   optparse import OptionParser
from   path     import Path
import TargetFitness

# Data types
nodeTemplate = { 'upper_coords_absolute' : {'x' : float, 'y' : float},
                 'lower_coords_absolute' : {'x' : float, 'y' : float},
                 'upper_coords_relative' : {'x' : float, 'y' : float},
                 'lower_coords_relative' : {'x' : float, 'y' : float},
                 'width' : 0, 'height' : 0,
                 'children' : [],
                 'grid'     : None,
                 'measures' : None,
                 'desc'     : None,
}

# Options
(options, args) = parseOption()

# Setup data and params
data   = makeDataDict(Path(options.config), Path(options.regionFile), Path(options.entitiesFile))
params = makeParamsDict(15, 100, 50)
tree   = { "root" : None, "numNodes" : 0, "numLeaves" : 0, "depth" : 0 }

# Run quadtree
tree["depth"] = quadgrid(tree, data, params)
# Get tree size
tree["numNodes"] = traverseTree(tree, False)
# Get leaves
leaves = getLeaves(tree)
tree["numLeaves"] = len(leaves)

# Calc leaf reward
getNodeListReward(leaves, data, params)
# Rank leaves
leavesRanked   = rankNodeList(leaves)
# Filter leaves
leavesFiltered = filterNodeList(leavesRanked)
# Select top N nodes
leavesSelected = leavesFiltered[0:options.numTargets]
# Get target descriptions
describeNodesList(leavesSelected, data, params)

# Get target table
targetsTable = getTargetsTable(leavesSelected, data, params)

# Output targets table
if options.outTable is not None: # Save target table as csv
    targetsTable.to_csv(options.outTable)
else: # Print to screen
    pd.set_option('display.max_rows', None)
    print(targetsTable)

# Save tree and targets as pickle
if options.pickle is not None:
    pickled = { "tree"     : tree,
                "selected" : leavesSelected,
                "pandas"   : targetsTable,
    }
    pickle.dump(pickled, open(options.pickle, "wb"))

printNodeList(leavesFiltered[0:20])
print ("Completed with depth:", tree["depth"],
       "Number of nodes",       tree["numNodes"],
       "Number of leaves:",     tree["numLeaves"],
       "Number remaining:",     len(leavesFiltered))

