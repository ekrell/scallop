# scallop
Simple analyst agent for mission planning where aspects of planning are handled by dedicated agents.

## Mission Planning with Dedicated Agent Roles

Much like a ship crewed by multiple persons, an autonomous vehicle may be controlled by multiple agents.
Commonly, there is a navigator role, planner role, map-maker role, etc. 

This is part of a project that does mission planning with _analyst_ and _surveyor_ roles.
The _analyst_ handles generation of targets, including data processing and statistics. 
The _surveyor_ generates motion plans to achieve targets, while considering safety and efficiency. 

**Scallop** is a simple analyst that assigns reward and selects targets based on properties of entities.
_Entities_ is a generic term for whatever the _analyst_ is interested in studying.
For example, _scallop_ may be dealing with subsurface seagrasses imaged by an unmanned surface vehicle.

**Scallop** currently does two tasks.
- Assign reward to a discrete region based on a history of entity presence in that region.
- Select subregion polygons to be the top n targets.


## Input Data (And how to generate it)

### Inputs

- **Entity Book** (yaml): Describes entites, color codes, weights.  
    - Example: test/data/entityBook.yaml
- **Entity Map(s)** (png): Color-coded images where pixel color indicated entity presence. 
    - Example: test/data/dataGen/experiments/results/EXP_R1_S1/

## Reward Assignment (assignReward.py)
Using the **Entity Book** and a history of entities (ordered set of **Entity Maps**), 
assigns reward to each cell in the discrete region grid. 

### Outputs
- **Analyzed Grid(s)** (geotiff): Multi-band rasters of stat-based rewards for cells in region.
    - Each entity has own **Analyzed Grid** and each band corresponds to a stat, such as entity density, speed, or acceleration.
    - Example: 


## Target Region Selection (selectRegions.py)
