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


## Input Data (and how to generate it)

### Inputs

- **Entity Book** (yaml): Describes entites, color codes, weights.  
    - Example: test/data/entityBook.yaml
- **Entity Map(s)** (png): Color-coded images where pixel color indicated entity presence. 
    - Example: test/data/dataGen/experiments/results/EXP_R1_S1/

### Generate Entity Maps

Entity maps are generated taking one entity map and a description of entities (csv) to perturb 
the input map with cellular automata to generate a number of other entity maps.

If you want each entity map to represent longer time durations (such as a periodic survey),
then you should generate the entity maps and take a uniform sample of the resulting maps.

This scheme does require a single entity map as input. If you want to start with a blank map,
just make a pure-white png whose dimensions are the same as the discrete target region.

Can optionally provide a csv script whose rows represent events that occur at a specified time step.
Events include sudden growth or decay at a specified location and radius. 

**Example run:** Creates 10 new maps based on entity map in Region 1 (R1)

        test/data/dataGen/genEntityDataCA.py \
            -e test/data/dataGen/samples/example_entities.csv \ # Entities description 
            -s test/data/dataGen/samples/example_events.csv \   # Events script
            -i 10 \                                             # Number of maps to generate
            -m test/data/entitiesMap_R1 \                       # Base Entity Map
            -d test/data/dataGen/result_entityMaps/R1_example \ # Store new maps
            -p R1_ex                                            # New maps path prefix

**Outputs**:
- **Entity Map(s)**: New maps in the directory specified.

## Reward Assignment (assignReward.py)
Using the **Entity Book** and a history of entities (ordered set of **Entity Maps**), 
assigns reward to each cell in the discrete region grid. 

### Outputs
- **Analyzed Grid(s)** (geotiff): Multi-band rasters of stat-based rewards for cells in region.
    - Each entity has own **Analyzed Grid** and each band corresponds to a stat, such as entity density, speed, or acceleration.
    - Example: 


## Target Region Selection (selectRegions.py)
