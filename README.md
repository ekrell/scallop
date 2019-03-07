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

## Dependencies

**entityFitness.py**
- [PyYAML](pyyaml.org)
- [pandas](pandas.pydata.org)
- [Yattag](www.yattag.org)
- [GDAL](pypi.org/project/GDAL)

**assignReward.py**
- [NumPY](www.numpy.org)
- [pandas](pandas.pydata.org)
- [path.py](pypi.org/project/path.py)
- [statistics](docs.python.org/3/library/statistics.html)


## Input Data (and how to generate it)

### Inputs

- **Entity Book** (yaml): Describes entites, color codes, weights.  
    - Example: test/data/entityBook.yaml
- **Entity Map(s)** (png): Color-coded images where pixel color indicated entity presence. 
    - Example: test/data/dataGen/experiments/results/EXP_R1_S1/
- **Region Map** (geotiff): Region with same dimensions and spatial extent as the Entity Map(s).

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

        python test/data/dataGen/genEntityDataCA.py \
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

        python assignReward.py \
            -r test/data/R1.tif \                                       # Region map
            -d test/data/dataGen/result_entityMaps/R1_uniform_growth    # Dir with entity maps
            -o test/data/output \                                       # Dir to store output
            -l R1_uniform_growth \                                      # Output file prefix
            -n 100 \                                 # len, width of each contextual subgrid
            -e test/data/entityBook.yaml                                # Path to Entity Book

### Outputs
- **Analyzed Grid(s)** (geotiff): Multi-band rasters of stat-based rewards for cells in region.
    - Each entity has own **Analyzed Grid** and each band corresponds to a stat, such as entity density, speed, or acceleration.
    - Example: /test/data/output/R1_uniform_growth_analysis_E4.tiff
- **Grid Stats** (pickle): Density, speed, acceleration of entities in each subgrid region.
    - Example: /test/data/output/R1_uniform_growth_gridstats_19.pickle 

## Target Region Selection (selectRegions.py)
