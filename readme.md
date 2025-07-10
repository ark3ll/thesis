# Encoder-less SLAM Implementation

This is my implementation for my bachelor thesis on encoder-less robot navigation using scan-to-map ICP.

## Requirements
- Python 3.9+
- NumPy
- SciPy  
- Matplotlib

Install with: `pip install numpy scipy matplotlib`

## Dataset Setup
You need to download the datasets and put them in a `RawData/` folder:
- Intel Research Lab: intel.clf and intel.gfs.log
- FR079: fr079_uncorrected.log and fr079-corrected-odometry.gt.txt

I got them from:
https://www.ipb.uni-bonn.de/datasets/index.html
## Running the Code

### Quick Test
To run all three SLAM modes on the Intel dataset:
```bash
python3 run_experiments.py
```

This creates visualizations comparing pure odometry, pure ICP, and hybrid approaches.

### Single Mode
To run just one mode:
```bash
python3 main.py --mode pure_icp --dataset intel
```

### Finding Tuned Parameters
Run parameter tuning to output the parameters that achieve smallest final drift values (warning: this takes forever):
```bash
python3 tuner.py --quick  # quick test with fewer params
python3 tuner.py          # full parameter search (takes hours)
```


## My Best Parameters
After lots of tuning on the Intel dataset, these work best:
- Voxel size: 0.05m
- Max iterations: 70
- Tolerance: 1e-6  
- Max correspondence distance: 0.3m

## File Descriptions
- `icp.py` - Core ICP algorithm and map management
- `slam.py` - Main SLAM system with three modes
- `data_loader.py` - Loads Intel and FR079 datasets
- `visualization.py` - Creates all the plots
- `main.py` - Run single experiments
- `run_experiments.py` - Run all modes for thesis results
- `tuner.py` - Parameter optimization (grid search)

## Notes
- The FR079 dataset has a coordinate offset issue that I fixed in the data loader
- Processing all 13,631 scans can take a long time!
- Pure odometry drift is really bad (49.63m on Intel dataset)
- Parameter tuning was done on the hybrid approach

## Thesis Results
To reproduce my thesis results exactly:
1. Run `python3 run_experiments.py` with default params
2. Look for `thesis_final_comparison.png` 
3. Check `thesis_results/` folder for detailed outputs

Sam Daly (2756086)