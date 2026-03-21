# Powderday SED Job Submission: Parallelization Guide

## Overview

After running `create_master()`, two types of job submission scripts are generated for each snapshot. Each uses a different parallelization strategy optimized for different use cases.

---

## Strategy 1: GNU Parallel (Intra-node Parallelization)

**File:** `snap_XXX/master.snap{snap}.job`

**How it works:**
- Single Slurm job receives 8 CPUs
- GNU `parallel` command launches 8 galaxy processing jobs **simultaneously**
- Processes galaxies in waves as CPUs become available

**Timeline for 16 galaxies @ 8 CPUs:**
```
Wave 1 (0-10 min):  [Gal1] [Gal2] [Gal3] [Gal4] [Gal5] [Gal6] [Gal7] [Gal8]
Wave 2 (10-20 min): [Gal9] [Gal10] [Gal11] [Gal12] [Gal13] [Gal14] [Gal15] [Gal16]
Total: ~20 minutes
```

### When to use:
- ✓ 1-50 galaxies per snapshot
- ✓ Want quickest absolute runtime
- ✓ High cluster queue times (single job = faster scheduling)
- ✓ Sufficient memory available (30GB per job)
- ✓ Want consistent resource allocation within one node

### Advantages:
- High CPU utilization (all 8 cores busy)
- Low memory overhead
- Minimal startup overhead
- Faster for small datasets

### Disadvantages:
- Limited by single node resources
- Limited scalability beyond 50 galaxies

---

## Strategy 2: Slurm Job Array (Inter-node Parallelization)

**File:** `snap_XXX/master.snap{snap}.array.job`

**How it works:**
- One "parent" job array with N tasks (N = number of galaxies)
- Slurm specification: `#SBATCH --array=1-N%16` means:
  - Create N tasks total
  - Run **maximum 16 tasks in parallel**
  - Remaining tasks queue automatically
  - As tasks finish, new ones start automatically
- Each galaxy gets 1 CPU, isolated execution

**Timeline for 130 galaxies @ 16 parallel limit:**
```
Batch 1 (0-10 min):   16 galaxies process in parallel
Batch 2 (10-20 min):  Next 16 galaxies
Batch 3 (20-30 min):  Next 16 galaxies
... (9 batches total)
Total: ~90 minutes (ceil(130/16) = 9 batches × ~10 min/batch)
```

**Slurm queue visualization:**
```
Time 0:  [Running: Gal1-16 (16 active)] [Pending: Gal17-130 (114 queued)]
Time 10: [Running: Gal17-32 (16 active)] [Pending: Gal33-130 (98 queued)]
Time 20: [Running: Gal33-48 (16 active)] [Pending: Gal49-130 (82 queued)]
... etc ...
```

### When to use:
- ✓ 100+ galaxies per snapshot (like 130 example)
- ✓ Cluster has many available nodes
- ✓ Want better distribution across infrastructure
- ✓ Want automatic load balancing
- ✓ Individual job failures should be isolated
- ✓ Need to query per-galaxy status

### Advantages:
- Scales to 1000+ galaxies efficiently
- Automatic load balancing across cluster
- Each job is independent (failure isolation)
- Better distribution of I/O
- Many small jobs easier to schedule than few large ones

### Disadvantages:
- Each galaxy gets 1 CPU only (vs 8 shared in parallel mode)
- Each job has 1-2 minute startup overhead
- More total I/O requests (each loads Powderday data)
- Noisier job queue (many tickets instead of 1)

---

## Quick Comparison

| Aspect | GNU Parallel | Job Array |
|--------|--------------|-----------|
| **Galaxies per job** | 8 parallel | 1 per task |
| **Best for** | 1-50 galaxies | 100-1000+ galaxies |
| **Node requirement** | Single | Multiple |
| **CPU per galaxy** | 1-2 (shared pool) | 1 (dedicated) |
| **Memory per job** | 27GB shared | 8GB per task |
| **Total runtime (130 gals)** | ~170 min+ | ~90 min |
| **Job queue tickets** | 1 | 130 |
| **Queue wait** | Varies | Usually shorter |

---

## Submission Methods

### Option A: Auto-submit all snapshots (parallel mode)
```bash
bash submit_all_snaps.sh
```
Automatically submits all `master.snap{snap}.job` files in parallel

### Option B: Auto-submit all snapshots (array mode)
```bash
for snap in snap_*/; do 
  sbatch "$snap/master.snap"*.array.job
done
```

### Option C: Manual submission per snapshot
```bash
# Parallel mode for snap 044
sbatch snap_044/master.snap044.job

# Array mode for snap 044
sbatch snap_044/master.snap044.array.job
```

### Option D: Local machine (auto-parallelizes)
```bash
bash snap_044/run_local.sh
```
Uses `gnu parallel -j 0` (auto-detects all cores)

---

## Monitoring Jobs

### Parallel mode (single job):
```bash
# View job
squeue -u $USER

# Follow output in real-time
tail -f snap_044/out/*.o

# Check completion
ls snap_044/gal_*/snap044_*.LOG | wc -l
```

### Array mode (multiple jobs):
```bash
# View all running/pending tasks
squeue -u $USER -t RUNNING
squeue -u $USER -t PENDING

# View array job status
sacct -j PARENT_JOB_ID

# Count completed galaxies
find snap_044/gal_* -name "*.LOG" | wc -l
```

---

## Tuning the %16 Limit (Array Mode)

The `--array=1-N%16` setting controls parallelism:

```bash
# More aggressive parallelization (cluster has many nodes)
--array=1-N%32    # Run 32 jobs in parallel (faster, more load)

# Conservative parallelization (cluster is shared)
--array=1-N%8     # Run 8 jobs in parallel (slower, less load)

# For 130 galaxies:
%8:  ceil(130/8)  = 17 batches ≈ 170 min total
%16: ceil(130/16) = 9 batches  ≈ 90 min total
%32: ceil(130/32) = 5 batches  ≈ 50 min total
```

Edit `snap_XXX/master.snap{snap}.array.job` and change the `--array` line:
```bash
#SBATCH --array=1-130%16    # Change 16 to your desired limit
```

---

## Log File Locations

All output logs stored here:
```
snap_XXX/out/              # Slurm stdout
snap_XXX/err/              # Slurm stderr  
snap_XXX/gal_NNNN/         # Per-galaxy output directory
snap_XXX/gal_NNNN/*.LOG    # Per-galaxy Powderday logs
```

## Recommendations

📊 **For your 130 galaxies example:**
- **Use Array Mode** (`master.snap{snap}.array.job`)
- This is the best fit for medium-large surveys
- Runs in ~90 minutes with automatic load balancing
- Submit with: `sbatch snap_*/master.snap*.array.job`

🚀 **Quick-and-dirty 5-10 galaxies:**
- **Use Parallel Mode** (`master.snap{snap}.job`)
- Faster scheduling, complete in ~10-20 minutes
- Submit with: `just sbatch snap_044/master.snap044.job`

🖥️ **On your local machine:**
- Run `bash snap_*/run_local.sh`
- Uses all available cores automatically
