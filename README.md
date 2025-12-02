# 2D Transfer Functions for Direct Volume Rendering

## Project Overview

This project implements an interactive volume rendering system using **2D Transfer Functions** for better visualization of volumetric medical data. Unlike traditional 1D transfer functions that only use intensity values, this system uses **both intensity and gradient magnitude** to create more detailed visualizations.

### What Makes This Different
- **2D Transfer Functions**: Uses intensity (voxel brightness) and gradient magnitude (edge strength) for better material classification
- **Semi-Automatic Design**: Uses machine learning (K-means, Mean-shift) to automatically identify materials in the volume
- **Interactive Web Interface**: Real-time parameter adjustment with immediate feedback

## Features

### Core Functionality
- ✅ **2D Histogram Computation**: Analyzes intensity vs gradient magnitude distribution
- ✅ **Semi-Automatic Mode**: K-Means and Mean-Shift clustering for automatic material classification
- ✅ **Manual Mode**: User-controlled intensity range, opacity, and color schemes
- ✅ **Real-time Volume Rendering**: Interactive 3D visualization using VTK
- ✅ **Gradient Boundary Emphasis**: Highlights material boundaries and surfaces
- ✅ **Histogram Visualization**: Shows 2D histogram with cluster boundaries

### Additional Features
- 📊 Interactive cut planes (XY, XZ, YZ)
- 🎨 Multiple color schemes (Rainbow, Grayscale, Warm, Cool)
- 🔍 Iso-surface extraction
- 📷 **Screenshot export** - Save high-quality PNG images
- ⏱️ **Performance metrics** - Displays histogram and clustering computation time
- 🔄 Camera controls with reset
- ⚡ Loading progress indicator
- 🎛️ Gradient shading parameters (disabled by default)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Packages
```bash
pip install vtk
pip install trame
pip install trame-vuetify
pip install trame-vtk
pip install scikit-learn
pip install matplotlib
pip install numpy
```

### Quick Install
```bash
pip install vtk trame trame-vuetify trame-vtk scikit-learn matplotlib numpy
```

## Usage

### 1. Start the Application
```bash
python r.py
```

The server will start on port 7654. You should see:
```
============================================================
2D Transfer Function Volume Rendering Application
============================================================
Scikit-learn available: True
Matplotlib available: True
Starting server on port 7654...
============================================================

App running at:
 - Local:   http://localhost:7654/
```

### 2. Open in Browser
Navigate to: `http://localhost:7654/`

### 3. Basic Workflow

#### Step 1: Load Data
1. Enter the data folder path (default: `./Chamaeleo_calyptratus/8bit`)
2. Click **"Load Dataset"**
3. Wait for the loading progress indicator (loads 1080 TIFF images)

#### Step 2: Compute 2D Histogram
1. Click **"Compute 2D Histogram"** button in the toolbar
2. Wait for histogram computation (shows intensity vs gradient distribution)

#### Step 3: Enable Volume Rendering
1. Check **"Enable Volume Rendering"** checkbox
2. The 3D volume should appear in the viewer

#### Step 4: Choose Transfer Function Mode

**Option A: Semi-Automatic (Recommended)**
1. Select **"Semi-Automatic"** radio button
2. Choose clustering method:
   - **K-Means**: Fast, need to specify number of clusters
   - **Mean-Shift**: Finds clusters automatically (slower)
3. Adjust **"Number of Clusters"** (2-10) for K-Means
4. Colors and opacity are assigned automatically to each cluster

**Option B: Manual**
1. Select **"Manual"** radio button
2. Adjust **"Intensity Range"** slider
3. Set opacity for low, mid, and high intensity regions
4. Choose color scheme (Rainbow, Grayscale, Warm, Cool)

#### Step 5: Adjust Settings
- Check **"Emphasize Boundaries"** to highlight edges
- Check **"Show 2D Histogram"** to see the histogram with cluster boundaries
- Uncheck **"Enable Gradient Shading"** (off by default) for faster rendering

#### Step 6: Export and Analyze
- Click **"Screenshot"** button in toolbar to save PNG images
- View **performance metrics** to see computation times
- Adjust parameters and compare results

## Dataset

### Chameleon CT Scan
- **Source**: DigiMorph (digimorph.org)
- **Subject**: Chameleon (Chamaeleo calyptratus)
- **Resolution**: 1024 × 1024 × 1080 voxels
- **Format**: 8-bit TIFF image stack
- **Intensity Range**: 0-255
- **Total Slices**: 1080 images

### Data Structure
```
Chamaeleo_calyptratus/
└── 8bit/
    ├── slice_0001.tif
    ├── slice_0002.tif
    ├── ...
    └── slice_1080.tif
```

## Controls Reference

### 1. Data Loading
- **Data Folder Path**: Path to TIFF image stack
- **Load Dataset**: Button to load the volume data
- **Progress Indicator**: Shows loading status

### 2. 2D Histogram
- **Compute 2D Histogram**: Analyzes data distribution
- **Show 2D Histogram**: Displays histogram

### 3. Volume Rendering
- **Enable Volume Rendering**: Turn 3D visualization on/off

### 4. Transfer Function Settings

#### Manual Mode
- **Intensity Range**: Select which intensities to show
- **Opacity (Low/Mid/High)**: Control transparency at different intensity levels
- **Color Scheme**: Choose color palette

#### Semi-Automatic Mode
- **Clustering Method**: K-Means or Mean-Shift
- **Number of Clusters**: How many material types to identify (K-Means only)

### 5. Gradient Parameters
- **Emphasize Boundaries**: Highlight edges using gradient
- **Enable Gradient Shading**: Add lighting effects

### 6. Cut Planes
- **XY/XZ/YZ Plane**: Show/hide orthogonal slices
- **Plane Position**: Slider to move cut plane

### 7. Iso-Surface
- **Iso-Surface Value**: Extract surface at specific intensity
- **Checkbox**: Show/hide iso-surface

### 8. Toolbar
- **Screenshot**: Save current view as PNG (with timestamp)
- **Reset Parameters**: Restore defaults
- **Reset Camera**: Return to default view

### 9. Performance Metrics
- **Computation Time**: Shows how long histogram and clustering took
- **Real-time Display**: Updates after each operation

## Technical Details

### Components
- **Backend**: VTK for volume rendering
- **Frontend**: Trame web framework with Vuetify3 UI
- **Clustering**: Scikit-learn (KMeans, MeanShift)
- **Visualization**: Matplotlib for histograms

### How It Works
1. **Data Input**: Load 3D volume from TIFF stack
2. **Gradient Computation**: Calculate gradient magnitude using VTK
3. **2D Histogram**: Compute intensity vs gradient distribution
4. **Clustering** (Semi-Automatic): Find material clusters
5. **Transfer Function**: Map intensity+gradient to color+opacity
6. **Volume Rendering**: Apply TF and render volume

### Clustering Methods

#### K-Means
- Splits histogram into K clusters
- Fast and deterministic
- You specify number of clusters
- Works well when you know how many materials to expect

#### Mean-Shift
- Finds cluster count automatically
- Slower but more flexible
- No need to specify cluster count
- Good for exploring unknown datasets

## Project Structure

```
Project/
├── r.py                          # Main application file
├── README.md                     # This file
├── Chamaeleo_calyptratus/       # Dataset folder
│   └── 8bit/                    # 8-bit TIFF images
│       ├── slice_0001.tif
│       └── ...
```

## Troubleshooting

### Common Issues

**Issue: "No module named 'vtk'"**
- Solution: `pip install vtk`

**Issue: "Scikit-learn not available"**
- Solution: `pip install scikit-learn`
- Semi-automatic mode will be disabled without it

**Issue: "Dataset not found"**
- Solution: Check that the data path is correct
- Ensure TIFF files are in the specified folder

**Issue: "Segmentation fault"**
- Solution: Ensure VTK is properly installed
- Try: `pip uninstall vtk && pip install vtk`

**Issue: "Histogram shows individual letters instead of options"**
- Solution: Update to latest version (fixed in current release)

**Issue: "Volume rendering shows nothing"**
- Solution: 
  1. Ensure data is loaded first
  2. Click "Compute 2D Histogram"
  3. Then enable volume rendering

**Issue: "Screenshot not saving"**
- Solution:
  1. Check that you have write permissions in the current directory
  2. Screenshots are saved as `screenshot_YYYYMMDD_HHMMSS.png`
  3. Look in the same directory as `r.py`

## Performance Notes

- **Loading**: ~10-30 seconds for 1080 images
- **Histogram Computation**: ~2-5 seconds
- **K-Means Clustering**: ~0.5-1 seconds
- **Mean-Shift Clustering**: ~2-5 seconds
- **Rendering**: Real-time (60 FPS on modern hardware)

## Educational Context

**Course**: COSC 6344 - Visualization  
**Institution**: University of Houston  
**Semester**: Fall 2025  
**Project Type**: Final Project - Direct Volume Rendering

### Learning Objectives
1. Understand multidimensional transfer functions
2. Apply machine learning for visualization
3. Implement interactive scientific visualization
4. Use VTK for volume rendering
5. Create web-based visualization applications

## References

### Papers & Resources
- Kniss et al. (2002) - "Multidimensional Transfer Functions for Interactive Volume Rendering"
- Kindlmann & Durkin (1998) - "Semi-Automatic Generation of Transfer Functions"
- VTK Documentation: https://vtk.org/doc/
- Trame Framework: https://kitware.github.io/trame/

### Dataset Source
- DigiMorph Digital Morphology Library
- URL: http://digimorph.org/
- Specimen: Chamaeleo calyptratus (Veiled Chameleon)

## Future Enhancements

Potential improvements for future versions:
- [ ] Interactive transfer function editor (click on histogram to add/edit control points)
- [ ] Side-by-side comparison view (1D vs 2D, requires advanced rendering setup)
- [ ] Video export and animation capabilities
- [ ] Transfer function presets library (CT Bone, Soft Tissue, etc.)
- [ ] Undo/redo functionality for parameter changes
- [ ] Multiple dataset support and switching
- [ ] Collaborative viewing mode
- [ ] Advanced lighting models

## License

This project is created for educational purposes as part of COSC 6344 coursework.

## Author

**Mengmei He**  
University of Houston  
Fall 2025

## Acknowledgments

- Professor and teaching staff of COSC 6344
- DigiMorph for providing the dataset
- VTK and Trame communities
- Kitware for visualization tools

---

**Last Updated**: December 1, 2025
