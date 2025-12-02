"""
COSC 6344 Visualization - Final Project
Direct Volume Rendering with 2D Transfer Functions

This extends the base volume rendering with semi-automatic 2D transfer
function design using intensity and gradient magnitude.
"""

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets.vtk import VtkRemoteView
from trame.widgets import html

from vtkmodules.vtkIOLegacy import vtkDataSetReader 
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonCore import vtkFloatArray

import vtk as vtk_lib
import numpy as np
import math
import tempfile
import os
import colorsys
import random

# Import for 2D TF clustering
try:
    from sklearn.cluster import KMeans, MeanShift
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
    print("✓ Scikit-learn available for semi-automatic clustering")
except ImportError:
    print("⚠ Warning: scikit-learn not installed. Semi-automatic clustering will be disabled.")
    print("  Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Import for histogram visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    MATPLOTLIB_AVAILABLE = True
    print("✓ Matplotlib available for 2D histogram visualization")
except ImportError:
    print("⚠ Warning: matplotlib not installed. 2D histogram visualization will be disabled.")
    print("  Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

# Additional imports for digimorph data handling
import glob
from pathlib import Path

# Trame setup
server = get_server()
state, ctrl = server.state, server.controller

# VTK pipeline setup
reader = None
geometry_filter = vtk_lib.vtkGeometryFilter()
mapper = vtk_lib.vtkPolyDataMapper()
actor = vtk_lib.vtkActor()
renderer = vtk_lib.vtkRenderer()
renderer.SetBackground(0.15, 0.15, 0.25)  # Darker background for volume rendering
render_window = vtk_lib.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetOffScreenRendering(1)

interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor_style = vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(interactor_style)
interactor.Initialize()

lut = vtk_lib.vtkLookupTable()

# Global variables
loaded_data = None
surface_actor = None
dim = None

# Original 1D transfer functions
color_transfer_functions = {
    "cylinder": [
        {"scalar": -4.8, "r": 0, "g": 0, "b": 0, "opacity": 0},
        {"scalar": 2, "r": 1, "g": 1, "b": 1, "opacity": 1},
    ],
    "fullhead": [
        {"scalar": 0, "r": 0, "g": 0, "b": 0, "opacity": 0},
        {"scalar": 500, "r": 0.33, "g": 0.33, "b": 0.33, "opacity": 0.15},
        {"scalar": 1000, "r": 0.66, "g": 0.66, "b": 0.66, "opacity": 0.15},
        {"scalar": 1150, "r": 1, "g": 1, "b": 1, "opacity": 0.85},
    ]
}

# Outline box
outlineData = vtk_lib.vtkOutlineFilter()
mapOutline = vtk_lib.vtkPolyDataMapper()
outlineActor = vtk_lib.vtkActor()

# Cut plane actors
xy_plane_actor = vtk_lib.vtkImageActor()
xz_plane_actor = vtk_lib.vtkImageActor()
yz_plane_actor = vtk_lib.vtkImageActor()
plane_Colors = vtk_lib.vtkImageMapToColors()

# Volume rendering
volume_mapper = vtk_lib.vtkSmartVolumeMapper()
volume = vtk_lib.vtkVolume()
volume_property = vtk_lib.vtkVolumeProperty()
gradient_filter = None

# 2D Transfer Function Data
joint_histogram = None
intensity_range = [0, 255]
gradient_range = [0, 1]
cluster_labels = None
cluster_centers = None

# State variables - Original
state.iso_alpha = 1
state.xy_plane_visible = False
state.xz_plane_visible = False
state.yz_plane_visible = False
state.volume_render_enabled = False
state.show_transfer_function_dialog = False
state.transfer_function_csv = ""
state.edit_mode = False
state.scalar_range = [0, 100]
state.selected_transfer_function = "cylinder"
state.transfer_function = color_transfer_functions["cylinder"]

# State variables - NEW for 2D Transfer Functions
state.tf_mode = "1D"  # "1D" or "2D"
state.tf_method = "manual"  # "manual" or "semi-automatic"
state.use_2d_tf = True  # Default to 2D transfer functions
state.show_2d_histogram = False
state.histogram_image = ""

# Transfer function presets
state.tf_preset = "custom"  # Selected preset
state.tf_presets = [
    {"title": "Custom", "value": "custom"},
    {"title": "CT Bone", "value": "ct_bone"},
    {"title": "CT Soft Tissue", "value": "ct_soft_tissue"},
    {"title": "CT Muscle & Bone", "value": "ct_muscle_bone"},
    {"title": "MRI Default", "value": "mri_default"},
    {"title": "High Contrast", "value": "high_contrast"},
    {"title": "Low Opacity", "value": "low_opacity"},
]

# Clustering parameters
state.clustering_method = "kmeans"  # "kmeans" or "meanshift"
state.num_clusters = 3
state.boundary_emphasis = True
state.gradient_opacity_weight = 0.7

# Manual 2D TF parameters
state.manual_intensity_min = 0.0  # Minimum intensity threshold (0-1 normalized)
state.manual_intensity_max = 1.0  # Maximum intensity threshold (0-1 normalized)
state.manual_intensity_range = [0.0, 1.0]  # Range slider value
state.manual_opacity_low = 0.2    # Opacity at low intensities (increased from 0.0)
state.manual_opacity_mid = 0.6    # Opacity at mid intensities (increased from 0.5)
state.manual_opacity_high = 0.9   # Opacity at high intensities
state.manual_gradient_weight = 0.5  # How much gradient affects opacity (0-1)
state.manual_color_scheme = "rainbow"  # Color scheme: rainbow, grayscale, warm, cool

state.clustering_options = [
    {"title": "K-Means", "value": "kmeans"},
    {"title": "Mean-Shift", "value": "meanshift"},
]

# Rendering quality
state.enable_gradient_shading = False
state.ambient = 0.3
state.diffuse = 0.7
state.specular = 0.4
state.specular_power = 20

# Performance metrics
state.histogram_time = 0.0
state.clustering_time = 0.0
state.loading_time = 0.0
state.performance_message = ""
state.screenshot_in_progress = False  # Prevent multiple simultaneous screenshots

# NEW: Digimorph dataset support
state.digimorph_data_path = "./Chamaeleo_calyptratus/8bit"  # Default path
state.auto_load_chameleon = False
state.is_loading = False  # Loading indicator
state.loading_message = ""  # Loading progress message

# Helper functions
def is_data_loaded():
    return loaded_data is not None and loaded_data.GetNumberOfPoints() > 0

def reset_camera():
    if is_data_loaded():
        renderer.ResetCamera()
        ctrl.view_reset_camera()
        ctrl.view_update()
    else:
        state.error_message = "No data loaded to reset camera."

def reset_parameters():
    """Reset all parameters to their default values."""
    print("🔄 Resetting all parameters to defaults...")
    
    # Volume rendering
    state.volume_render_enabled = False
    
    # 2D Transfer Function
    state.tf_method = "manual"
    state.use_2d_tf = True
    state.show_2d_histogram = False
    
    # Manual 2D TF parameters
    state.manual_intensity_min = 0.0
    state.manual_intensity_max = 1.0
    state.manual_intensity_range = [0.0, 1.0]
    state.manual_opacity_low = 0.5  # Increased from 0.2 for better visibility
    state.manual_opacity_mid = 0.7  # Increased from 0.6
    state.manual_opacity_high = 0.9
    state.manual_gradient_weight = 0.3  # Reduced from 0.5 for less gradient emphasis
    state.manual_color_scheme = "rainbow"
    
    # Clustering parameters
    state.clustering_method = "kmeans"
    state.num_clusters = 3
    state.boundary_emphasis = True
    
    # Gradient shading
    state.enable_gradient_shading = True
    state.ambient = 0.3
    state.diffuse = 0.7
    state.specular = 0.4
    state.specular_power = 20
    
    # Iso-surface
    state.isosurface_visible = False
    state.iso_alpha = 1
    
    # Cut planes
    state.xy_plane_visible = False
    state.xz_plane_visible = False
    state.yz_plane_visible = False
    
    # Reset camera if data is loaded
    if is_data_loaded():
        reset_camera()
    
    ctrl.view_update()
    state.error_message = "✅ All parameters reset to defaults"
    print("✅ Parameters reset complete")

def save_screenshot():
    """Save current rendering to a PNG file."""
    # Prevent multiple simultaneous screenshots
    if state.screenshot_in_progress:
        print("⚠️ Screenshot already in progress...")
        return
        
    try:
        state.screenshot_in_progress = True
        import datetime
        
        # Generate filename with timestamp (add milliseconds for uniqueness)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"screenshot_{timestamp}.png"
        
        # Ensure render is complete
        render_window.Render()
        
        # Create VTK window to image filter
        window_to_image = vtk_lib.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetScale(1)  # Changed from 2 to avoid 2x2 tiling artifact
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.ShouldRerenderOff()  # Don't trigger additional renders
        window_to_image.Update()
        
        # Write to PNG
        writer = vtk_lib.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        
        state.error_message = f"✅ Screenshot saved: {filename}"
        print(f"📸 Screenshot saved: {filename}")
        return True
        
    except Exception as e:
        state.error_message = f"❌ Failed to save screenshot: {str(e)}"
        print(f"❌ Screenshot error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        state.screenshot_in_progress = False

def hsv_to_rgb(hsv):
    return colorsys.hsv_to_rgb(*hsv)

###############################
### 2D TRANSFER FUNCTION SECTION
###############################

def compute_joint_histogram(volume_data, num_bins=128):
    """
    Compute 2D histogram of intensity vs. gradient magnitude.
    This is the core data structure for 2D transfer functions.
    
    Args:
        volume_data: VTK volume data object containing scalar field
        num_bins: Number of bins for histogram (default 128x128)
        
    Returns:
        tuple: (histogram 2D array, x_edges, y_edges) or (None, None, None) on error
    """
    import time
    start_time = time.time()
    
    global joint_histogram, intensity_range, gradient_range, gradient_filter
    
    print("Computing joint histogram (intensity vs gradient magnitude)...")
    
    # Get scalar array
    scalars = volume_data.GetPointData().GetScalars()
    if scalars is None:
        print("No scalar data found in volume")
        return None, None, None
    
    # Convert to numpy array efficiently using VTK's numpy interface
    from vtkmodules.util import numpy_support
    intensity = numpy_support.vtk_to_numpy(scalars)
    
    # Sample data if too large using efficient stride-based sampling
    num_points = len(intensity)
    max_samples = 5_000_000  # 5 million points is plenty for histogram statistics
    stride = 1  # Default stride
    
    if num_points > max_samples:
        # Use stride-based sampling (much faster than random sampling)
        stride = num_points // max_samples
        print(f"⚠ Large dataset detected ({num_points:,} points). Using stride sampling (every {stride} points)...")
        intensity_sampled = intensity[::stride]
        actual_samples = len(intensity_sampled)
        print(f"  Sampled {actual_samples:,} points")
        use_stride = True
    else:
        print(f"Processing {num_points:,} points...")
        intensity_sampled = intensity
        use_stride = False
    
    intensity_range = [float(intensity_sampled.min()), float(intensity_sampled.max())]
    print(f"Intensity range: {intensity_range}")
    
    # Compute gradient magnitude using VTK
    print("Computing gradient magnitude...")
    gradient_filter = vtk_lib.vtkImageGradientMagnitude()
    gradient_filter.SetInputData(volume_data)
    gradient_filter.SetDimensionality(3)
    gradient_filter.Update()
    
    gradient_data = gradient_filter.GetOutput()
    gradient_scalars = gradient_data.GetPointData().GetScalars()
    
    # Convert gradient to numpy array efficiently
    gradient_mag = numpy_support.vtk_to_numpy(gradient_scalars)
    
    # Sample gradient with same stride if we sampled intensity
    if use_stride:
        gradient_mag_sampled = gradient_mag[::stride]
    else:
        gradient_mag_sampled = gradient_mag
    
    # Update global gradient range
    gradient_range[0] = float(gradient_mag_sampled.min())
    gradient_range[1] = float(gradient_mag_sampled.max())
    print(f"Gradient range: {gradient_range}")
    
    # Compute 2D histogram using sampled data
    print(f"Computing 2D histogram with {len(intensity_sampled):,} samples...")
    histogram, x_edges, y_edges = np.histogram2d(
        intensity_sampled, gradient_mag_sampled,
        bins=num_bins,
        range=[[intensity_range[0], intensity_range[1]], 
               [gradient_range[0], gradient_range[1]]]
    )
    
    joint_histogram = histogram
    
    # Record timing
    elapsed_time = time.time() - start_time
    state.histogram_time = elapsed_time
    print(f"✅ Histogram computed: {histogram.shape}, non-zero bins: {np.count_nonzero(histogram)}")
    print(f"⏱️  Computation time: {elapsed_time:.2f}s")
    
    return histogram, x_edges, y_edges


def apply_clustering(histogram, method="kmeans", n_clusters=3):
    """
    Apply clustering to joint histogram to detect material boundaries.
    This implements the semi-automatic transfer function design.
    
    Args:
        histogram: 2D numpy array (intensity x gradient magnitude)
        method: "kmeans" or "meanshift"
        n_clusters: Number of clusters (only for K-means)
        
    Returns:
        tuple: (label_map 2D array, cluster_centers array) or (None, None) on error
    """
    import time
    start_time = time.time()
    
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn not available. Clustering disabled.")
        return None, None
    
    print(f"Applying {method} clustering with {n_clusters} clusters...")
    
    # Prepare data: create (intensity, gradient) pairs weighted by histogram counts
    h, w = histogram.shape
    points = []
    weights = []
    
    for i in range(h):
        for j in range(w):
            if histogram[i, j] > 0:  # Only non-zero histogram bins
                intensity_norm = i / h
                gradient_norm = j / w
                points.append([intensity_norm, gradient_norm])
                weights.append(histogram[i, j])
    
    if len(points) == 0:
        print("No valid points for clustering")
        return None, None
    
    points = np.array(points)
    weights = np.array(weights)
    
    print(f"Clustering {len(points)} histogram bins...")
    
    # Standardize features
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    # Apply clustering
    try:
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(points_scaled, sample_weight=weights)
            centers = scaler.inverse_transform(clusterer.cluster_centers_)
        elif method == "meanshift":
            # Mean-Shift doesn't support sample_weight, so we replicate points based on weights
            # to give histogram bins with higher counts more influence
            print("Preparing weighted samples for Mean-Shift...")
            
            # Subsample weights to manageable size (max 10000 points)
            max_samples = 10000
            if len(points) > max_samples:
                # Sample proportionally to weights
                weights_norm = weights / weights.sum()
                sample_indices = np.random.choice(
                    len(points), 
                    size=max_samples, 
                    replace=True, 
                    p=weights_norm
                )
                points_sampled = points_scaled[sample_indices]
            else:
                # Replicate points based on weights (scaled down)
                weights_scaled = (weights / weights.max() * 10).astype(int) + 1
                points_sampled = np.repeat(points_scaled, weights_scaled, axis=0)
                
                # If still too many, subsample
                if len(points_sampled) > max_samples:
                    sample_indices = np.random.choice(len(points_sampled), size=max_samples, replace=False)
                    points_sampled = points_sampled[sample_indices]
            
            print(f"Mean-Shift clustering with {len(points_sampled)} weighted samples...")
            
            # Estimate bandwidth if not specified, or use adaptive bandwidth
            from sklearn.cluster import estimate_bandwidth
            bandwidth = estimate_bandwidth(points_sampled, quantile=0.2, n_samples=min(500, len(points_sampled)))
            if bandwidth == 0:
                bandwidth = 0.3  # Fallback
            print(f"Using bandwidth: {bandwidth:.3f}")
            
            clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            clusterer.fit(points_sampled)
            
            # Predict labels for original points
            labels = clusterer.predict(points_scaled)
            centers = scaler.inverse_transform(clusterer.cluster_centers_)
        else:
            print(f"Unknown clustering method: {method}")
            return None, None
        
        print(f"Clustering complete. Found {len(centers)} clusters.")
        
        # Create cluster label map
        label_map = np.zeros_like(histogram, dtype=int) - 1  # -1 for unlabeled
        for idx, (i, j) in enumerate([(int(p[0] * h), int(p[1] * w)) for p in points]):
            if 0 <= i < h and 0 <= j < w:
                label_map[i, j] = labels[idx]
        
        # Record timing
        elapsed_time = time.time() - start_time
        state.clustering_time = elapsed_time
        print(f"⏱️  Clustering time: {elapsed_time:.2f}s")
        
        return label_map, centers
    
    except Exception as e:
        print(f"Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_histogram_image(histogram, label_map=None):
    """
    Generate matplotlib visualization of 2D histogram with optional cluster boundaries.
    
    Args:
        histogram: 2D numpy array (intensity x gradient magnitude)
        label_map: 2D array of cluster labels (optional, for boundary visualization)
        
    Returns:
        str: Base64 encoded PNG image data URL, or empty string on error
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""
    
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Plot histogram with log scale for better visibility
        hist_log = np.log10(histogram.T + 1)  # +1 to avoid log(0)
        im = ax.imshow(
            hist_log,
            origin='lower',
            cmap='hot',
            aspect='auto',
            extent=[intensity_range[0], intensity_range[1], 
                    gradient_range[0], gradient_range[1]],
            interpolation='bilinear'
        )
        
        ax.set_xlabel('Intensity', fontsize=12)
        ax.set_ylabel('Gradient Magnitude', fontsize=12)
        ax.set_title('2D Histogram: Intensity vs Gradient', fontsize=14, fontweight='bold')
        
        # Add cluster boundaries if available
        if label_map is not None:
            try:
                from scipy import ndimage
                # Find cluster boundaries
                for cluster_id in np.unique(label_map):
                    if cluster_id >= 0:  # Skip unlabeled regions
                        mask = (label_map == cluster_id).astype(float)
                        edges = ndimage.sobel(mask)
                        edges = edges > 0.1
                        
                        if np.any(edges):
                            # Overlay cluster boundaries
                            ax.contour(
                                np.linspace(intensity_range[0], intensity_range[1], label_map.shape[0]),
                                np.linspace(gradient_range[0], gradient_range[1], label_map.shape[1]),
                                mask.T,
                                levels=[0.5],
                                colors='cyan',
                                linewidths=2,
                                alpha=0.8
                            )
            except ImportError:
                # If scipy not available, just show histogram without boundaries
                pass
        
        plt.colorbar(im, ax=ax, label='Log10(Count + 1)')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Failed to generate histogram image: {e}")
        import traceback
        traceback.print_exc()
        return ""


def apply_2d_transfer_function():
    """
    Apply 2D transfer function using intensity and gradient magnitude.
    This is the key innovation of the project - using both intensity AND gradient
    magnitude to create more sophisticated visualizations.
    
    Transfer Function Design Modes:
    1. SEMI-AUTOMATIC MODE: Uses clustering results to generate transfer functions
    2. MANUAL MODE: Simple gradient-based coloring with linear opacity
    """
    global volume_property, joint_histogram, cluster_labels, cluster_centers
    global intensity_range, gradient_range, volume_mapper, volume
    
    if not is_data_loaded():
        print("❌ Cannot apply 2D transfer function - no data loaded")
        return
    
    print("🎨 Applying 2D transfer function...")
    
    # Check if histogram has been computed
    if joint_histogram is None:
        print("⚠️ 2D histogram not computed. Please compute histogram first.")
        state.error_message = "Please compute 2D histogram first using the button."
        return
    
    scalar_range = loaded_data.GetScalarRange()
    print(f"✓ Scalar range: {scalar_range}")
    
    # Create or update volume mapper
    if volume_mapper is None:
        print("🔧 Creating volume mapper...")
        volume_mapper = vtk_lib.vtkSmartVolumeMapper()
    
    volume_mapper.SetInputData(loaded_data)
    
    # CRITICAL: Set sample distance for large volumes
    # For very large datasets (1B+ voxels), use larger sample distance
    sample_dist = 1.0  # Larger value = faster rendering, coarser quality
    volume_mapper.SetSampleDistance(sample_dist)
    volume_mapper.SetAutoAdjustSampleDistances(0)  # Disable auto-adjustment
    volume_mapper.SetBlendModeToComposite()  # Use composite blending
    
    # Force interactive quality for better performance
    volume_mapper.SetRequestedRenderModeToGPU()
    
    print(f"🔧 Volume mapper configured:")
    print(f"   - Data connected: {volume_mapper.GetNumberOfInputConnections(0) > 0 or volume_mapper.GetInputDataObject(0, 0) is not None}")
    print(f"   - Sample distance: {sample_dist}")
    print(f"   - Render mode: GPU (if available)")
    print(f"   - Blend mode: Composite")
    
    # Create volume property if not exists
    if volume_property is None:
        print("🔧 Creating volume property...")
        volume_property = vtk_lib.vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()
        
        # Set shading properties
        if state.enable_gradient_shading:
            volume_property.ShadeOn()
            volume_property.SetAmbient(0.3)
            volume_property.SetDiffuse(0.7)
            volume_property.SetSpecular(0.4)
            volume_property.SetSpecularPower(20)
        else:
            volume_property.ShadeOff()
    
    # Apply clustering if semi-automatic mode
    print(f"🔍 Checking semi-automatic condition:")
    print(f"   - state.tf_method = '{state.tf_method}' (should be 'semi-automatic')")
    print(f"   - SKLEARN_AVAILABLE = {SKLEARN_AVAILABLE}")
    print(f"   - joint_histogram is None? {joint_histogram is None}")
    
    if state.tf_method == "semi-automatic" and SKLEARN_AVAILABLE:
        # Compute clustering with current settings
        print(f"🔧 Computing clustering with {state.num_clusters} clusters using {state.clustering_method}...")
        new_labels, new_centers = apply_clustering(
            joint_histogram,
            method=state.clustering_method,
            n_clusters=state.num_clusters
        )
        
        # Update global cluster variables
        if new_labels is not None and new_centers is not None:
            cluster_labels = new_labels
            cluster_centers = new_centers
            print(f"✅ Clustering updated: {len(cluster_centers)} clusters found")
        
        # Update histogram visualization with clusters
        if state.show_2d_histogram:
            state.histogram_image = generate_histogram_image(joint_histogram, cluster_labels)
    else:
        # Manual mode - just show histogram
        if state.show_2d_histogram:
            state.histogram_image = generate_histogram_image(joint_histogram, None)
    
    # Create color transfer function
    color_tf = vtk_lib.vtkColorTransferFunction()
    
    # Create opacity transfer function (intensity-based)
    opacity_tf = vtk_lib.vtkPiecewiseFunction()
    
    # Create gradient opacity transfer function (boundary emphasis)
    gradient_opacity_tf = vtk_lib.vtkPiecewiseFunction()
    
    if state.tf_method == "semi-automatic":
        if cluster_centers is not None and len(cluster_centers) > 0:
            # Semi-automatic: use cluster centers to define transfer function
            print(f"Using {len(cluster_centers)} clusters for transfer function")
            
            # Sort clusters by intensity
            sorted_indices = np.argsort([c[0] for c in cluster_centers])
            sorted_centers = cluster_centers[sorted_indices]
            
            # Define colors for clusters (extended to 10+ colors)
            colors = [
                (0.9, 0.2, 0.2),  # Red
                (0.2, 0.9, 0.2),  # Green
                (0.2, 0.2, 0.9),  # Blue
                (0.9, 0.9, 0.2),  # Yellow
                (0.9, 0.2, 0.9),  # Magenta
                (0.2, 0.9, 0.9),  # Cyan
                (1.0, 0.5, 0.0),  # Orange
                (0.5, 0.0, 1.0),  # Purple
                (0.0, 1.0, 0.5),  # Spring Green
                (1.0, 0.0, 0.5),  # Pink
                (0.6, 0.4, 0.2),  # Brown
                (0.5, 0.5, 0.5),  # Gray
            ]
            
            # Build transfer functions based on clusters
            for i, center in enumerate(sorted_centers):
                intensity_norm = center[0]
                gradient_norm = center[1]

                print(f"  - Cluster {i}: Intensity norm={intensity_norm:.3f}, Gradient norm={gradient_norm:.3f}")
                # Map to actual intensity value
                intensity_val = scalar_range[0] + intensity_norm * (scalar_range[1] - scalar_range[0])
                
                # Color assignment
                color = colors[i % len(colors)]
                color_tf.AddRGBPoint(intensity_val, color[0], color[1], color[2])
                
                # Opacity based on gradient magnitude (higher gradient = boundary = more visible)
                if state.boundary_emphasis:
                    opacity = 0.1 + 0.9 * gradient_norm  # Scale by gradient
                else:
                    opacity = 0.3 + 0.7 * (i / len(sorted_centers))
                
                opacity_tf.AddPoint(intensity_val, opacity)
            
            # Gradient opacity: emphasize boundaries (high gradients)
            if state.boundary_emphasis:
                gradient_opacity_tf.AddPoint(gradient_range[0], 0.0)  # Low gradient: transparent
                gradient_opacity_tf.AddPoint(gradient_range[1] * 0.25, 0.3)
                gradient_opacity_tf.AddPoint(gradient_range[1] * 0.5, 0.6)
                gradient_opacity_tf.AddPoint(gradient_range[1], 1.0)  # High gradient: opaque
            else:
                gradient_opacity_tf.AddPoint(gradient_range[0], 0.5)
                gradient_opacity_tf.AddPoint(gradient_range[1], 0.5)
        else:
            # No clustering computed yet - show a warning and use basic transfer function
            print("⚠️ Semi-automatic mode selected but clustering not computed. Using basic transfer function.")
            print("   Click 'COMPUTE 2D HISTOGRAM' button first, then clustering will be applied.")
            
            # Use a simple visible transfer function so user sees something
            color_tf.AddRGBPoint(scalar_range[0], 0.0, 0.0, 1.0)  # Blue
            color_tf.AddRGBPoint((scalar_range[0] + scalar_range[1]) * 0.5, 0.0, 1.0, 0.0)  # Green
            color_tf.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)  # Red
            
            opacity_tf.AddPoint(scalar_range[0], 0.0)
            opacity_tf.AddPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.2, 0.3)
            opacity_tf.AddPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.5, 0.6)
            opacity_tf.AddPoint(scalar_range[1], 0.9)
            
            gradient_opacity_tf.AddPoint(gradient_range[0], 0.3)
            gradient_opacity_tf.AddPoint(gradient_range[1], 1.0)
    
    else:
        # Manual 2D mode: user-controlled transfer function
        print("Using manual 2D transfer function")
        
        # Get normalized intensity range based on user sliders
        intensity_min = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * state.manual_intensity_min
        intensity_max = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * state.manual_intensity_max
        intensity_mid = (intensity_min + intensity_max) * 0.5
        
        # Apply color scheme
        if state.manual_color_scheme == "none":
            # No color mapping - use single neutral color
            color_tf.AddRGBPoint(scalar_range[0], 0.8, 0.8, 0.8)  # Light gray
            color_tf.AddRGBPoint(scalar_range[1], 0.8, 0.8, 0.8)  # Light gray
        elif state.manual_color_scheme == "rainbow":
            color_tf.AddRGBPoint(intensity_min, 0.0, 0.0, 1.0)  # Blue (low)
            color_tf.AddRGBPoint(intensity_mid, 0.0, 1.0, 0.0)  # Green (mid)
            color_tf.AddRGBPoint(intensity_max, 1.0, 0.0, 0.0)  # Red (high)
        elif state.manual_color_scheme == "grayscale":
            color_tf.AddRGBPoint(intensity_min, 0.0, 0.0, 0.0)  # Black
            color_tf.AddRGBPoint(intensity_max, 1.0, 1.0, 1.0)  # White
        elif state.manual_color_scheme == "warm":
            color_tf.AddRGBPoint(intensity_min, 0.5, 0.0, 0.0)  # Dark red
            color_tf.AddRGBPoint(intensity_mid, 1.0, 0.5, 0.0)  # Orange
            color_tf.AddRGBPoint(intensity_max, 1.0, 1.0, 0.0)  # Yellow
        elif state.manual_color_scheme == "cool":
            color_tf.AddRGBPoint(intensity_min, 0.0, 0.0, 0.5)  # Dark blue
            color_tf.AddRGBPoint(intensity_mid, 0.0, 0.5, 1.0)  # Light blue
            color_tf.AddRGBPoint(intensity_max, 0.0, 1.0, 1.0)  # Cyan
        
        # User-controlled opacity curve
        # Build opacity transfer function with proper handling of range boundaries
        print(f"📊 Manual TF - Intensity range slider: [{state.manual_intensity_min:.2f}, {state.manual_intensity_max:.2f}]")
        print(f"📊 Manual TF - Actual intensity values: [{intensity_min:.1f}, {intensity_mid:.1f}, {intensity_max:.1f}]")
        print(f"📊 Manual TF - Opacity: [{state.manual_opacity_low:.2f}, {state.manual_opacity_mid:.2f}, {state.manual_opacity_high:.2f}]")
        print(f"📊 Gradient range: {gradient_range}")
        
        # The opacity sliders control opacity WITHIN the selected intensity range
        # Everything outside the range is made transparent (0.0 opacity)
        
        if intensity_min > scalar_range[0]:
            # Make everything below the selected range transparent
            opacity_tf.AddPoint(scalar_range[0], 0.0)
            opacity_tf.AddPoint(intensity_min, state.manual_opacity_low)
        else:
            # User selected from the minimum, so start with low opacity
            opacity_tf.AddPoint(scalar_range[0], state.manual_opacity_low)
        
        # Apply opacity values at the three control points within the selected range
        opacity_tf.AddPoint(intensity_mid, state.manual_opacity_mid)
        opacity_tf.AddPoint(intensity_max, state.manual_opacity_high)
        
        if intensity_max < scalar_range[1]:
            # Make everything above the selected range transparent
            opacity_tf.AddPoint(scalar_range[1], 0.0)
        else:
            # User selected to the maximum, so maintain high opacity
            pass
        
        # Gradient opacity controlled by manual weight
        # IMPORTANT: Gradient opacity multiplies with scalar opacity
        # Values should generally be high (0.5-1.0) to avoid making everything transparent
        if gradient_range[1] > 0:
            # Simplified gradient opacity - keep it mostly opaque
            if state.boundary_emphasis:
                # Emphasize boundaries slightly
                gradient_opacity_tf.AddPoint(gradient_range[0], 0.8)  # High base opacity
                gradient_opacity_tf.AddPoint(gradient_range[1], 1.0)  # Full opacity at boundaries
            else:
                # Uniform gradient opacity
                gradient_opacity_tf.AddPoint(gradient_range[0], 1.0)
                gradient_opacity_tf.AddPoint(gradient_range[1], 1.0)
            print(f"📊 Gradient opacity: boundary_emphasis={state.boundary_emphasis}")
        else:
            gradient_opacity_tf.AddPoint(0.0, 1.0)
            gradient_opacity_tf.AddPoint(1.0, 1.0)
    
    # Apply transfer functions to volume property
    volume_property.SetColor(color_tf)
    volume_property.SetScalarOpacity(opacity_tf)
    volume_property.SetGradientOpacity(gradient_opacity_tf)
    
    # CRITICAL: Set scalar opacity unit distance based on dataset spacing
    # This affects how opacity accumulates along rays
    spacing = loaded_data.GetSpacing()
    avg_spacing = (spacing[0] + spacing[1] + spacing[2]) / 3.0
    volume_property.SetScalarOpacityUnitDistance(avg_spacing)
    print(f"🔧 Scalar opacity unit distance set to {avg_spacing:.2f} (based on spacing {spacing})")
    
    # Create volume if not exists
    if volume is None:
        print("🔧 Creating volume...")
        volume = vtk_lib.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        # Ensure volume is visible
        volume.SetVisibility(True)
        print(f"🔍 Volume visibility set to: {volume.GetVisibility()}")
        
        # Add to renderer if not already there
        if volume not in renderer.GetVolumes():
            renderer.AddVolume(volume)
            print(f"✅ Volume added to renderer. Total volumes in renderer: {renderer.GetVolumes().GetNumberOfItems()}")
    else:
        # Update existing volume - make sure mapper is also updated
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        volume.SetVisibility(True)
        print(f"🔄 Volume property AND mapper updated. Visibility: {volume.GetVisibility()}")
    
    # Force a render update
    renderer.Modified()
    
    print("✅ 2D transfer function applied successfully")
    print(f"🔍 Final check - Volume in renderer: {volume in renderer.GetVolumes()}")
    ctrl.view_update()


###############################
### UI EVENT HANDLERS
###############################
# RESPONSIBILITY: Mengmei He (Interactive UI)
# These @state.change decorators create reactive UI behavior
# They automatically trigger when user changes UI controls

@state.change("use_2d_tf", "tf_method")
def update_tf_mode(**kwargs):
    """
    Update when switching between manual/semi-automatic modes or enabling/disabling 2D TF.
    """
    if state.use_2d_tf and state.volume_render_enabled:
        apply_2d_transfer_function()

@state.change("clustering_method", "num_clusters", "boundary_emphasis", "gradient_opacity_weight")
def update_semi_automatic_settings(**kwargs):
    """
    Update when semi-automatic TF settings change.
    Only applies if we're in semi-automatic mode.
    """
    if state.use_2d_tf and state.volume_render_enabled and state.tf_method == "semi-automatic":
        apply_2d_transfer_function()

@state.change("manual_intensity_min", "manual_intensity_max",
              "manual_opacity_low", "manual_opacity_mid", "manual_opacity_high",
              "manual_gradient_weight", "manual_color_scheme")
def update_manual_settings(**kwargs):
    """
    Update when manual TF settings change.
    Only applies if we're in manual mode.
    """
    if state.use_2d_tf and state.volume_render_enabled and state.tf_method == "manual":
        apply_2d_transfer_function()


@state.change("manual_intensity_range")
def update_intensity_range(manual_intensity_range, **kwargs):
    """
    Handle range slider changes for intensity range.
    """
    if manual_intensity_range and len(manual_intensity_range) == 2:
        state.manual_intensity_min = manual_intensity_range[0]
        state.manual_intensity_max = manual_intensity_range[1]


@state.change("show_2d_histogram")
def toggle_histogram_display(**kwargs):
    """
    Show/hide 2D histogram visualization.
    Generates histogram image when user toggles visibility on.
    """
    if state.show_2d_histogram and joint_histogram is not None:
        if cluster_labels is not None:
            state.histogram_image = generate_histogram_image(joint_histogram, cluster_labels)
        else:
            state.histogram_image = generate_histogram_image(joint_histogram, None)
    else:
        state.histogram_image = ""


def compute_2d_histogram_action():
    """Action to compute/recompute 2D histogram."""
    if is_data_loaded():
        compute_joint_histogram(loaded_data, num_bins=128)
        if state.show_2d_histogram:
            state.histogram_image = generate_histogram_image(joint_histogram, None)
        
        # Update performance message
        state.performance_message = f"⏱️ Histogram: {state.histogram_time:.2f}s"
        state.error_message = f"✅ 2D histogram computed in {state.histogram_time:.2f}s"
    else:
        state.error_message = "Please load data first"


###############################
### DIGIMORPH DATA LOADING ###
###############################

def load_from_file_list(file_list):
    """
    Load dataset from uploaded files.
    This allows users to browse their local computer and select TIFF files.
    
    Args:
        file_list: List of uploaded file dictionaries with 'name' and 'content'
    """
    try:
        if not file_list:
            state.error_message = "No files selected"
            return False
            
        state.is_loading = True
        state.loading_message = "📂 Processing uploaded files..."
        ctrl.view_update()
        
        print(f"✓ Received {len(file_list)} files")
        
        # Filter for TIFF files and sort
        tiff_files = [f for f in file_list if f['name'].lower().endswith(('.tif', '.tiff'))]
        
        if not tiff_files:
            state.error_message = "No TIFF files found in selection"
            state.is_loading = False
            return False
        
        state.loading_message = f"✓ Found {len(tiff_files)} TIFF files"
        ctrl.view_update()
        
        # Sort files numerically
        try:
            tiff_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x['name']))))
        except:
            tiff_files.sort(key=lambda x: x['name'])
        
        print(f"✓ Sorted {len(tiff_files)} TIFF files")
        
        # Create volume from file stack
        global loaded_data, reader, dim, joint_histogram, cluster_labels, cluster_centers
        
        if len(tiff_files) > 1:
            volume_reader = vtk_lib.vtkImageAppend()
            volume_reader.SetAppendAxis(2)
            
            total_files = len(tiff_files)
            state.loading_message = f"📥 Loading {total_files} images..."
            ctrl.view_update()
            
            for i, file_data in enumerate(tiff_files):
                if i % 50 == 0:
                    progress_percent = int((i / total_files) * 100)
                    state.loading_message = f"📥 Loading images: {i+1}/{total_files} ({progress_percent}%)"
                    ctrl.view_update()
                
                # Create temporary file to load with VTK
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                    tmp_file.write(file_data['content'])
                    tmp_path = tmp_file.name
                
                img_reader = vtk_lib.vtkTIFFReader()
                img_reader.SetFileName(tmp_path)
                img_reader.Update()
                volume_reader.AddInputData(img_reader.GetOutput())
                
                # Clean up temp file
                import os
                os.unlink(tmp_path)
            
            volume_reader.Update()
            volume_data = volume_reader.GetOutput()
            print(f"✓ Created 3D volume with {total_files} slices: {volume_data.GetDimensions()}")
            image_data_reader = volume_reader
            
        else:
            # Single file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tmp_file.write(tiff_files[0]['content'])
                tmp_path = tmp_file.name
            
            tiff_reader = vtk_lib.vtkTIFFReader()
            tiff_reader.SetFileName(tmp_path)
            tiff_reader.Update()
            volume_data = tiff_reader.GetOutput()
            image_data_reader = tiff_reader
            
            import os
            os.unlink(tmp_path)
        
        # Set global variables
        loaded_data = volume_data
        reader = None
        dim = volume_data.GetDimensions()
        
        scalar_range = volume_data.GetScalarRange()
        state.scalar_range = list(scalar_range)
        print(f"✓ Scalar range: {scalar_range}")
        
        # Reset 2D TF data
        joint_histogram = None
        cluster_labels = None
        cluster_centers = None
        
        # Setup visualization
        state.loading_message = "⚙️ Setting up visualization..."
        ctrl.view_update()
        
        update_cutplane_slider_ranges()
        update_iso_slider_range()
        setup_outline()
        setup_cut_planes()
        setup_isosurface()
        
        # Position camera
        state.loading_message = "📷 Positioning camera..."
        ctrl.view_update()
        
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        bounds = loaded_data.GetBounds()
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]
        distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2
        camera.SetPosition(center[0] + distance, center[1] + distance, center[2] + distance)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        renderer.ResetCamera()
        render_window.Render()
        ctrl.view_update()
        
        state.error_message = "✅ Dataset loaded successfully!"
        state.is_loading = False
        state.loading_message = ""
        print("Dataset loaded successfully from uploaded files!")
        return True
        
    except Exception as e:
        state.error_message = f"Error loading files: {str(e)}"
        print(f"❌ Error loading files: {e}")
        import traceback
        traceback.print_exc()
        state.is_loading = False
        state.loading_message = ""
        return False

def load_chameleon_dataset():
    """
    Load the Chameleon dataset from the digimorph data folder.
    Handles the 8-bit image stack from digimorph.org.
    
    Process:
    1. Check for dataset directory (Chamaeleo_calyptratus/8bit)
    2. Scan for image files (PNG, TIFF, JPG formats)
    3. Sort images numerically for correct Z-ordering
    4. Use VTK image readers to load each slice
    5. Assemble slices into 3D volume using vtkImageAppend
    
    Returns:
        bool: True if successful, False on error
    """
    try:
        # Set loading state
        state.is_loading = True
        state.loading_message = "🔍 Checking data path..."
        ctrl.view_update()
        
        data_path = Path(state.digimorph_data_path)
        
        if not data_path.exists():
            state.error_message = f"Dataset not found at: {data_path}"
            print(f"❌ Dataset path not found: {data_path}")
            print("Please check the path and try again")
            state.is_loading = False
            return False
        
        state.loading_message = "📂 Scanning for image files..."
        ctrl.view_update()
        
        # Look for .tif files in the directory
        image_files = glob.glob(str(data_path / '*.tif'))
        
        if not image_files:
            state.error_message = "No image files found in the dataset directory"
            print(f"❌ No image files found in: {data_path}")
            state.is_loading = False
            return False
        
        print(f"✓ Found {len(image_files)} images in dataset")
        state.loading_message = f"✓ Found {len(image_files)} images"
        ctrl.view_update()
        
        # Sort filenames numerically for correct Z-ordering
        state.loading_message = "🔢 Sorting images..."
        ctrl.view_update()
        
        try:
            image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, Path(x).stem))))
        except:
            image_files.sort()  # Fallback to alphabetical
        
        print(f"✓ Sorted {len(image_files)} TIFF images")
        
        # Create a volume from the image sequence
        if len(image_files) > 1:
            # Multiple images - create 3D volume
            volume_reader = vtk_lib.vtkImageAppend()
            volume_reader.SetAppendAxis(2)  # Append along Z-axis
            
            # Load all images
            total_images = len(image_files)
            print(f"Found {total_images} TIFF images to load...")
            state.loading_message = f"📥 Loading {total_images} images..."
            ctrl.view_update()
            
            for i, img_file in enumerate(image_files):
                # Update progress every 50 images
                if i % 50 == 0:
                    progress_percent = int((i / total_images) * 100)
                    state.loading_message = f"📥 Loading images: {i+1}/{total_images} ({progress_percent}%)"
                    ctrl.view_update()
                    print(f"Loading image {i+1}/{total_images}...")
                    
                img_reader = vtk_lib.vtkTIFFReader()
                img_reader.SetFileName(img_file)
                img_reader.Update()
                volume_reader.AddInputData(img_reader.GetOutput())
            
            volume_reader.Update()
            volume_data = volume_reader.GetOutput()
            print(f"✓ Created 3D volume with {total_images} slices: {volume_data.GetDimensions()}")
            
            # Store the volume append reader for later use
            image_data_reader = volume_reader
            
        else:
            # Single image - treat as 2D slice
            tiff_reader = vtk_lib.vtkTIFFReader()
            tiff_reader.SetFileName(image_files[0])
            tiff_reader.Update()
            volume_data = tiff_reader.GetOutput()
            print("✓ Loaded single 2D image")
            image_data_reader = tiff_reader
        
        # Set global variables
        global loaded_data, reader, dim
        loaded_data = volume_data
        reader = None  # Don't store reader for image stacks - will cause issues with outline
        dim = volume_data.GetDimensions()
        
        # Set scalar range for the volume
        scalar_range = volume_data.GetScalarRange()
        state.scalar_range = list(scalar_range)
        print(f"✓ Scalar range: {scalar_range}")
        
        # Reset 2D TF data
        global joint_histogram, cluster_labels, cluster_centers
        joint_histogram = None
        cluster_labels = None
        cluster_centers = None
        
        # Update UI ranges
        state.loading_message = "⚙️ Setting up visualization..."
        ctrl.view_update()
        
        update_cutplane_slider_ranges()
        update_iso_slider_range()
        setup_outline()
        setup_cut_planes()
        setup_isosurface()
        # Don't setup volume rendering automatically - wait for user to enable it
        
        # Reset camera with proper positioning for volume
        state.loading_message = "📷 Positioning camera..."
        ctrl.view_update()
        
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        
        # Position camera to get a good 3D view
        # Calculate center of volume
        bounds = loaded_data.GetBounds()
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]
        
        # Set camera position for an angled view
        distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2
        camera.SetPosition(center[0] + distance, center[1] + distance, center[2] + distance)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        
        renderer.ResetCamera()
        render_window.Render()
        ctrl.view_update()
        
        state.error_message = "✅ Dataset loaded successfully!"
        state.is_loading = False
        state.loading_message = ""
        print("Dataset loaded successfully!")
        return True
        
    except Exception as e:
        state.error_message = f"Error loading dataset: {str(e)}"
        print(f"❌ Error loading dataset: {e}")
        state.is_loading = False
        state.loading_message = ""
        import traceback
        traceback.print_exc()
        return False

def auto_load_chameleon_on_startup():
    """Auto-load chameleon dataset if available."""
    if state.auto_load_chameleon:
        print("🦎 Auto-loading Chameleon dataset...")
        load_chameleon_dataset()

###############################
### DATASET LOADING SECTION ###
###############################

@state.change("file_type")
def update_file_inputs(file_type, **kwargs):
    if file_type == "vtk":
        state.show_vtk_input = True
        state.show_mhd_input = False
        state.show_raw_input = False
    elif file_type == "mhd_raw":
        state.show_vtk_input = False
        state.show_mhd_input = True
        state.show_raw_input = True

@state.change("vtk_file")
def update_vtk_file(vtk_file, **kwargs):
    if state.file_type == "vtk":
        update_file(vtk_file)

@state.change("mhd_file")
@state.change("raw_file")
def update_mhd_raw_files(**kwargs):
    if state.file_type == "mhd_raw" and state.mhd_file and state.raw_file:
        update_file({"mhd": state.mhd_file, "raw": state.raw_file})

def update_file(file_data, **kwargs):
    global loaded_data, reader, outlineData, mapOutline, outlineActor, dim
    global joint_histogram, cluster_labels, cluster_centers

    if not file_data:
        state.error_message = "Please select a file."
        return
    
    try:
        if state.file_type == "vtk":
            reader = vtkDataSetReader()
            if isinstance(file_data, dict) and 'content' in file_data:
                reader.ReadFromInputStringOn()
                reader.SetInputString(file_data['content'])
            else:
                raise ValueError("Invalid file_data format for VTK")
        elif state.file_type == "mhd_raw":
            reader = vtkMetaImageReader()
            if isinstance(file_data, dict) and 'mhd' in file_data and 'raw' in file_data:
                with tempfile.TemporaryDirectory() as temp_dir:
                    mhd_path = os.path.join(temp_dir, "temp.mhd")
                    raw_path = os.path.join(temp_dir, "temp.raw")
                    
                    mhd_content = file_data['mhd']['content']
                    if isinstance(mhd_content, bytes):
                        mhd_content = mhd_content.decode('utf-8')
                    
                    mhd_lines = mhd_content.splitlines()
                    for i, line in enumerate(mhd_lines):
                        if line.startswith("ElementDataFile"):
                            mhd_lines[i] = "ElementDataFile = temp.raw"
                            break
                    mhd_content = "\n".join(mhd_lines)
                    
                    with open(mhd_path, 'w', encoding='utf-8') as mhd_file:
                        mhd_file.write(mhd_content)
                    
                    raw_content = file_data['raw']['content']
                    with open(raw_path, 'wb') as raw_file:
                        if isinstance(raw_content, str):
                            raw_file.write(raw_content.encode('latin-1'))
                        else:
                            raw_file.write(raw_content)
                    
                    reader.SetFileName(mhd_path)
                    reader.Update()
                    
                    if reader.GetOutput() is None or reader.GetOutput().GetNumberOfPoints() == 0:
                        raise ValueError("Failed to read the MHD/RAW file.")
            else:
                raise ValueError("Invalid file_data format for MHD/RAW")
        
        reader.Update()
        loaded_data = reader.GetOutput()
        
        if not loaded_data or loaded_data.GetNumberOfPoints() == 0:
            state.error_message = "Failed to load the file."
            print(state.error_message)
            return
        
        if state.file_type == "vtk":
            loaded_data.GetPointData().SetActiveScalars('s')

        scalar_range = loaded_data.GetScalarRange()
        dim = reader.GetOutput().GetDimensions()
        
        print(f"Data loaded: {loaded_data.GetNumberOfPoints()} points")
        print(f"Dimensions: {dim}")
        print(f"Scalar range: {scalar_range}")
        
        # Reset 2D TF data
        joint_histogram = None
        cluster_labels = None
        cluster_centers = None
        
        update_cutplane_slider_ranges()
        update_iso_slider_range()
        setup_outline()
        setup_cut_planes()
        setup_isosurface()
        setup_volume_rendering()
        
        state.scalar_range = list(scalar_range)
        update_scalar_range(state.scalar_range)
        
        renderer.ResetCamera()
        render_window.Render()
        reset_camera()
        
        state.error_message = "Data loaded successfully"
    except Exception as e:
        state.error_message = f"Error loading file: {str(e)}"
        print(state.error_message)
        import traceback
        traceback.print_exc()

def setup_outline():
    global outlineData, mapOutline, outlineActor
    renderer.RemoveActor(outlineActor)
    # Use loaded_data directly since reader may not exist for image stacks
    outlineData.SetInputData(loaded_data)
    outlineData.Update()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())
    mapOutline.Update()
    outlineActor.SetMapper(mapOutline)
    colors = vtk_lib.vtkNamedColors()
    outlineActor.GetProperty().SetColor(colors.GetColor3d("White"))
    outlineActor.GetProperty().SetLineWidth(2.)
    renderer.AddActor(outlineActor)

##########################
### CUT PLANES SECTION ###
##########################

def setup_cut_planes():
    global xy_plane_actor, xz_plane_actor, yz_plane_actor, plane_Colors, lut

    renderer.RemoveActor(xy_plane_actor)
    renderer.RemoveActor(xz_plane_actor)
    renderer.RemoveActor(yz_plane_actor)

    plane_Colors.SetInputData(loaded_data)
    make_lut(state.color_scheme, lut)
    plane_Colors.SetLookupTable(lut)
    plane_Colors.Update()

    xy_plane_actor = vtk_lib.vtkImageActor()
    xz_plane_actor = vtk_lib.vtkImageActor()
    yz_plane_actor = vtk_lib.vtkImageActor()

    xy_plane_actor.GetMapper().SetInputConnection(plane_Colors.GetOutputPort())
    xz_plane_actor.GetMapper().SetInputConnection(plane_Colors.GetOutputPort())
    yz_plane_actor.GetMapper().SetInputConnection(plane_Colors.GetOutputPort())
    
    xy_plane_actor.VisibilityOff()
    renderer.AddActor(xy_plane_actor)
    renderer.AddActor(xz_plane_actor)
    renderer.AddActor(yz_plane_actor)

    update_cut_planes(state.z_slider, state.y_slider, state.x_slider)
    update_cut_plane_visibility()

def update_cut_planes(z_value, y_value, x_value):
    global xy_plane_actor, xz_plane_actor, yz_plane_actor, dim, loaded_data

    if not is_data_loaded():
        return
    
    xy_plane_actor.SetDisplayExtent(0, dim[0]-1, 0, dim[1]-1, z_value, z_value)
    xz_plane_actor.SetDisplayExtent(0, dim[0]-1, y_value, y_value, 0, dim[2]-1)
    yz_plane_actor.SetDisplayExtent(x_value, x_value, 0, dim[1]-1, 0, dim[2]-1)

    render_window.Render()
    ctrl.view_update()

def update_cutplane_slider_ranges():
    global dim
    if is_data_loaded():
        dim = loaded_data.GetDimensions()
        state.z_slider_max = dim[2]-1
        state.y_slider_max = dim[1]-1
        state.x_slider_max = dim[0]-1
    else:
        state.z_slider_max = 100
        state.y_slider_max = 100
        state.x_slider_max = 100

def update_cut_plane_visibility():
    global xy_plane_actor, xz_plane_actor, yz_plane_actor

    xy_plane_actor.SetVisibility(state.xy_plane_visible)
    xz_plane_actor.SetVisibility(state.xz_plane_visible)
    yz_plane_actor.SetVisibility(state.yz_plane_visible)
    ctrl.view_update()

def update_scalar_range(scalar_range):
    global plane_Colors, lut, loaded_data
    if not is_data_loaded():
        return

    lut.SetTableRange(scalar_range[0], scalar_range[1])
    lut.Build()
    plane_Colors.SetLookupTable(lut)
    plane_Colors.Update()
    xy_plane_actor.GetMapper().Update()

    ctrl.view_update()

@state.change("scalar_range")
def on_scalar_range_change(scalar_range, **kwargs):
    update_scalar_range(scalar_range)

@state.change("xy_plane_visible")
def update_xy_plane_visibility(xy_plane_visible, **kwargs):
    xy_plane_actor.SetVisibility(xy_plane_visible)
    ctrl.view_update()

@state.change("xz_plane_visible")
def update_xz_plane_visibility(xz_plane_visible, **kwargs):
    xz_plane_actor.SetVisibility(xz_plane_visible)
    ctrl.view_update()

@state.change("yz_plane_visible")
def update_yz_plane_visibility(yz_plane_visible, **kwargs):
    yz_plane_actor.SetVisibility(yz_plane_visible)
    ctrl.view_update()

@state.change("z_slider")    
@state.change("y_slider")  
@state.change("x_slider")
def on_zslider_change(z_slider, y_slider, x_slider, **kwargs):
    update_cut_planes(z_slider, y_slider, x_slider)

@state.change("color_scheme")
def on_color_scheme_change(color_scheme, **kwargs):
    update_color_scheme(color_scheme)
    
def update_color_scheme(color_scheme, **kwargs):
    global lut, plane_Colors
    if not is_data_loaded():
        state.error_message = "Please load a data file first."
        return
    
    make_lut(color_scheme, lut)
    lut.Build()
    
    plane_Colors.SetLookupTable(lut)
    plane_Colors.Update()
    
    xy_plane_actor.GetMapper().Update()
    xz_plane_actor.GetMapper().Update()
    yz_plane_actor.GetMapper().Update()
    
    ctrl.view_update()

def make_lut(color_scheme, lut):
    nc = 256
    lut.SetNumberOfTableValues(nc)
    lut.Build()

    if color_scheme == "rainbow":
        sMin, sMax = 0., 1.
        hsv = [0.0, 1.0, 1.0]
        for i in range(nc):
            s = float(i) / nc
            hsv[0] = 240. - 240. * (s - sMin) / (sMax - sMin)
            rgb = hsv_to_rgb(hsv)
            lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)

    elif color_scheme == "bwr":
        for i in range(nc):
            s = float(i) / nc
            if s <= 0.5:
                r = 2 * s
                g = 2 * s
                b = 1
            else:
                r = 1
                g = 2 - 2 * s
                b = 2 - 2 * s
            lut.SetTableValue(i, r, g, b, 1.0)
        
    elif color_scheme == "heatmap":
        for i in range(nc):
            s = float(i) / nc
            if s <= 1/3:
                r = 3 * s
                g = 0
                b = 0
            elif s <= 2/3:
                r = 1 
                g = 3 * s - 1
                b = 0
            else:
                r = 1
                g = 1
                b = 3 * s - 2
            lut.SetTableValue(i, r, g, b, 1.0)
        
    elif color_scheme == "gray":
        for i in range(nc):
            s = float(i) / nc
            r = g = b = s
            lut.SetTableValue(i, r, g, b, 1.0)

###########################
### ISO-SURFACE SECTION ###
###########################

def setup_isosurface():
    global surface_actor
    remove_surface_actor()
    extract_isosurface(state.iso_value, state.iso_alpha)

def extract_isosurface(iso_value, opacity):
    global surface_actor
    if not is_data_loaded():
        return

    if surface_actor is not None:
        renderer.RemoveActor(surface_actor)

    isoSurfExtractor = vtk_lib.vtkMarchingCubes() 
    isoSurfExtractor.SetInputData(loaded_data)
    isoSurfExtractor.SetValue(0, iso_value)

    isoSurfStripper = vtk_lib.vtkStripper()
    isoSurfStripper.SetInputConnection(isoSurfExtractor.GetOutputPort())
    isoSurfStripper.Update()

    isoSurfMapper = vtk_lib.vtkPolyDataMapper()
    isoSurfMapper.SetInputConnection(isoSurfStripper.GetOutputPort())
    isoSurfMapper.ScalarVisibilityOff()

    surface_actor = vtk_lib.vtkActor()
    surface_actor.SetMapper(isoSurfMapper)
    surface_actor.GetProperty().SetDiffuseColor([0.5, 0.5, 0.5])
    surface_actor.GetProperty().SetSpecular(.3)
    surface_actor.GetProperty().SetSpecularPower(20)
    surface_actor.GetProperty().SetOpacity(opacity)

    renderer.AddActor(surface_actor)
    update_surface_visibility()

def remove_surface_actor():
    global surface_actor
    renderer.RemoveActor(surface_actor)

def update_surface_visibility():
    if surface_actor:
        surface_actor.SetVisibility(state.isosurface_visible)

@state.change("isosurface_visible")
def update_isosurface_visibility(isosurface_visible, **kwargs):
    update_surface_visibility()
    ctrl.view_update()

@state.change("iso_value")
def update_iso_contour(iso_value, **kwargs):
    if not is_data_loaded():
        state.error_message = "Please load a data file first."
        return
    
    remove_surface_actor()
    extract_isosurface(iso_value, state.iso_alpha)
    ctrl.view_update()

@state.change("iso_alpha")
def update_transparency(iso_alpha, **kwargs):
    if not is_data_loaded() or not surface_actor:
        return
    surface_actor.GetProperty().SetOpacity(iso_alpha)
    ctrl.view_update()

def update_iso_slider_range():
    if is_data_loaded():
        scalar_range = loaded_data.GetScalarRange() 
        state.iso_slider_min = scalar_range[0]
        state.iso_slider_max = scalar_range[1]
    else:
        state.iso_slider_min = 0
        state.iso_slider_max = 100
        state.iso_value = 0

################################
### VOLUME RENDERING SECTION ###
################################

def apply_transfer_function():
    """Apply 1D transfer function (original implementation)."""
    global volume, volume_property, volume_mapper, renderer, loaded_data

    if state.edit_mode:
        new_transfer_function = csv_to_transfer_function(state.transfer_function_csv)
        if new_transfer_function:
            state.transfer_function = new_transfer_function
        else:
            state.error_message = "Failed to parse CSV."
            return

    if not is_data_loaded():
        state.error_message = "Please load a data file first."
        return
    
    print("Applying 1D transfer function...")
    
    # Get actual scalar range from data
    scalar_range = loaded_data.GetScalarRange()
    print(f"Data scalar range: {scalar_range}")
    
    volume_Color = vtk_lib.vtkColorTransferFunction()
    volume_Opacity = vtk_lib.vtkPiecewiseFunction()

    # Check if we should use the default transfer function or create a better one for the data
    tf_range = [point['scalar'] for point in state.transfer_function]
    tf_min, tf_max = min(tf_range), max(tf_range)
    
    # If transfer function doesn't match data range, create a better default
    if tf_max < scalar_range[1] * 0.5 or tf_min > scalar_range[0]:
        print(f"Creating default transfer function for scalar range {scalar_range}")
        # Create a good default for CT-like data
        volume_Color.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)  # Black for background
        volume_Color.AddRGBPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.2, 0.5, 0.3, 0.2)  # Brown (bone)
        volume_Color.AddRGBPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.5, 0.9, 0.9, 0.7)  # Light (tissue)
        volume_Color.AddRGBPoint(scalar_range[1], 1.0, 1.0, 1.0)  # White (dense)
        
        # Better opacity for showing internal structure
        volume_Opacity.AddPoint(scalar_range[0], 0.0)  # Transparent background
        volume_Opacity.AddPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.15, 0.0)  # Keep low values transparent
        volume_Opacity.AddPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.25, 0.05)  # Soft tissue starts
        volume_Opacity.AddPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.4, 0.2)  # Tissue visible
        volume_Opacity.AddPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.6, 0.4)  # Dense tissue
        volume_Opacity.AddPoint(scalar_range[1], 0.8)  # Bone/dense structures
    else:
        # Use the transfer function from state
        for point in state.transfer_function:
            scalar = point['scalar']
            r = point['r']
            g = point['g']
            b = point['b']
            opacity = point['opacity']

            volume_Color.AddRGBPoint(scalar, r, g, b)
            volume_Opacity.AddPoint(scalar, opacity)
    
    volume_mapper = vtk_lib.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(loaded_data)

    volume_GradientOpacity = vtk_lib.vtkPiecewiseFunction()
    
    # Use adaptive gradient opacity based on data range
    # For CT-like data (0-255 range), use appropriate gradient values
    max_gradient = scalar_range[1] - scalar_range[0]
    volume_GradientOpacity.AddPoint(0, 0.0)
    volume_GradientOpacity.AddPoint(max_gradient * 0.1, 0.1)
    volume_GradientOpacity.AddPoint(max_gradient * 0.3, 0.5)
    volume_GradientOpacity.AddPoint(max_gradient * 0.5, 0.8)
    volume_GradientOpacity.AddPoint(max_gradient, 1.0)

    volume_property = vtk_lib.vtkVolumeProperty()
    volume_property.SetColor(volume_Color)
    volume_property.SetScalarOpacity(volume_Opacity)
    volume_property.SetGradientOpacity(volume_GradientOpacity)
    volume_property.SetInterpolationTypeToLinear()

    if state.enable_gradient_shading:
        volume_property.ShadeOn()
        volume_property.SetAmbient(state.ambient)
        volume_property.SetDiffuse(state.diffuse)
        volume_property.SetSpecular(state.specular)
        volume_property.SetSpecularPower(state.specular_power)
    else:
        volume_property.ShadeOff()

    volume = vtk_lib.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    renderer.AddVolume(volume)
    renderer.ResetCamera()
    ctrl.view_update()

def setup_volume_rendering():
    """
    Initialize volume rendering pipeline with appropriate transfer function.
    Determines whether to use 1D or 2D transfer functions and delegates
    to the appropriate function.
    """
    global volume_mapper, volume, volume_property, loaded_data

    if not is_data_loaded():
        print("No data loaded. Cannot set up volume rendering.")
        state.error_message = "No data loaded. Please load data first."
        return
    
    print(f"✓ Setting up volume rendering with 2D TF: {state.use_2d_tf}")
    print(f"✓ Data dimensions: {loaded_data.GetDimensions()}")
    print(f"✓ Data scalar range: {loaded_data.GetScalarRange()}")
    
    # Decide which transfer function to apply
    if state.use_2d_tf:
        print("🎨 Using 2D Transfer Function")
        # Check if histogram has been computed
        if joint_histogram is None:
            print("⚠️ 2D histogram not computed yet. Please click 'COMPUTE 2D HISTOGRAM' button first.")
            state.error_message = "Please compute 2D histogram first using the button."
            # Apply a basic transfer function as fallback
            apply_transfer_function()
            return
        apply_2d_transfer_function()
    else:
        print("🎨 Using 1D Transfer Function")
        apply_transfer_function()

@state.change("volume_render_enabled")
def update_volume_rendering(volume_render_enabled, **kwargs):
    global renderer, volume

    print(f"🎛️ Volume rendering toggle: {volume_render_enabled}")
    
    if not is_data_loaded():
        state.error_message = "Please load a data file first."
        print("❌ No data loaded for volume rendering")
        return
    
    print("✓ Data is loaded, proceeding with volume rendering setup")
    
    if volume_render_enabled:
        if volume is None or volume not in renderer.GetVolumes():
            print("🔧 Setting up volume rendering...")
            setup_volume_rendering()
            if volume and volume not in renderer.GetVolumes():
                renderer.AddVolume(volume)
                print("✅ Volume added to renderer")
        else:
            print("🔄 Re-applying transfer function...")
            # Re-apply transfer function
            if state.use_2d_tf:
                apply_2d_transfer_function()
            else:
                apply_transfer_function()
        
        # Verify volume is visible
        print(f"🔍 Volume exists: {volume is not None}")
        if volume:
            print(f"🔍 Volume in renderer: {volume in renderer.GetVolumes()}")
            print(f"🔍 Volume visibility: {volume.GetVisibility()}")
    else:
        print("🚫 Disabling volume rendering...")
        if volume and volume in renderer.GetVolumes():
            renderer.RemoveVolume(volume)
            print("✅ Volume removed from renderer")

    renderer.ResetCamera()
    ctrl.view_update()
    print("🎨 Volume rendering update complete")

@state.change("enable_gradient_shading", "ambient", "diffuse", "specular", "specular_power")
def update_shading_parameters(**kwargs):
    """Update shading parameters."""
    if volume_property and state.volume_render_enabled:
        if state.enable_gradient_shading:
            volume_property.ShadeOn()
            volume_property.SetAmbient(state.ambient)
            volume_property.SetDiffuse(state.diffuse)
            volume_property.SetSpecular(state.specular)
            volume_property.SetSpecularPower(state.specular_power)
        else:
            volume_property.ShadeOff()
        ctrl.view_update()

def transfer_function_to_csv():
    if state.transfer_function is None:
        return ""
    csv_lines = ["scalar,r,g,b,opacity"]
    for point in state.transfer_function:
        csv_lines.append(f"{point['scalar']},{point['r']},{point['g']},{point['b']},{point['opacity']}")
    return "\n".join(csv_lines)

def csv_to_transfer_function(csv_text):
    lines = csv_text.strip().split("\n")[1:]
    new_transfer_function = []
    for line in lines:
        values = line.split(",")
        if len(values) == 5:
            new_transfer_function.append({
                "scalar": float(values[0]),
                "r": float(values[1]),
                "g": float(values[2]),
                "b": float(values[3]),
                "opacity": float(values[4])
            })
    return new_transfer_function

def toggle_transfer_function_dialog():
    state.show_transfer_function_dialog = not state.show_transfer_function_dialog

@state.change("edit_mode")
def update_csv_on_edit_mode_change(edit_mode, **kwargs):
    if edit_mode:
        state.transfer_function_csv = transfer_function_to_csv()

@state.change("selected_transfer_function")
def update_selected_transfer_function(selected_transfer_function, **kwargs):
    state.transfer_function = color_transfer_functions[selected_transfer_function]
    state.table_key = random.random()
    if not state.use_2d_tf:
        apply_transfer_function()

@state.change("transfer_function")
def update_transfer_function(transfer_function, **kwargs):
    print("Transfer function updated:", transfer_function)
    state.table_key = random.random()

@state.change("show_transfer_function_dialog")
def update_transfer_function_csv(show_transfer_function_dialog, **kwargs):
    if show_transfer_function_dialog:
        state.transfer_function_csv = transfer_function_to_csv()

########################
### UI SETUP SECTION ###
########################

def setup_ui():
    """
    Create the interactive web-based UI using Trame and Vuetify.
    Builds the complete user interface for the 2D transfer function visualization system.
    """
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("2D Transfer Functions for Volume Rendering - Final Project")

        with layout.toolbar:
            vuetify.VToolbarTitle("DVR with 2D Transfer Functions")
            vuetify.VSpacer()
            vuetify.VBtn(
                "Screenshot",
                prepend_icon="mdi-camera",
                click=save_screenshot,
                color="success",
                variant="outlined",
                classes="mr-2"
            )
            vuetify.VBtn(
                "Reset Parameters",
                prepend_icon="mdi-refresh",
                click=reset_parameters,
                color="warning",
                variant="outlined",
                classes="mr-2"
            )
            vuetify.VBtn("Reset Camera", prepend_icon="mdi-crop-free", click=reset_camera)

        with layout.drawer:
            with vuetify.VContainer(fluid=True):
                # File loading
                vuetify.VCardSubtitle(
                    "1. Data Loading", 
                    classes="text-h6 pa-2 font-weight-bold text-primary"
                )
                
                # Folder browser with custom HTML for directory selection
                # Data path input
                vuetify.VTextField(
                    v_model=("digimorph_data_path", "./Chamaeleo_calyptratus/8bit"),
                    label="Data Folder Path",
                    density="compact",
                    prepend_icon="mdi-folder-open",
                    hint="Path to folder containing TIFF image stack",
                    persistent_hint=True,
                    clearable=True,
                    disabled=("is_loading", False),
                )
                
                # Loading indicator - shown when is_loading is True
                with vuetify.VAlert(
                    v_if=("is_loading",),
                    type="info",
                    density="compact",
                    classes="my-2",
                    prominent=True,
                    border="start",
                ):
                    with vuetify.VRow(align="center", dense=True):
                        with vuetify.VCol(cols="auto"):
                            vuetify.VProgressCircular(indeterminate=True, size=40, width=4, color="primary")
                        with vuetify.VCol():
                            vuetify.VCardText("{{ loading_message }}", classes="pa-0 text-body-1 font-weight-medium")
                
                # Load button
                with vuetify.VRow(dense=True, classes="mt-2"):
                    with vuetify.VCol(cols=8):
                        vuetify.VBtn(
                            "Load Dataset",
                            color="primary",
                            variant="outlined",
                            prepend_icon="mdi-database",
                            click=lambda: load_chameleon_dataset(),
                            block=True,
                            density="comfortable",
                            disabled=("is_loading", False),
                        )
                
                vuetify.VDivider(classes="my-4", thickness=3, color="primary")
                

                # 2D Histogram Controls - Right below volume rendering
                vuetify.VCardSubtitle(
                    "2. 2D Histogram", 
                    classes="text-h6 pa-2 font-weight-bold text-primary"
                )
                with vuetify.VRow(dense=True, classes="mt-2 mb-2"):
                    with vuetify.VCol(cols=12):
                        vuetify.VBtn(
                            "Compute 2D Histogram",
                            click=compute_2d_histogram_action,
                            prepend_icon="mdi-chart-histogram",
                            color="accent",
                            variant="outlined",
                            block=True,
                            density="comfortable",
                        )
                
                vuetify.VCheckbox(
                    v_model=("show_2d_histogram", False),
                    label="Show 2D Histogram",
                    density="compact",
                    color="secondary",
                    disabled=not MATPLOTLIB_AVAILABLE,
                )
                
                # Performance metrics display
                with vuetify.VAlert(
                    v_if=("performance_message",),
                    type="success",
                    density="compact",
                    classes="mt-2",
                    variant="tonal",
                ):
                    html.Div("{{ performance_message }}", classes="text-caption")
                
                
                vuetify.VDivider(classes="my-4", thickness=3, color="primary")
                # Volume Rendering Controls
                vuetify.VCardSubtitle(
                    "3. Volume Rendering", 
                    classes="text-h6 pa-2 font-weight-bold text-primary"
                )
                vuetify.VCheckbox(
                    v_model=("volume_render_enabled", False),
                    label="Enable Volume Rendering",
                    color="primary",
                    density="compact",
                )
                vuetify.VDivider(classes="my-4", thickness=3, color="primary")
                
                # 2D Transfer Function Controls
                vuetify.VCardSubtitle(
                    "4. 2D Transfer Function Settings", 
                    classes="text-h6 pa-2 font-weight-bold text-primary"
                )
                
                with vuetify.VRadioGroup(
                    v_model=("tf_method", "manual"),
                    density="compact",
                ):
                    vuetify.VRadio(label="Manual", value="manual")
                    vuetify.VRadio(label="Semi-Automatic", value="semi-automatic")
                
                # Manual 2D TF Controls
                with vuetify.VCard(v_show="tf_method === 'manual'", flat=True, classes="mt-2"):
                    vuetify.VCardSubtitle("Manual Controls", classes="text-subtitle-2 pa-1")
                    
                    # Intensity range controls
                    vuetify.VRangeSlider(
                        v_model=("manual_intensity_range", [0.0, 1.0]),
                        label="Intensity Range",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        thumb_label="always",
                        density="compact",
                    )
                    
                    # Opacity controls
                    vuetify.VSlider(
                        v_model=("manual_opacity_low", 0.2),
                        label="Opacity(Low Intensity)",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        thumb_label="always",
                        density="compact",
                    )
                    vuetify.VSlider(
                        v_model=("manual_opacity_mid", 0.6),
                        label="Opacity (Mid Intensity)",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        thumb_label="always",
                        density="compact",
                    )
                    vuetify.VSlider(
                        v_model=("manual_opacity_high", 0.9),
                        label="Opacity (High Intensity)",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        thumb_label="always",
                        density="compact",
                    )
                    
                    # Color scheme selector
                    vuetify.VSelect(
                        v_model=("manual_color_scheme", "rainbow"),
                        items=(
                            "color_scheme_options",
                            [
                                {"title": "Rainbow", "value": "rainbow"},
                                {"title": "Grayscale", "value": "grayscale"},
                                {"title": "Warm (Red-Orange-Yellow)", "value": "warm"},
                                {"title": "Cool (Blue-Cyan)", "value": "cool"},
                            ],
                        ),
                        label="Color Scheme",
                        density="compact",
                        variant="outlined",
                    )
                
                # Semi-Automatic Controls
                with vuetify.VCard(v_show="tf_method === 'semi-automatic'", flat=True, classes="mt-2"):
                    vuetify.VSelect(
                        v_model=("clustering_method", "kmeans"),
                        items=(
                            "clustering_options",
                            [
                                {"title": "K-Means", "value": "kmeans"},
                                {"title": "Mean-Shift", "value": "meanshift"},
                            ],
                        ),
                        label="Clustering Method",
                        density="compact",
                        variant="outlined",
                    )
                    vuetify.VSlider(
                        v_model=("num_clusters", 3),
                        label="Number of Clusters",
                        min=2,
                        max=10,
                        step=1,
                        thumb_label="always",
                        density="compact",
                        v_show="clustering_method === 'kmeans'",
                    )
                
                # Gradient Controls
                vuetify.VDivider(classes="my-4", thickness=3, color="primary")
                vuetify.VCardSubtitle(
                    "5.Gradient Parameters", 
                    classes="text-h6 pa-2 font-weight-bold text-primary"
                )
                vuetify.VCheckbox(
                    v_model=("boundary_emphasis", True),
                    label="Emphasize Boundaries (Gradient)",
                    density="compact",
                    color="primary",
                )
                vuetify.VCheckbox(
                    v_model=("enable_gradient_shading", True),
                    label="Enable Gradient Shading",
                    density="compact",
                    color="primary",
                )
                
                vuetify.VDivider(classes="my-4", thickness=3, color="primary")
                
                # Iso-surface
                vuetify.VCardSubtitle(
                    "6. Iso-Surface (Optional)", 
                    classes="text-h6 pa-2 font-weight-bold text-primary"
                )
                with vuetify.VRow(align="center", dense=True):
                    with vuetify.VCol(cols=1):
                        vuetify.VCheckbox(v_model=("isosurface_visible", False))
                    with vuetify.VCol(cols=11):
                        vuetify.VSlider(
                            v_model=("iso_value", 0),
                            min=("iso_slider_min", 0),
                            max=("iso_slider_max", 100),
                            step=0.1,
                            label="Iso-Surface Value",
                            thumb_label="always",
                            density="compact",
                        )

                vuetify.VDivider(classes="my-4", thickness=3, color="primary")
                
                # Cut Planes
                vuetify.VCardSubtitle(
                    "7. Cut Planes (Optional)", 
                    classes="text-h6 pa-2 font-weight-bold text-primary"
                )

                with vuetify.VRow(align="center", dense=True):
                    with vuetify.VCol(cols=1):
                        vuetify.VCheckbox(v_model=("xy_plane_visible", False))
                    with vuetify.VCol(cols=11):
                        vuetify.VSlider(
                            v_model=("z_slider", 1),
                            min=("z_slider_min", 0),
                            max=("z_slider_max", 100),
                            step=1,
                            label="XY plane",
                            thumb_label="always",
                            density="compact",
                        )

                with vuetify.VRow(align="center", dense=True):
                    with vuetify.VCol(cols=1):
                        vuetify.VCheckbox(v_model=("xz_plane_visible", False))
                    with vuetify.VCol(cols=11):
                        vuetify.VSlider(
                            v_model=("y_slider", 1),
                            min=("y_slider_min", 0),
                            max=("y_slider_max", 100),
                            step=1,
                            label="XZ plane",
                            thumb_label="always",
                            density="compact",
                        )

                with vuetify.VRow(align="center", dense=True):
                    with vuetify.VCol(cols=1):
                        vuetify.VCheckbox(v_model=("yz_plane_visible", False))
                    with vuetify.VCol(cols=11):
                        vuetify.VSlider(
                            v_model=("x_slider", 1),
                            min=("x_slider_min", 0),
                            max=("x_slider_max", 100),
                            step=1,
                            label="YZ plane",
                            thumb_label="always",
                            density="compact",
                        )

                vuetify.VRangeSlider(
                    v_model=("scalar_range", [0, 100]),
                    min=("iso_slider_min", 0),
                    max=("iso_slider_max", 100),
                    label="Scalar Range",
                    thumb_label="always",
                    density="compact",
                )


        # Main content area
        with layout.content:
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                # Single view mode
                with vuetify.VRow(classes="fill-height ma-0"):
                    # 3D View - dynamically resizes based on histogram visibility
                    with vuetify.VCol(cols=("histogram_cols", 12), classes="pa-0"):
                        view = VtkRemoteView(render_window)
                        ctrl.on_server_ready.add(view.update)
                        ctrl.view_update = view.update
                        ctrl.view_reset_camera = view.reset_camera
                    
                    # 2D Histogram View - Shows on the right when checkbox is enabled
                    with vuetify.VCol(cols=4, classes="pa-2", v_show="show_2d_histogram"):
                        with vuetify.VCard(elevation=4):
                            vuetify.VCardTitle("2D Histogram (Intensity vs Gradient)")
                            vuetify.VCardSubtitle("Cyan lines show cluster boundaries")
                            with vuetify.VCardText():
                                html.Img(
                                    src=("histogram_image", ""),
                                    style="width: 100%; height: auto;",
                                )

        # Update histogram column sizing based on checkbox
        @state.change("show_2d_histogram")
        def update_histogram_cols(show_2d_histogram, **kwargs):
            state.histogram_cols = 8 if show_2d_histogram else 12

if __name__ == "__main__":
    setup_ui()
    print("\n" + "="*60)
    print("2D Transfer Function Volume Rendering Application")
    print("="*60)
    print(f"Scikit-learn available: {SKLEARN_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print("Starting server on port 7654...")
    print("="*60 + "\n")
    server.start(port=7654)
